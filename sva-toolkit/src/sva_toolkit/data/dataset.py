from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from sva_toolkit.data.cache import load_cached_result as _shared_load_cached_result
from sva_toolkit.data.cache import write_cached_result as _shared_write_cached_result
from sva_toolkit.describe import SVACoTBuilder, SVADTranslator
from sva_toolkit.runtime.diagnostics import DEFAULT_DIAGNOSTICS, LOGGER, Diagnostics
from sva_toolkit.runtime.llm import LLMClient, LLMConfig


def _dataset_cache_key(sva_code: str, model: str, *, generate_svad: bool, generate_cot: bool) -> str:
    payload = json.dumps(
        {
            "sva_code": sva_code,
            "model": model,
            "generate_svad": generate_svad,
            "generate_cot": generate_cot,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_result(cache_dir: str | None, cache_key: str) -> dict[str, Any] | None:
    return _shared_load_cached_result(cache_dir, cache_key)


def _write_cached_result(cache_dir: str | None, cache_key: str, payload: dict[str, Any]) -> None:
    _shared_write_cached_result(cache_dir, cache_key, payload)


def _serialize_llm_config(llm_client: object | None) -> dict[str, Any] | None:
    config = getattr(llm_client, "config", None)
    if config is None:
        return None
    return {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "max_retries": getattr(config, "max_retries", 3),
        "backoff_base": getattr(config, "backoff_base", 1.0),
        "backoff_cap": getattr(config, "backoff_cap", 30.0),
        "jitter": getattr(config, "jitter", True),
    }


def _materialize_llm_client(
    llm_client: object | None,
    llm_config_dict: dict[str, Any] | None,
) -> object | None:
    if llm_client is not None:
        return llm_client
    if llm_config_dict is None:
        return None
    return LLMClient(LLMConfig(**llm_config_dict))


def _process_dataset_item(
    item_data: dict[str, Any],
    *,
    llm_client: object | None = None,
    llm_config_dict: dict[str, Any] | None = None,
    cot_builder: SVACoTBuilder | None = None,
    translator: SVADTranslator | None = None,
    generate_svad: bool,
    generate_cot: bool,
    system_prompt: str,
    user_prompt_template: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    model = "offline"
    if llm_client is not None and getattr(llm_client, "config", None) is not None:
        model = llm_client.config.model
    elif llm_config_dict is not None:
        model = str(llm_config_dict["model"])

    cache_key = _dataset_cache_key(
        item_data["SVA"],
        model,
        generate_svad=generate_svad,
        generate_cot=generate_cot,
    )
    cached = _load_cached_result(cache_dir, cache_key)
    if cached is not None:
        return cached

    active_llm_client = _materialize_llm_client(llm_client, llm_config_dict)
    active_cot_builder = cot_builder or SVACoTBuilder()
    active_translator = translator or SVADTranslator()

    metadata = dict(item_data.get("metadata", {}))
    result: dict[str, Any] = {
        "index": item_data["index"],
        "SVA": item_data["SVA"],
        "SVAD": item_data.get("SVAD"),
        "CoT": item_data.get("CoT"),
        "metadata": metadata,
        "extra_fields": dict(item_data.get("extra_fields", {})),
        "from_cache": False,
    }

    if generate_svad and not result["SVAD"]:
        if active_llm_client is None:
            metadata["svad_skipped"] = "offline_mode"
        else:
            user_prompt = user_prompt_template.format(sva_code=item_data["SVA"])
            try:
                result["SVAD"] = active_llm_client.generate(system_prompt, user_prompt)
                metadata["svad_source"] = "llm"
            except Exception as exc:
                metadata["svad_error"] = str(exc)
                try:
                    result["SVAD"] = active_translator.translate(item_data["SVA"])
                    metadata["svad_source"] = "translator_fallback"
                except Exception as translator_exc:
                    metadata["svad_fallback_error"] = str(translator_exc)

    if generate_cot and not result["CoT"]:
        try:
            result["CoT"] = active_cot_builder.build(item_data["SVA"])
        except Exception as exc:
            metadata["cot_error"] = str(exc)

    _write_cached_result(cache_dir, cache_key, result)
    return result


def _surface_dataset_diagnostics(result: dict[str, Any], diagnostics: Diagnostics) -> None:
    metadata = result.get("metadata", {})
    if metadata.get("svad_source") != "translator_fallback":
        return

    diagnostics.record("translator_fallback", detail=str(metadata.get("svad_error", "")))
    detail = metadata.get("svad_error")
    suffix = f": {detail}" if detail else ""
    LOGGER.warning("dataset entry %s used translator fallback%s", result.get("index"), suffix)


def _worker_process_dataset_item(
    item_data: dict[str, Any],
    llm_config_dict: dict[str, Any] | None,
    generate_svad: bool,
    generate_cot: bool,
    system_prompt: str,
    user_prompt_template: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    return _process_dataset_item(
        item_data,
        llm_config_dict=llm_config_dict,
        generate_svad=generate_svad,
        generate_cot=generate_cot,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        cache_dir=cache_dir,
    )


@dataclass
class DatasetEntry:
    SVA: str
    SVAD: str | None = None
    CoT: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra_fields)
        payload["SVA"] = self.SVA
        if self.SVAD is not None:
            payload["SVAD"] = self.SVAD
        if self.CoT is not None:
            payload["CoT"] = self.CoT
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class DatasetBuilder:
    SVAD_SYSTEM_PROMPT = (
        "You are an expert in SystemVerilog Assertions. Describe the intent, trigger, "
        "expected behavior, timing, and reset/disable semantics concisely."
    )
    SVAD_USER_PROMPT_TEMPLATE = (
        "Generate a concise natural-language description for this SystemVerilog Assertion.\n\n"
        "```systemverilog\n{sva_code}\n```"
    )

    def __init__(
        self,
        *,
        llm_client: object | None = None,
        cot_builder: SVACoTBuilder | None = None,
        translator: SVADTranslator | None = None,
        num_workers: int = 4,
        cache_dir: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        diagnostics: Diagnostics | None = None,
    ) -> None:
        self.diagnostics = diagnostics or DEFAULT_DIAGNOSTICS
        self.llm_client = llm_client
        if self.llm_client is not None and hasattr(self.llm_client, "diagnostics"):
            self.llm_client.diagnostics = self.diagnostics
        self.cot_builder = cot_builder or SVACoTBuilder()
        self.translator = translator or SVADTranslator()
        self.num_workers = num_workers
        self.system_prompt = system_prompt or self.SVAD_SYSTEM_PROMPT
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="sva_dataset_cache_")
        else:
            self.cache_dir = str(cache_dir)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_llm_config(
        cls,
        config: LLMConfig,
        *,
        cot_builder: SVACoTBuilder | None = None,
        translator: SVADTranslator | None = None,
        num_workers: int = 4,
        cache_dir: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        diagnostics: Diagnostics | None = None,
    ) -> "DatasetBuilder":
        return cls(
            llm_client=LLMClient(config, diagnostics=diagnostics),
            cot_builder=cot_builder,
            translator=translator,
            num_workers=num_workers,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            diagnostics=diagnostics,
        )

    def generate_svad(self, sva_code: str) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client not configured. Cannot generate SVAD.")
        return self.llm_client.generate(
            self.system_prompt,
            self.SVAD_USER_PROMPT_TEMPLATE.format(sva_code=sva_code),
        )

    def generate_cot(self, sva_code: str) -> str:
        return self.cot_builder.build(sva_code)

    def process_entry(
        self,
        entry: DatasetEntry,
        *,
        generate_svad: bool = True,
        generate_cot: bool = True,
    ) -> DatasetEntry:
        result = _process_dataset_item(
            {
                "index": 0,
                "SVA": entry.SVA,
                "SVAD": entry.SVAD,
                "CoT": entry.CoT,
                "metadata": entry.metadata,
                "extra_fields": entry.extra_fields,
            },
            llm_client=self.llm_client,
            cot_builder=self.cot_builder,
            translator=self.translator,
            generate_svad=generate_svad,
            generate_cot=generate_cot,
            system_prompt=self.system_prompt,
            user_prompt_template=self.SVAD_USER_PROMPT_TEMPLATE,
            cache_dir=self.cache_dir,
        )
        _surface_dataset_diagnostics(result, self.diagnostics)
        return DatasetEntry(
            SVA=result["SVA"],
            SVAD=result.get("SVAD"),
            CoT=result.get("CoT"),
            metadata=result.get("metadata", {}),
            extra_fields=result.get("extra_fields", {}),
        )

    def build_dataset(
        self,
        input_data: list[dict[str, Any]],
        *,
        generate_svad: bool = True,
        generate_cot: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> list[DatasetEntry]:
        items = self._prepare_items(input_data)
        total = len(items)
        if total == 0:
            return []

        llm_config_dict = _serialize_llm_config(self.llm_client) if generate_svad else None
        if use_multiprocessing and self.num_workers > 1 and total > 1:
            results = self._run_multiprocess(
                items,
                llm_config_dict=llm_config_dict,
                generate_svad=generate_svad,
                generate_cot=generate_cot,
                progress_callback=progress_callback,
                total=total,
            )
        else:
            results = self._run_single_process(
                items,
                llm_config_dict=llm_config_dict,
                generate_svad=generate_svad,
                generate_cot=generate_cot,
                progress_callback=progress_callback,
                total=total,
                rate_limit_delay=rate_limit_delay,
            )

        return [
            DatasetEntry(
                SVA=result["SVA"],
                SVAD=result.get("SVAD"),
                CoT=result.get("CoT"),
                metadata=result.get("metadata", {}),
                extra_fields=result.get("extra_fields", {}),
            )
            for result in results
        ]

    def build_from_file(
        self,
        *,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        generate_svad: bool = True,
        generate_cot: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> list[DatasetEntry]:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        entries = self.build_dataset(
            payload,
            generate_svad=generate_svad,
            generate_cot=generate_cot,
            progress_callback=progress_callback,
            rate_limit_delay=rate_limit_delay,
            use_multiprocessing=use_multiprocessing,
        )
        Path(output_path).write_text(
            json.dumps([entry.to_dict() for entry in entries], indent=2),
            encoding="utf-8",
        )
        return entries

    def validate_dataset(self, entries: list[DatasetEntry]) -> dict[str, Any]:
        total = len(entries)
        svad_count = sum(1 for entry in entries if entry.SVAD)
        cot_count = sum(1 for entry in entries if entry.CoT)
        error_count = sum(
            1
            for entry in entries
            if any(key.endswith("_error") for key in entry.metadata)
        )
        return {
            "total_entries": total,
            "entries_with_svad": svad_count,
            "entries_with_cot": cot_count,
            "entries_with_errors": error_count,
            "svad_coverage": svad_count / total if total else 0.0,
            "cot_coverage": cot_count / total if total else 0.0,
        }

    def get_cache_stats(self) -> dict[str, Any]:
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return {"cached_items": 0, "cache_dir": self.cache_dir}
        return {
            "cached_items": len(list(cache_path.glob("*.json"))),
            "cache_dir": self.cache_dir,
        }

    def clear_cache(self) -> int:
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return 0
        count = 0
        for path in cache_path.glob("*.json"):
            path.unlink()
            count += 1
        return count

    def _prepare_items(self, input_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, item in enumerate(input_data):
            sva_code = item.get("SVA") or item.get("sva")
            if not sva_code:
                continue
            extra_fields = {
                key: value
                for key, value in item.items()
                if key not in {"SVA", "sva", "SVAD", "CoT", "metadata"}
            }
            items.append(
                {
                    "index": index,
                    "SVA": sva_code,
                    "SVAD": item.get("SVAD"),
                    "CoT": item.get("CoT"),
                    "metadata": dict(item.get("metadata", {})),
                    "extra_fields": extra_fields,
                }
            )
        return items

    def _run_multiprocess(
        self,
        items: list[dict[str, Any]],
        *,
        llm_config_dict: dict[str, Any] | None,
        generate_svad: bool,
        generate_cot: bool,
        progress_callback: Callable[[int, int], None] | None,
        total: int,
    ) -> list[dict[str, Any]]:
        worker = _DatasetWorker(
            llm_config_dict=llm_config_dict,
            generate_svad=generate_svad,
            generate_cot=generate_cot,
            system_prompt=self.system_prompt,
            user_prompt_template=self.SVAD_USER_PROMPT_TEMPLATE,
            cache_dir=self.cache_dir,
        )
        results: list[dict[str, Any]] = []
        completed = 0
        with Pool(processes=self.num_workers) as pool:
            for result in pool.imap_unordered(worker, items):
                _surface_dataset_diagnostics(result, self.diagnostics)
                results.append(result)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
        results.sort(key=lambda item: item["index"])
        return results

    def _run_single_process(
        self,
        items: list[dict[str, Any]],
        *,
        llm_config_dict: dict[str, Any] | None,
        generate_svad: bool,
        generate_cot: bool,
        progress_callback: Callable[[int, int], None] | None,
        total: int,
        rate_limit_delay: float,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            result = _process_dataset_item(
                item,
                llm_client=self.llm_client,
                llm_config_dict=llm_config_dict,
                cot_builder=self.cot_builder,
                translator=self.translator,
                generate_svad=generate_svad,
                generate_cot=generate_cot,
                system_prompt=self.system_prompt,
                user_prompt_template=self.SVAD_USER_PROMPT_TEMPLATE,
                cache_dir=self.cache_dir,
            )
            _surface_dataset_diagnostics(result, self.diagnostics)
            results.append(result)
            if progress_callback is not None:
                progress_callback(index, total)
            if generate_svad and self.llm_client is not None and not result.get("from_cache", False):
                time.sleep(rate_limit_delay)
        return results


@dataclass(frozen=True)
class _DatasetWorker:
    llm_config_dict: dict[str, Any] | None
    generate_svad: bool
    generate_cot: bool
    system_prompt: str
    user_prompt_template: str
    cache_dir: str | None

    def __call__(self, item_data: dict[str, Any]) -> dict[str, Any]:
        return _worker_process_dataset_item(
            item_data,
            llm_config_dict=self.llm_config_dict,
            generate_svad=self.generate_svad,
            generate_cot=self.generate_cot,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            cache_dir=self.cache_dir,
        )
