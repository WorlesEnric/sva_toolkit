from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

from sva_toolkit.formal import FormalService
from sva_toolkit.runtime.llm import LLMClient, LLMConfig


class RelationshipType(str, Enum):
    EQUIVALENT = "equivalent"
    GENERATED_IMPLIES_REFERENCE = "generated_implies_reference"
    REFERENCE_IMPLIES_GENERATED = "reference_implies_generated"
    NO_RELATIONSHIP = "no_relationship"
    ERROR = "error"


def _benchmark_cache_key(svad: str, reference_sva: str, model: str) -> str:
    payload = json.dumps(
        {"svad": svad, "reference_sva": reference_sva, "model": model},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_result(cache_dir: str | None, cache_key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cached["from_cache"] = True
    return cached


def _write_cached_result(cache_dir: str | None, cache_key: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    try:
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return


def _serialize_llm_config(llm_client: object) -> dict[str, Any]:
    config = llm_client.config
    return {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }


def _clean_sva_output(response: str) -> str:
    response = response.strip()
    fenced_blocks = re.findall(r"```(?:systemverilog|verilog|sv)?\s*\n?(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        response = fenced_blocks[0].strip()
    response = re.sub(r"/\*.*?\*/", "", response, flags=re.DOTALL)
    cleaned_lines: list[str] = []
    for line in response.splitlines():
        line = re.sub(r"//.*$", "", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _classify_relationship(relationship: tuple[bool, bool]) -> RelationshipType:
    generated_implies_reference, reference_implies_generated = relationship
    if generated_implies_reference and reference_implies_generated:
        return RelationshipType.EQUIVALENT
    if generated_implies_reference:
        return RelationshipType.GENERATED_IMPLIES_REFERENCE
    if reference_implies_generated:
        return RelationshipType.REFERENCE_IMPLIES_GENERATED
    return RelationshipType.NO_RELATIONSHIP


def _process_benchmark_item(
    item_data: dict[str, Any],
    *,
    llm_client: object | None = None,
    llm_config_dict: dict[str, Any] | None = None,
    formal_service: object | None = None,
    formal_service_kwargs: dict[str, Any] | None = None,
    system_prompt: str,
    user_prompt_template: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    active_llm_client = llm_client
    if active_llm_client is None and llm_config_dict is not None:
        active_llm_client = LLMClient(LLMConfig(**llm_config_dict))
    if active_llm_client is None:
        raise RuntimeError("BenchmarkRunner requires an LLM client for generation.")

    model = active_llm_client.config.model
    cache_key = _benchmark_cache_key(item_data["SVAD"], item_data["SVA"], model)
    cached = _load_cached_result(cache_dir, cache_key)
    if cached is not None:
        return cached

    active_formal_service = formal_service
    if active_formal_service is None:
        active_formal_service = FormalService(**(formal_service_kwargs or {}))

    prompt = user_prompt_template.format(svad=item_data["SVAD"])
    generation_started = time.time()
    try:
        response = active_llm_client.generate(system_prompt, prompt)
        generated_sva = _clean_sva_output(response)
        generation_error = None
    except Exception as exc:
        generated_sva = ""
        generation_error = f"Generation error: {exc}"
    generation_time = time.time() - generation_started

    verification_started = time.time()
    relationship = RelationshipType.ERROR
    error_message = generation_error
    if generation_error is None:
        try:
            pair = active_formal_service.get_relationship(generated_sva, item_data["SVA"])
            relationship = _classify_relationship(pair)
            error_message = None
        except Exception as exc:
            relationship = RelationshipType.ERROR
            error_message = f"Verification error: {exc}"
    verification_time = time.time() - verification_started

    result = {
        "index": item_data["index"],
        "svad": item_data["SVAD"],
        "reference_sva": item_data["SVA"],
        "generated_sva": generated_sva,
        "relationship": relationship.value,
        "cot": item_data.get("CoT"),
        "error_message": error_message,
        "generation_time": generation_time,
        "verification_time": verification_time,
        "from_cache": False,
    }
    _write_cached_result(cache_dir, cache_key, result)
    return result


def _worker_process_benchmark_item(
    item_data: dict[str, Any],
    llm_config_dict: dict[str, Any],
    formal_service_kwargs: dict[str, Any],
    system_prompt: str,
    user_prompt_template: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    return _process_benchmark_item(
        item_data,
        llm_config_dict=llm_config_dict,
        formal_service_kwargs=formal_service_kwargs,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        cache_dir=cache_dir,
    )


@dataclass
class SingleResult:
    svad: str
    reference_sva: str
    generated_sva: str
    relationship: RelationshipType
    cot: str | None = None
    error_message: str | None = None
    generation_time: float = 0.0
    verification_time: float = 0.0


@dataclass
class BenchmarkResult:
    model_name: str
    total_items: int
    equivalent_count: int = 0
    generated_implies_reference_count: int = 0
    reference_implies_generated_count: int = 0
    no_relationship_count: int = 0
    error_count: int = 0
    avg_generation_time: float = 0.0
    avg_verification_time: float = 0.0
    individual_results: list[SingleResult] = field(default_factory=list)

    @property
    def equivalent_rate(self) -> float:
        return self.equivalent_count / self.total_items if self.total_items else 0.0

    @property
    def any_implication_rate(self) -> float:
        numerator = (
            self.equivalent_count
            + self.generated_implies_reference_count
            + self.reference_implies_generated_count
        )
        return numerator / self.total_items if self.total_items else 0.0

    @property
    def success_rate(self) -> float:
        return (self.total_items - self.error_count) / self.total_items if self.total_items else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_items": self.total_items,
            "equivalent_count": self.equivalent_count,
            "generated_implies_reference_count": self.generated_implies_reference_count,
            "reference_implies_generated_count": self.reference_implies_generated_count,
            "no_relationship_count": self.no_relationship_count,
            "error_count": self.error_count,
            "equivalent_rate": self.equivalent_rate,
            "any_implication_rate": self.any_implication_rate,
            "success_rate": self.success_rate,
            "avg_generation_time": self.avg_generation_time,
            "avg_verification_time": self.avg_verification_time,
        }


class BenchmarkRunner:
    SVA_GENERATION_SYSTEM_PROMPT = (
        "You are an expert SystemVerilog Assertion engineer. Convert each requirement "
        "into complete, syntactically correct SVA code with no markdown or explanation."
    )
    SVA_GENERATION_USER_PROMPT_TEMPLATE = (
        "Translate this requirement into SystemVerilog Assertion code:\n\n{svad}"
    )

    def __init__(
        self,
        *,
        llm_clients: list[object],
        formal_service: object | None = None,
        num_workers: int = 4,
        cache_dir: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
    ) -> None:
        self.llm_clients = llm_clients
        self.formal_service = formal_service or FormalService()
        self.num_workers = num_workers
        self.system_prompt = system_prompt or self.SVA_GENERATION_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or self.SVA_GENERATION_USER_PROMPT_TEMPLATE
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="sva_benchmark_cache_")
        else:
            self.cache_dir = str(cache_dir)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self._formal_service_kwargs = self._infer_formal_service_kwargs(self.formal_service)

    @classmethod
    def from_configs(
        cls,
        llm_configs: list[LLMConfig | dict[str, Any]],
        *,
        formal_service: object | None = None,
        num_workers: int = 4,
        cache_dir: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
    ) -> "BenchmarkRunner":
        llm_clients: list[LLMClient] = []
        for config in llm_configs:
            materialized = LLMConfig(**config) if isinstance(config, dict) else config
            llm_clients.append(LLMClient(materialized))
        return cls(
            llm_clients=llm_clients,
            formal_service=formal_service,
            num_workers=num_workers,
            cache_dir=cache_dir,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )

    def generate_sva(self, llm_client: object, svad: str) -> str:
        response = llm_client.generate(self.system_prompt, self.user_prompt_template.format(svad=svad))
        return _clean_sva_output(response)

    def evaluate_relationship(self, generated_sva: str, reference_sva: str) -> RelationshipType:
        relationship = self.formal_service.get_relationship(generated_sva, reference_sva)
        return _classify_relationship(relationship)

    def run_single(
        self,
        llm_client: object,
        svad: str,
        reference_sva: str,
        cot: str | None = None,
    ) -> SingleResult:
        result = _process_benchmark_item(
            {"index": 0, "SVAD": svad, "SVA": reference_sva, "CoT": cot},
            llm_client=llm_client,
            formal_service=self.formal_service,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            cache_dir=None,
        )
        return SingleResult(
            svad=result["svad"],
            reference_sva=result["reference_sva"],
            generated_sva=result["generated_sva"],
            relationship=RelationshipType(result["relationship"]),
            cot=result.get("cot"),
            error_message=result.get("error_message"),
            generation_time=result["generation_time"],
            verification_time=result["verification_time"],
        )

    def run_benchmark(
        self,
        dataset: list[dict[str, Any]],
        llm_client: object,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> BenchmarkResult:
        items = [
            {"index": index, "SVAD": item["SVAD"], "SVA": item["SVA"], "CoT": item.get("CoT")}
            for index, item in enumerate(dataset)
            if item.get("SVAD") and item.get("SVA")
        ]
        total = len(items)
        model_name = llm_client.config.model
        if total == 0:
            return BenchmarkResult(model_name=model_name, total_items=0)

        llm_config_dict = _serialize_llm_config(llm_client)
        if use_multiprocessing and self.num_workers > 1 and total > 1:
            results = self._run_multiprocess(
                items,
                llm_config_dict=llm_config_dict,
                progress_callback=progress_callback,
                total=total,
            )
        else:
            results = self._run_single_process(
                items,
                llm_client=llm_client,
                progress_callback=progress_callback,
                total=total,
                rate_limit_delay=rate_limit_delay,
            )

        single_results = [
            SingleResult(
                svad=result["svad"],
                reference_sva=result["reference_sva"],
                generated_sva=result["generated_sva"],
                relationship=RelationshipType(result["relationship"]),
                cot=result.get("cot"),
                error_message=result.get("error_message"),
                generation_time=result["generation_time"],
                verification_time=result["verification_time"],
            )
            for result in results
        ]
        benchmark = BenchmarkResult(
            model_name=model_name,
            total_items=len(single_results),
            individual_results=single_results,
        )
        for item in single_results:
            if item.relationship is RelationshipType.EQUIVALENT:
                benchmark.equivalent_count += 1
            elif item.relationship is RelationshipType.GENERATED_IMPLIES_REFERENCE:
                benchmark.generated_implies_reference_count += 1
            elif item.relationship is RelationshipType.REFERENCE_IMPLIES_GENERATED:
                benchmark.reference_implies_generated_count += 1
            elif item.relationship is RelationshipType.NO_RELATIONSHIP:
                benchmark.no_relationship_count += 1
            else:
                benchmark.error_count += 1
        if single_results:
            benchmark.avg_generation_time = sum(item.generation_time for item in single_results) / len(single_results)
            benchmark.avg_verification_time = sum(item.verification_time for item in single_results) / len(single_results)
        return benchmark

    def run_all_benchmarks(
        self,
        dataset: list[dict[str, Any]],
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        rate_limit_delay: float = 0.5,
        use_multiprocessing: bool = True,
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for llm_client in self.llm_clients:
            callback = None
            if progress_callback is not None:
                model_name = llm_client.config.model

                def callback(current: int, total: int, *, _model_name: str = model_name) -> None:
                    progress_callback(_model_name, current, total)

            results.append(
                self.run_benchmark(
                    dataset,
                    llm_client,
                    progress_callback=callback,
                    rate_limit_delay=rate_limit_delay,
                    use_multiprocessing=use_multiprocessing,
                )
            )
        return results

    @staticmethod
    def compare_results(results: list[BenchmarkResult]) -> dict[str, Any]:
        if not results:
            return {}
        best_equivalent = max(results, key=lambda item: item.equivalent_rate)
        best_any_implication = max(results, key=lambda item: item.any_implication_rate)
        return {
            "models": [result.model_name for result in results],
            "equivalent_rates": [result.equivalent_rate for result in results],
            "any_implication_rates": [result.any_implication_rate for result in results],
            "success_rates": [result.success_rate for result in results],
            "avg_generation_times": [result.avg_generation_time for result in results],
            "best_equivalent_model": best_equivalent.model_name,
            "best_equivalent_rate": best_equivalent.equivalent_rate,
            "best_any_implication_model": best_any_implication.model_name,
            "best_any_implication_rate": best_any_implication.any_implication_rate,
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

    def _run_multiprocess(
        self,
        items: list[dict[str, Any]],
        *,
        llm_config_dict: dict[str, Any],
        progress_callback: Callable[[int, int], None] | None,
        total: int,
    ) -> list[dict[str, Any]]:
        worker = _BenchmarkWorker(
            llm_config_dict=llm_config_dict,
            formal_service_kwargs=self._formal_service_kwargs,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            cache_dir=self.cache_dir,
        )
        results: list[dict[str, Any]] = []
        completed = 0
        with Pool(processes=self.num_workers) as pool:
            for result in pool.imap_unordered(worker, items):
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
        llm_client: object,
        progress_callback: Callable[[int, int], None] | None,
        total: int,
        rate_limit_delay: float,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            result = _process_benchmark_item(
                item,
                llm_client=llm_client,
                formal_service=self.formal_service,
                system_prompt=self.system_prompt,
                user_prompt_template=self.user_prompt_template,
                cache_dir=self.cache_dir,
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(index, total)
            if not result.get("from_cache", False):
                time.sleep(rate_limit_delay)
        return results

    def _infer_formal_service_kwargs(self, formal_service: object) -> dict[str, Any]:
        if not isinstance(formal_service, FormalService):
            return {}
        return {
            "backend": formal_service.backend_name,
            "timeout": formal_service.timeout,
            "depth": formal_service.depth,
            "verbose": formal_service.verbose,
            "keep_files": formal_service.keep_files,
        }


@dataclass(frozen=True)
class _BenchmarkWorker:
    llm_config_dict: dict[str, Any]
    formal_service_kwargs: dict[str, Any]
    system_prompt: str
    user_prompt_template: str
    cache_dir: str | None

    def __call__(self, item_data: dict[str, Any]) -> dict[str, Any]:
        return _worker_process_benchmark_item(
            item_data,
            llm_config_dict=self.llm_config_dict,
            formal_service_kwargs=self.formal_service_kwargs,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            cache_dir=self.cache_dir,
        )
