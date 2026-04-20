from __future__ import annotations

import json
from multiprocessing import get_context
from pathlib import Path

import sva_toolkit.data.cache as cache_module
import sva_toolkit.data.dataset as dataset_module
from sva_toolkit.data import DatasetBuilder
from sva_toolkit.runtime.llm import LLMConfig


class _FakeLLMClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.config = LLMConfig(model="fake-model", api_key="secret")
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.pop(0)
        return "Generated SVAD"


def _build_dataset_in_worker(cache_dir: str, items: list[dict[str, str]], start_event: object) -> None:
    if not start_event.wait(10):
        raise TimeoutError("cache-locking worker did not receive the start signal")

    builder = DatasetBuilder(cache_dir=cache_dir, num_workers=1)
    builder.build_dataset(
        items,
        generate_svad=False,
        generate_cot=False,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )


def test_dataset_cache_survives_parallel_process_writes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    items = [
        {"SVA": f"assert property (@(posedge clk) req_{index} |-> ack_{index});"}
        for index in range(64)
    ]

    context = get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(target=_build_dataset_in_worker, args=(str(cache_dir), items, start_event))
        for _ in range(4)
    ]

    for process in processes:
        process.start()

    start_event.set()

    for process in processes:
        process.join(30)

    still_running = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

    assert still_running == []
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]

    cache_files = sorted(cache_dir.glob("*.json"))
    assert len(cache_files) == 64
    for cache_file in cache_files:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        assert payload["__cache_schema"] == cache_module.CACHE_SCHEMA_VERSION


def test_dataset_cache_ignores_legacy_entries_without_schema(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sva_code = "assert property (@(posedge clk) req |-> ack);"
    cache_key = dataset_module._dataset_cache_key(
        sva_code,
        "fake-model",
        generate_svad=True,
        generate_cot=False,
    )
    cache_path = cache_dir / f"{cache_key}.json"
    cache_path.write_text(
        json.dumps(
            {
                "SVA": sva_code,
                "SVAD": "stale cache entry",
                "metadata": {"svad_source": "legacy"},
            }
        ),
        encoding="utf-8",
    )

    llm_client = _FakeLLMClient(["Fresh SVAD"])
    builder = DatasetBuilder(llm_client=llm_client, cache_dir=cache_dir)
    entries = builder.build_dataset(
        [{"SVA": sva_code}],
        generate_svad=True,
        generate_cot=False,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    assert entries[0].SVAD == "Fresh SVAD"
    assert len(llm_client.calls) == 1
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["__cache_schema"] == cache_module.CACHE_SCHEMA_VERSION
    assert payload["SVAD"] == "Fresh SVAD"


def test_write_cached_result_uses_atomic_write_json(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_atomic_write_json(path: str | Path, payload: object, *, indent: int = 2, sort_keys: bool = True) -> None:
        captured["path"] = Path(path)
        captured["payload"] = payload
        captured["indent"] = indent
        captured["sort_keys"] = sort_keys

    monkeypatch.setattr(cache_module, "atomic_write_json", fake_atomic_write_json)

    dataset_module._write_cached_result(
        str(tmp_path),
        "cache-key",
        {
            "index": 0,
            "SVA": "assert property (@(posedge clk) req |-> ack);",
            "metadata": {},
            "from_cache": False,
        },
    )

    assert captured["path"] == tmp_path / "cache-key.json"
    assert captured["indent"] == 2
    assert captured["sort_keys"] is True
    assert captured["payload"]["__cache_schema"] == cache_module.CACHE_SCHEMA_VERSION
