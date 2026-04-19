from __future__ import annotations

from pathlib import Path

from sva_toolkit.data import BenchmarkResult, BenchmarkRunner, RelationshipType, SingleResult
from sva_toolkit.runtime.llm import LLMConfig


class _FakeLLMClient:
    def __init__(self, responses: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.config = LLMConfig(model="fake-model", api_key="secret")
        self._responses = list(responses or [])
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return "assert property (@(posedge clk) req |-> ack);"


class _FakeFormalService:
    def __init__(self, relationships: list[tuple[bool, bool]] | None = None, *, error: Exception | None = None) -> None:
        self._relationships = list(relationships or [])
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def get_relationship(self, generated_sva: str, reference_sva: str) -> tuple[bool, bool]:
        self.calls.append((generated_sva, reference_sva))
        if self._error is not None:
            raise self._error
        if self._relationships:
            return self._relationships.pop(0)
        return (False, False)


def test_single_result_can_store_relationship() -> None:
    result = SingleResult(
        svad="Request must be acknowledged",
        reference_sva="assert property (@(posedge clk) req |-> ack);",
        generated_sva="assert property (@(posedge clk) req |-> ack);",
        relationship=RelationshipType.EQUIVALENT,
    )

    assert result.relationship is RelationshipType.EQUIVALENT


def test_benchmark_result_rates_are_derived_from_counts() -> None:
    result = BenchmarkResult(
        model_name="fake-model",
        total_items=10,
        equivalent_count=4,
        generated_implies_reference_count=2,
        reference_implies_generated_count=1,
        error_count=1,
    )

    assert result.equivalent_rate == 0.4
    assert result.any_implication_rate == 0.7
    assert result.success_rate == 0.9


def test_benchmark_runner_classifies_relationships() -> None:
    runner = BenchmarkRunner(
        llm_clients=[_FakeLLMClient()],
        formal_service=_FakeFormalService([(True, False), (False, True), (False, False)]),
        cache_dir=Path("/tmp") / "unused-cache",
    )

    assert runner.evaluate_relationship("g1", "r1") is RelationshipType.GENERATED_IMPLIES_REFERENCE
    assert runner.evaluate_relationship("g2", "r2") is RelationshipType.REFERENCE_IMPLIES_GENERATED
    assert runner.evaluate_relationship("g3", "r3") is RelationshipType.NO_RELATIONSHIP


def test_benchmark_runner_runs_dataset_and_aggregates_counts(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        llm_clients=[_FakeLLMClient(["```systemverilog\nassert property (@(posedge clk) req |-> ack);\n```"])],
        formal_service=_FakeFormalService([(True, True)]),
        cache_dir=tmp_path / "cache",
    )

    result = runner.run_benchmark(
        [{"SVAD": "Acknowledge a request", "SVA": "assert property (@(posedge clk) req |-> ack);"}],
        runner.llm_clients[0],
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    assert result.model_name == "fake-model"
    assert result.total_items == 1
    assert result.equivalent_count == 1
    assert result.error_count == 0
    assert result.individual_results[0].generated_sva == "assert property (@(posedge clk) req |-> ack);"


def test_benchmark_runner_uses_cache_without_repeating_llm_or_formal_calls(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    dataset = [{"SVAD": "Acknowledge a request", "SVA": "assert property (@(posedge clk) req |-> ack);"}]
    first_client = _FakeLLMClient(["assert property (@(posedge clk) req |-> ack);"])
    first_formal = _FakeFormalService([(True, True)])
    first_runner = BenchmarkRunner(
        llm_clients=[first_client],
        formal_service=first_formal,
        cache_dir=cache_dir,
    )

    first_runner.run_benchmark(
        dataset,
        first_client,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    second_client = _FakeLLMClient(error=AssertionError("cache was not used"))
    second_formal = _FakeFormalService(error=AssertionError("cache was not used"))
    second_runner = BenchmarkRunner(
        llm_clients=[second_client],
        formal_service=second_formal,
        cache_dir=cache_dir,
    )
    result = second_runner.run_benchmark(
        dataset,
        second_client,
        use_multiprocessing=False,
        rate_limit_delay=0,
    )

    assert result.equivalent_count == 1
    assert second_client.calls == []
    assert second_formal.calls == []


def test_benchmark_runner_reports_generation_or_formal_errors(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        llm_clients=[_FakeLLMClient(error=RuntimeError("llm unavailable"))],
        formal_service=_FakeFormalService(),
        cache_dir=tmp_path / "cache",
    )

    result = runner.run_single(
        runner.llm_clients[0],
        "Acknowledge a request",
        "assert property (@(posedge clk) req |-> ack);",
    )

    assert result.relationship is RelationshipType.ERROR
    assert "llm unavailable" in (result.error_message or "")
