# Data Workflows

## Purpose

`sva_toolkit.data` packages the V3 dataset builder and benchmark runner. It combines parser-driven description generation, optional LLM calls, caching, multiprocessing, and formal relationship checks for dataset construction and evaluation workflows.

## CLI Commands

Build a dataset offline:

```bash
sva data build examples/data/dataset_input.json -o examples/out/dataset.jsonl --workers 1
```

Build a dataset with LLM-backed SVAD generation:

```bash
OPENAI_API_KEY=... sva data build examples/data/dataset_input.json -o examples/out/dataset.jsonl --model gpt-4o-mini --workers 1
```

Benchmark a model against `(SVAD, SVA)` pairs:

```bash
OPENAI_API_KEY=... sva data benchmark examples/data/benchmark_input.json --model gpt-4o-mini --workers 1 -o examples/out/benchmark_results.json
```

Current CLI behavior:

- `sva data build` works without `--model`; in that case SVAD generation is skipped and CoT is still produced.
- `sva data benchmark` always needs a model name, either from `--model` or `SVA_TOOLKIT_MODEL`.
- The CLI uses `OPENAI_API_KEY` for authentication. Custom `base_url` settings are available in the Python API today, not yet via CLI flags.

## Usage Examples

Build entries from Python:

```python
from sva_toolkit.data import DatasetBuilder

builder = DatasetBuilder(num_workers=1)
entries = builder.build_dataset(
    [{"SVA": "assert property (@(posedge clk) req |-> ##1 ack);"}],
    generate_svad=True,
    generate_cot=True,
    use_multiprocessing=False,
    rate_limit_delay=0,
)
```

Use the LLM-configured constructor:

```python
from sva_toolkit.data import DatasetBuilder
from sva_toolkit.runtime.llm import LLMConfig

builder = DatasetBuilder.from_llm_config(
    LLMConfig(model="gpt-4o-mini", api_key="placeholder"),
    num_workers=1,
)
```

Run a benchmark:

```python
from sva_toolkit.data import BenchmarkRunner
from sva_toolkit.runtime.llm import LLMConfig

runner = BenchmarkRunner.from_configs([LLMConfig(model="gpt-4o-mini", api_key="placeholder")], num_workers=1)
```

## API Reference

Dataset APIs:

- `DatasetEntry`
- `DatasetBuilder`
- `DatasetBuilder.from_llm_config(...)`
- `DatasetBuilder.generate_svad(sva_code: str) -> str`
- `DatasetBuilder.generate_cot(sva_code: str) -> str`
- `DatasetBuilder.process_entry(entry, ...) -> DatasetEntry`
- `DatasetBuilder.build_dataset(input_data, ...) -> list[DatasetEntry]`
- `DatasetBuilder.build_from_file(input_path, output_path, ...) -> list[DatasetEntry]`
- `DatasetBuilder.validate_dataset(entries) -> dict[str, object]`
- `DatasetBuilder.get_cache_stats() -> dict[str, object]`
- `DatasetBuilder.clear_cache() -> int`

Benchmark APIs:

- `RelationshipType`
- `SingleResult`
- `BenchmarkResult`
- `BenchmarkRunner`
- `BenchmarkRunner.from_configs(...)`
- `BenchmarkRunner.generate_sva(llm_client, svad: str) -> str`
- `BenchmarkRunner.evaluate_relationship(generated_sva: str, reference_sva: str) -> RelationshipType`
- `BenchmarkRunner.run_single(...) -> SingleResult`
- `BenchmarkRunner.run_benchmark(dataset, llm_client, ...) -> BenchmarkResult`
- `BenchmarkRunner.run_all_benchmarks(dataset, ...) -> list[BenchmarkResult]`
- `BenchmarkRunner.compare_results(results) -> dict[str, object]`

## Operational Notes

- Both dataset and benchmark flows use per-item JSON cache files.
- Offline dataset builds are safe for local documentation/demo workflows.
- Benchmarks depend on both an LLM client and at least one formal backend for meaningful semantic evaluation.
- Translator fallback is available when LLM-backed SVAD generation fails during dataset building.

## Related Docs

- [Formal verification](sva-formal.md)
- [Description engine](sva-describe.md)
- [Examples](../examples/README.md)
