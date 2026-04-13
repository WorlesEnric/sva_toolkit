# SVA Benchmark Runner (`sva-benchmark`)

## Overview

The SVA Benchmark Runner is a comprehensive evaluation framework designed to assess the performance of large language models (LLMs) on the task of SystemVerilog Assertion generation from natural language descriptions. The tool orchestrates end-to-end benchmarking workflows that include LLM inference, syntax validation, and formal semantic verification, producing quantitative metrics for model comparison and analysis.

## Theoretical Framework

Evaluating code generation models requires metrics beyond simple syntactic correctness. For SVA generation specifically, the benchmark framework assesses:

1. **Syntactic validity**: Whether generated SVA code compiles without syntax errors
2. **Semantic correctness**: Whether generated assertions are logically equivalent to reference specifications
3. **Semantic proximity**: The degree of implication relationship (strong/weak/none) between generated and reference properties
4. **Robustness**: Model performance across diverse property types and complexities
5. **Efficiency**: Generation latency and computational resource utilization

The benchmark employs formal verification to establish semantic relationships:

- **Equivalence** (`A ⇔ B`): Generated and reference properties are logically identical
- **Generation implies Reference** (`G → R`): Generated property is stronger (more restrictive)
- **Reference implies Generation** (`R → G`): Generated property is weaker (more permissive)
- **No Relationship**: Properties are semantically unrelated

This taxonomy enables fine-grained analysis of model behavior beyond binary correct/incorrect classification.

## Architecture

The benchmarking pipeline consists of:

1. **Dataset ingestion**: Load (SVAD, Reference SVA) pairs from JSON
2. **LLM invocation**: Generate SVA predictions from SVAD descriptions via API calls
3. **Syntax validation**: Verify generated code with Verible parser
4. **Semantic verification**: Check implication relationships using formal tools (EBMC)
5. **Relationship classification**: Categorize each result as EQUIVALENT, IMPLIES, NO_RELATIONSHIP, or ERROR
6. **Metrics aggregation**: Compute equivalence rate, implication rate, success rate, timing statistics
7. **Multi-model comparison**: Side-by-side evaluation of competing models

The tool supports:
- **Multiprocessing**: Parallel evaluation across CPU cores
- **Persistent caching**: Avoid redundant LLM calls and verification runs
- **Progress tracking**: Real-time feedback during long-running benchmarks
- **Detailed logging**: Per-item results for error analysis

## Evaluation Metrics

### Primary Metrics

1. **Equivalence Rate**: Percentage of test cases where generated SVA is semantically equivalent to reference
   - Formula: `equivalent_count / total_items`
   - Interpretation: Higher is better; 100% indicates perfect semantic alignment

2. **Any Implication Rate**: Percentage where generated SVA has *any* implication relationship with reference
   - Formula: `(equivalent + gen→ref + ref→gen) / total_items`
   - Interpretation: Measures semantic relevance; generated properties are "close" to reference

3. **Success Rate**: Percentage of valid generations (excludes syntax errors and verification failures)
   - Formula: `(total_items - error_count) / total_items`
   - Interpretation: Measures reliability; low values indicate frequent crashes

### Secondary Metrics

4. **Generation Time**: Average LLM inference latency per property
5. **Verification Time**: Average formal verification latency per property
6. **Error Analysis**: Breakdown of error types (syntax, timeout, API failure)

## Use Cases

1. **Model selection**: Compare GPT-4, Claude, Llama, and domain-specific models
2. **Hyperparameter tuning**: Evaluate impact of temperature, top-p, prompt engineering
3. **Fine-tuning validation**: Measure improvement after supervised training
4. **Ablation studies**: Isolate impact of training data characteristics
5. **Publication metrics**: Reproducible quantitative results for academic papers
6. **Regression testing**: Detect performance degradation across model versions

## Command Reference

### Benchmark Single Model

Evaluate one LLM on a dataset:

```bash
sva-benchmark run \
  dataset.json \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key sk-YOUR_KEY \
  --output results.json \
  --workers 8 \
  --verible-path /path/to/verible-verilog-syntax
```

Output:
```
Benchmark Results: gpt-4o
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Value          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Total Items           │ 500            │
│ Equivalent            │ 425 (85.0%)    │
│ Generated → Reference │ 30             │
│ Reference → Generated │ 15             │
│ No Relationship       │ 20             │
│ Errors                │ 10             │
│ Any Implication Rate  │ 94.0%          │
│ Success Rate          │ 98.0%          │
│ Avg Generation Time   │ 2.34s          │
│ Avg Verification Time │ 1.87s          │
└───────────────────────┴────────────────┘
```

### Benchmark Multiple Models

Compare several LLMs simultaneously:

```bash
sva-benchmark run-multi \
  dataset.json \
  --config-file llm_configs.json \
  --output comparison_results.json \
  --workers 8
```

`llm_configs.json`:
```json
[
  {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key": "sk-KEY1"
  },
  {
    "base_url": "https://api.anthropic.com/v1",
    "model": "claude-3-opus-20240229",
    "api_key": "sk-KEY2"
  },
  {
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key": "sk-KEY3"
  }
]
```

Output includes comparison table:
```
Model Comparison
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Model             ┃ Equivalent ┃ Any Impl. ┃ Success ┃ Avg Gen Time┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
│ gpt-4o            │ 85.0% ★    │ 94.0% ★   │ 98.0%   │ 2.34s       │
│ claude-3-opus     │ 82.5%      │ 92.3%     │ 97.5%   │ 3.12s       │
│ deepseek-chat     │ 78.0%      │ 89.0%     │ 96.0%   │ 1.45s       │
└───────────────────┴────────────┴───────────┴─────────┴─────────────┘

Best Equivalent Rate: gpt-4o
Best Any Implication Rate: gpt-4o
```

### Dataset Statistics

Inspect benchmark dataset characteristics:

```bash
sva-benchmark stats dataset.json
```

Output:
```
Dataset Statistics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                ┃ Count ┃ Percentage ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ Total Entries         │ 1000  │ 100%       │
│ Has SVA               │ 1000  │ 100.0%     │
│ Has SVAD              │ 985   │ 98.5%      │
│ Has Both (Benchmarkable)│ 985 │ 98.5%      │
│ Has CoT               │ 950   │ 95.0%      │
└───────────────────────┴───────┴────────────┘
```

### Single-Process Mode

Disable multiprocessing for debugging:

```bash
sva-benchmark run dataset.json \
  --base-url ... --model ... --api-key ... \
  --single-process \
  --rate-limit 1.0
```

### Custom Verification Parameters

Configure EBMC and Verible paths:

```bash
sva-benchmark run dataset.json \
  --ebmc-path /custom/path/ebmc \
  --depth 30 \
  --verible-path /custom/path/verible-verilog-syntax \
  --base-url ... --model ... --api-key ...
```

### Cache Management

Clear cached results:

```bash
sva-benchmark clear-cache /path/to/cache_dir --force
```

Use custom cache directory:

```bash
sva-benchmark run dataset.json \
  --cache-dir ./benchmark_cache \
  --base-url ... --model ... --api-key ...
```

## Input Format

The benchmark dataset should be a JSON array with this structure:

```json
[
  {
    "id": "test_001",
    "SVAD": "Check that a request is followed by an acknowledge within 1-3 cycles",
    "SVA": "assert property (@(posedge clk) req |-> ##[1:3] ack);",
    "CoT": "Optional chain-of-thought explanation"
  },
  {
    "id": "test_002",
    "SVAD": "Verify valid signal implies data ready on next cycle",
    "SVA": "property p; @(posedge clk) valid |=> data_ready; endproperty"
  }
]
```

Required fields:
- `SVAD`: Natural language description (input to LLM)
- `SVA`: Reference implementation (ground truth)

Optional fields:
- `id`: Unique identifier
- `CoT`: Chain-of-thought reasoning (for future research)

## Output Format

Detailed results JSON:

```json
{
  "summary": {
    "model_name": "gpt-4o",
    "total_items": 500,
    "equivalent_count": 425,
    "equivalent_rate": 0.85,
    "generated_implies_reference_count": 30,
    "reference_implies_generated_count": 15,
    "no_relationship_count": 20,
    "error_count": 10,
    "any_implication_rate": 0.94,
    "success_rate": 0.98,
    "avg_generation_time": 2.34,
    "avg_verification_time": 1.87
  },
  "individual_results": [
    {
      "svad": "Check request-acknowledge handshake",
      "reference_sva": "req |-> ##1 ack",
      "generated_sva": "req |-> ##[1:1] ack",
      "relationship": "EQUIVALENT",
      "error_message": null,
      "generation_time": 2.1,
      "verification_time": 1.5
    }
  ],
  "sva_pairs": [
    {
      "SVA1": "req |-> ##1 ack",
      "SVA2": "req |-> ##[1:1] ack",
      "SVAD": "Check request-acknowledge handshake",
      "CoT": "...",
      "relationship": "EQUIVALENT"
    }
  ]
}
```

## Integration Examples

### Automated Model Evaluation Pipeline

```python
# evaluate_models.py
import subprocess
import json

def benchmark_model(dataset, model_config, output_file):
    """Run benchmark for a single model."""
    cmd = [
        "sva-benchmark", "run",
        dataset,
        "--base-url", model_config["base_url"],
        "--model", model_config["model"],
        "--api-key", model_config["api_key"],
        "--output", output_file,
        "--workers", "8"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    with open(output_file) as f:
        return json.load(f)

# Evaluate multiple models
models = [
    {"base_url": "https://api.openai.com/v1", "model": "gpt-4o", "api_key": "..."},
    {"base_url": "https://api.anthropic.com/v1", "model": "claude-3-opus", "api_key": "..."}
]

for i, model in enumerate(models):
    print(f"Benchmarking {model['model']}...")
    results = benchmark_model("dataset.json", model, f"results_{i}.json")
    print(f"Equivalence Rate: {results['summary']['equivalent_rate']:.2%}")
```

### Statistical Significance Testing

```python
# significance_test.py
import json
from scipy import stats

def load_benchmark_results(file):
    with open(file) as f:
        data = json.load(f)
    return [1 if r["relationship"] == "EQUIVALENT" else 0
            for r in data["individual_results"]]

# Compare two models
model_a = load_benchmark_results("gpt4_results.json")
model_b = load_benchmark_results("claude_results.json")

# McNemar's test for paired binary data
contingency_table = [[0, 0], [0, 0]]
for a, b in zip(model_a, model_b):
    contingency_table[a][b] += 1

statistic, pvalue = stats.mcnemar(contingency_table, exact=True).statistic, stats.mcnemar(contingency_table, exact=True).pvalue

print(f"McNemar's test: p-value = {pvalue:.4f}")
if pvalue < 0.05:
    print("Statistically significant difference detected")
```

### Continuous Benchmarking

```bash
#!/bin/bash
# continuous_benchmark.sh - Nightly evaluation pipeline

DATASET="benchmark_datasets/nightly_eval.json"
RESULTS_DIR="benchmark_results/$(date +%Y%m%d)"

mkdir -p "$RESULTS_DIR"

# Benchmark all models from config
sva-benchmark run-multi "$DATASET" \
  --config-file configs/production_models.json \
  --output "$RESULTS_DIR/results.json" \
  --workers 16

# Extract key metrics
python scripts/extract_metrics.py \
  "$RESULTS_DIR/results.json" \
  > "$RESULTS_DIR/summary.txt"

# Upload to dashboard
curl -X POST https://dashboard.example.com/api/benchmark \
  -H "Content-Type: application/json" \
  -d @"$RESULTS_DIR/results.json"

echo "Benchmark complete: $RESULTS_DIR"
```

## Performance Optimization

### Optimal Worker Count

For LLM-based benchmarking:

```
workers = min(
    CPU_cores,
    API_rate_limit / avg_request_time,
    available_memory / memory_per_worker
)
```

Example:
- 16-core machine
- OpenAI rate limit: 500 req/min ≈ 8.3 req/sec
- Average request time: 3 seconds
- Optimal workers: `min(16, 8.3 * 3, ...) ≈ 8 workers`

### Cache Efficiency

Enable caching to avoid redundant work:

```bash
# First run: generates and caches all results
sva-benchmark run dataset.json --cache-dir ./cache ...

# Subsequent runs: instant if same dataset/model
sva-benchmark run dataset.json --cache-dir ./cache ...
```

## Error Analysis

Common error types:

### Syntax Errors

Generated SVA fails Verible parsing:

```
Error: Syntax error in generated SVA
```

Mitigation:
- Improve prompt engineering
- Add few-shot examples
- Fine-tune model on syntactically valid SVA

### Verification Timeout

EBMC exceeds time limit:

```
Error: Verification timeout
```

Solution:
- Increase `--depth` bound if properties are shallow
- Reduce complexity of test cases
- Use faster verification engine (VCFormal)

### API Failures

LLM API returns errors:

```
Error: API request failed (429 Too Many Requests)
```

Solution:
- Reduce `--workers`
- Increase `--rate-limit`
- Implement exponential backoff

## Limitations

1. **Formal verification dependency**: Requires EBMC or VCFormal; unbounded proofs may be inconclusive
2. **Bounded semantics**: Verification depth limits may miss deep temporal behaviors
3. **Cost**: Benchmarking 1000 properties with GPT-4 costs approximately $10-50
4. **Latency**: Full benchmark on 1000 items takes 30-60 minutes with 8 workers
5. **Reference quality**: Benchmark results are only as good as the ground-truth dataset

## Best Practices

1. **Curate diverse datasets**: Include simple/complex properties, various temporal operators
2. **Control for difficulty**: Stratify results by property complexity
3. **Report confidence intervals**: Use bootstrap or cross-validation for statistical rigor
4. **Version control results**: Track model versions, dataset versions, and tool versions
5. **Reproduce baselines**: Re-run benchmarks periodically to detect infrastructure changes
6. **Qualitative analysis**: Manually inspect errors to identify systematic failure modes

## References

- Evaluating Large Language Models Trained on Code (Chen et al., 2021)
- CodeBLEU: A Method for Automatic Evaluation of Code Synthesis (Ren et al., 2020)
- Program Synthesis Benchmarking (survey paper)
- IEEE 1800-2017 SystemVerilog Assertions Standard
