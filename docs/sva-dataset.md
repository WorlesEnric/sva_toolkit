# SVA Dataset Builder (`sva-dataset`)

## Overview

The SVA Dataset Builder is a comprehensive data preprocessing pipeline designed to construct high-quality training datasets for machine learning applications targeting SystemVerilog Assertion generation and reasoning. The tool augments raw SVA code with two critical annotations: SVAD (SystemVerilog Assertion Description) generated via large language models, and CoT (Chain-of-Thought) reasoning generated through structured template-based methods. The builder supports multiprocessing for scalable dataset construction, persistent caching for fault tolerance, and extensive validation for quality assurance.

## Theoretical Background

Supervised learning for code generation and formal reasoning requires datasets that pair code artifacts with natural language descriptions and reasoning traces. For SVA-focused applications:

1. **SVAD (Assertion Description)**: Natural language specifications that describe the intended behavior of an SVA property. These serve as input prompts for LLM-based assertion generation models.
2. **CoT (Chain-of-Thought)**: Step-by-step reasoning explanations that decompose complex temporal logic into interpretable narratives, improving model understanding and generalization.

The dataset construction process must address several challenges:

1. **Scalability**: Processing thousands of SVA properties requires parallelization and caching
2. **Quality control**: LLM-generated descriptions require validation and error handling
3. **Consistency**: CoT generation must be deterministic and aligned with formal semantics
4. **Fault tolerance**: Long-running pipelines need checkpointing and resume capabilities

This tool provides a production-grade solution for building datasets suitable for fine-tuning models like GPT, Claude, or domain-specific transformers.

## Architecture

The dataset builder employs a multi-stage pipeline:

1. **Input ingestion**: JSON files containing SVA properties (with optional existing SVAD/CoT)
2. **SVAD generation** (optional):
   - LLM API calls with custom system prompts
   - Rate limiting to respect API quotas
   - Retry logic for transient failures
3. **CoT generation** (optional):
   - Template-based explanation synthesis
   - No LLM dependency, fully deterministic
4. **Parallel processing**:
   - Multiprocessing with configurable worker pool
   - Per-item caching with SHA-256 hashing
   - Progress tracking with real-time updates
5. **Validation**:
   - Coverage metrics (SVAD/CoT completeness)
   - Error detection and reporting
   - Dataset integrity verification

## Caching Mechanism

The builder implements a persistent cache system to enable:

1. **Resumability**: Interrupted processes can continue from the last checkpoint
2. **Cost efficiency**: Avoids redundant LLM API calls (which incur monetary costs)
3. **Debugging**: Incremental validation during dataset construction
4. **Reproducibility**: Cached results ensure consistent outputs across runs

Cache keys are computed as `SHA-256(SVA_code + generation_mode)`, stored as JSON files in a designated cache directory.

## Use Cases

1. **LLM fine-tuning**: Creating (SVAD → SVA) pairs for supervised learning
2. **Reasoning model training**: Generating (SVA → CoT) datasets for explainability research
3. **Benchmark construction**: Building evaluation datasets for assertion generation tasks
4. **Data augmentation**: Expanding existing datasets with automated annotations
5. **Quality assurance**: Validating consistency between descriptions and implementations

## Command Reference

### Build Complete Dataset

Generate both SVAD and CoT annotations for all entries:

```bash
sva-dataset build \
  input_dataset.json \
  output_dataset.json \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --api-key sk-YOUR_API_KEY \
  --workers 8 \
  --temperature 0.3
```

This processes `input_dataset.json`, generates missing SVAD via GPT-4 and CoT via templates, and writes results to `output_dataset.json`.

### Build with Custom System Prompt

Use a domain-specific system prompt for SVAD generation:

```bash
sva-dataset build \
  input.json output.json \
  --base-url https://api.anthropic.com/v1 \
  --model claude-3-opus-20240229 \
  --api-key YOUR_KEY \
  --system-prompt-file custom_prompt.txt
```

`custom_prompt.txt` might contain:
```
You are an expert in SystemVerilog Assertions for networking protocols.
Generate concise, technically accurate descriptions of SVA properties
focusing on packet processing and flow control semantics.
```

### Build CoT-Only (No LLM Required)

Generate only Chain-of-Thought annotations without LLM API calls:

```bash
sva-dataset build-cot-only input.json output.json --workers 16
```

This is useful for:
- Adding reasoning traces to existing datasets
- Testing pipeline without API costs
- High-throughput processing (no rate limits)

### Single-Process Mode

Disable multiprocessing for debugging:

```bash
sva-dataset build input.json output.json \
  --base-url URL --model MODEL --api-key KEY \
  --single-process \
  --rate-limit 1.0
```

Single-process mode enables:
- Easier debugging with sequential execution
- Rate limiting (sleep between API calls)
- Simplified error tracing

### Generate SVAD for Single Property

Test SVAD generation on a single SVA:

```bash
sva-dataset generate-svad \
  "assert property (@(posedge clk) req |-> ##1 ack);" \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --api-key sk-KEY
```

Output:
```
Generated SVAD:
This property verifies that whenever 'req' is asserted on a positive clock edge,
the 'ack' signal must be asserted exactly one clock cycle later.
```

### Generate CoT for Single Property

Test CoT generation locally:

```bash
sva-dataset generate-cot "property p; a ##[1:3] b; endproperty"
```

### Validate Existing Dataset

Check completeness and quality of a dataset:

```bash
sva-dataset validate dataset.json
```

Output:
```
Dataset Validation Report
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric            ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Entries     │ 500   │
│ Entries with SVAD │ 495   │
│ Entries with CoT  │ 500   │
│ Entries with Errors│ 5    │
│ SVAD Coverage     │ 99.0% │
│ CoT Coverage      │ 100.0%│
└───────────────────┴───────┘
```

### Cache Management

#### Clear Cache

Remove all cached results to force regeneration:

```bash
sva-dataset clear-cache /path/to/cache_dir --force
```

#### Custom Cache Directory

Specify a persistent cache location:

```bash
sva-dataset build input.json output.json \
  --cache-dir ./sva_cache \
  --base-url URL --model MODEL --api-key KEY
```

#### Disable Caching

Force fresh generation (ignore existing cache):

```bash
sva-dataset build input.json output.json \
  --no-cache \
  --base-url URL --model MODEL --api-key KEY
```

## Input Format

The input JSON should be a list of dictionaries containing SVA properties:

```json
[
  {
    "id": "property_001",
    "SVA": "assert property (@(posedge clk) req |-> ##1 ack);",
    "SVAD": null,
    "CoT": null
  },
  {
    "id": "property_002",
    "SVA": "property p; valid && ready |=> data; endproperty",
    "SVAD": "Existing description if available",
    "CoT": null
  }
]
```

Fields:
- `id` (optional): Unique identifier for the property
- `SVA` (required): SystemVerilog Assertion code
- `SVAD` (optional): Pre-existing description (will skip generation if present)
- `CoT` (optional): Pre-existing reasoning (will skip generation if present)

## Output Format

The output JSON contains augmented entries:

```json
[
  {
    "id": "property_001",
    "SVA": "assert property (@(posedge clk) req |-> ##1 ack);",
    "SVAD": "This property checks that whenever 'req' is high at a rising clock edge, 'ack' must be high exactly one cycle later.",
    "CoT": "# Chain-of-Thought Explanation\n\n## Property Overview\nThis assertion verifies a single-cycle request-acknowledge handshake...",
    "error": null
  }
]
```

Fields:
- `error`: Populated with error message if generation failed, otherwise `null`

## Integration Examples

### Fine-Tuning Dataset Preparation

```python
# prepare_finetuning_data.py
import json
import subprocess

def build_finetune_dataset(input_file, output_file, model):
    """Build dataset for LLM fine-tuning."""
    # Generate SVAD and CoT
    subprocess.run([
        "sva-dataset", "build",
        input_file, "temp_output.json",
        "--base-url", "https://api.openai.com/v1",
        "--model", model,
        "--api-key", os.environ["OPENAI_API_KEY"],
        "--workers", "8"
    ])

    # Transform to fine-tuning format
    with open("temp_output.json") as f:
        data = json.load(f)

    finetune_data = []
    for item in data:
        if item["error"]:
            continue  # Skip failed entries

        finetune_data.append({
            "messages": [
                {"role": "system", "content": "You are an expert in SystemVerilog Assertions."},
                {"role": "user", "content": f"Generate SVA for: {item['SVAD']}"},
                {"role": "assistant", "content": item["SVA"]}
            ]
        })

    with open(output_file, "w") as f:
        for item in finetune_data:
            f.write(json.dumps(item) + "\n")

build_finetune_dataset("raw_svas.json", "finetune.jsonl", "gpt-4")
```

### Parallel Batch Processing

```bash
#!/bin/bash
# parallel_build.sh - Process multiple dataset files in parallel

INPUT_DIR="raw_datasets"
OUTPUT_DIR="processed_datasets"

mkdir -p "$OUTPUT_DIR"

# Process each file in background
for input_file in "$INPUT_DIR"/*.json; do
    base_name=$(basename "$input_file")
    output_file="$OUTPUT_DIR/$base_name"

    echo "Processing: $base_name"
    sva-dataset build "$input_file" "$output_file" \
        --base-url https://api.siliconflow.cn/v1 \
        --model deepseek-chat \
        --api-key "$API_KEY" \
        --workers 4 \
        --cache-dir ./shared_cache &
done

# Wait for all background jobs
wait
echo "All datasets processed!"
```

### Quality Filtering

```python
# filter_dataset.py - Remove low-quality entries
import json

def filter_by_quality(dataset_file, output_file, min_svad_length=50):
    """Filter dataset by quality criteria."""
    with open(dataset_file) as f:
        data = json.load(f)

    filtered = []
    for item in data:
        if item.get("error"):
            continue  # Remove failed entries

        if len(item.get("SVAD", "")) < min_svad_length:
            continue  # Remove short/trivial descriptions

        if not item.get("CoT"):
            continue  # Require CoT

        filtered.append(item)

    print(f"Filtered: {len(data)} → {len(filtered)} entries")

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

filter_by_quality("raw_dataset.json", "clean_dataset.json")
```

## Performance Optimization

### Multiprocessing Efficiency

Optimal worker count depends on:
- CPU cores: `workers = CPU_count - 1` for CPU-bound CoT generation
- API rate limits: `workers = requests_per_second * avg_latency`
- Memory: Each worker maintains separate state; monitor RAM usage

Example:
```bash
# 16-core machine with OpenAI API (500 req/min limit)
sva-dataset build input.json output.json \
  --workers 8 \
  --rate-limit 0.12 \
  --base-url ... --model ... --api-key ...
```

### Cache Hit Rate Optimization

Maximize cache reuse:
1. Use consistent cache directory across runs
2. Avoid `--no-cache` unless necessary
3. Group similar SVAs together (improves LLM cache efficiency)

Check cache statistics:
```python
from sva_toolkit.dataset_builder import DatasetBuilder

builder = DatasetBuilder(cache_dir="./cache")
stats = builder.get_cache_stats()
print(f"Cached items: {stats['cached_items']}")
```

## Error Handling

Common errors and solutions:

### API Authentication Failure

```
Error: Invalid API key
```

Solution:
```bash
export OPENAI_API_KEY="sk-your-valid-key"
sva-dataset build ... --api-key "$OPENAI_API_KEY"
```

### Rate Limit Exceeded

```
Error: Rate limit exceeded
```

Solution:
```bash
# Increase delay between requests
sva-dataset build ... --rate-limit 2.0 --single-process
```

### Memory Exhaustion

```
Error: Out of memory
```

Solution:
```bash
# Reduce worker count
sva-dataset build ... --workers 2
```

### Incomplete Generation

If the process crashes midway:

```bash
# Resume using cache (already-processed items will be skipped)
sva-dataset build input.json output.json \
  --cache-dir ./cache \
  --base-url ... --model ... --api-key ...
```

## Limitations

1. **LLM dependency**: SVAD quality depends on the chosen language model
2. **Cost**: Large datasets incur significant API costs (estimate: $0.01-0.10 per property)
3. **Latency**: LLM calls introduce network latency (1-10 seconds per property)
4. **SVAD variability**: Non-zero temperature produces non-deterministic descriptions
5. **Context limits**: Very long SVA properties may exceed LLM context windows

## Best Practices

1. **Start small**: Test pipeline on 10-100 properties before scaling
2. **Use caching**: Always enable caching for production runs
3. **Monitor costs**: Track API usage and set budget alerts
4. **Validate incrementally**: Run `validate` command periodically during large builds
5. **Version control**: Track dataset versions with git or DVC
6. **Temperature tuning**: Use 0.0-0.3 for consistency, 0.7-1.0 for diversity

## References

- Fine-Tuning Language Models for Code Generation (Chen et al.)
- Supervised Fine-Tuning of Large Language Models (Ouyang et al., 2022)
- Dataset Quality in Machine Learning (Sambasivan et al., 2021)
