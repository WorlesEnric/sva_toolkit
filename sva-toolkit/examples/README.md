# Examples

This directory contains runnable examples for the V3 package. All scripts use the unified `sva` CLI.

## Layout

| Path | Contents |
| --- | --- |
| `cli/` | One shell script per CLI subcommand |
| `inputs/parse/` | Small parser/description input files |
| `data/` | Sample dataset-builder and benchmark JSON inputs |
| `td/` | Ported timing DSL examples from V2 |
| `sva/` | Ported SVA bridge examples from V2 |
| `out/` | Default output directory used by scripts |

## Prerequisites

- Install the package first: `python -m pip install -e .`
- Run scripts from the repository root or any location where `sva` is on `PATH`
- Optional requirements:
  - `ebmc` or `vcf` for `formal` examples
  - `cairosvg` for PNG rendering
  - `verible-verilog-syntax` for generator validation paths
  - `.[llm]`, `OPENAI_API_KEY`, and `SVA_TOOLKIT_MODEL` for the benchmark example

## CLI Scripts

| Script | Command exercised | Notes |
| --- | --- | --- |
| `cli/parse.sh` | `sva parse` | Local only |
| `cli/formal-check.sh` | `sva formal check` | Requires formal backend |
| `cli/formal-equivalent.sh` | `sva formal equivalent` | Requires formal backend |
| `cli/formal-relationship.sh` | `sva formal relationship` | Requires formal backend |
| `cli/timing-validate.sh` | `sva timing validate` | Local only |
| `cli/timing-render.sh` | `sva timing render` | SVG path is local only |
| `cli/timing-emit-sva.sh` | `sva timing emit-sva` | Local only |
| `cli/timing-extract-sva.sh` | `sva timing extract-sva` | Local only |
| `cli/timing-bundle-sva.sh` | `sva timing bundle-sva` | Local only |
| `cli/generate.sh` | `sva generate` | Local only; add `--validate` manually if Verible exists |
| `cli/describe-svad.sh` | `sva describe svad` | Local only |
| `cli/describe-cot.sh` | `sva describe cot` | Local only |
| `cli/data-build.sh` | `sva data build` | Uses offline mode by default |
| `cli/data-benchmark.sh` | `sva data benchmark` | Requires LLM support and credentials |

## Timing Example Port

The `td/` and `sva/` directories are copied from the V2 example suite so V3 keeps the same timing bridge fixtures while using the new unified CLI.

## Sample Commands

```bash
bash examples/cli/parse.sh
bash examples/cli/timing-render.sh
bash examples/cli/data-build.sh
```

Examples that depend on external tools or credentials fail fast with a short prerequisite message instead of running partially configured commands.
