[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CLI](https://img.shields.io/badge/cli-sva-0A7EA4.svg)](./src/sva_toolkit/cli/main.py)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](./docs/architecture.md)

# SVA Toolkit V3

`sva-toolkit` is a consolidated SystemVerilog Assertion toolkit built around a single `sva` CLI. V3 keeps V2's Python 3.11 package structure and unified runtime while bringing parsing, formal checking, timing-diagram workflows, generation, natural-language description, and dataset tooling into one installable package.

## Installation

Core install:

```bash
python -m pip install -e .
```

Useful extras:

```bash
python -m pip install -e ".[timing-render]"
python -m pip install -e ".[llm]"
python -m pip install -e ".[all]"
```

External tools are discovered at runtime, not import time:

- `ebmc` or `vcf` for `sva formal ...`
- `verible-verilog-syntax` for `sva generate --validate`
- `cairosvg` for `sva timing render --format png`

## Quickstart

Parse an assertion:

```bash
sva parse "assert property (@(posedge clk) req |-> ##1 ack);" --format json
```

Validate and render a timing diagram:

```bash
sva timing validate examples/td/01_simple_handshake.td
sva timing render examples/td/01_simple_handshake.td -o examples/out/01_simple_handshake.svg
```

Generate a few properties:

```bash
sva generate --count 3 --coverage
```

Describe an assertion in natural language:

```bash
sva describe svad examples/inputs/parse/req_ack.sv
sva describe cot examples/inputs/parse/req_ack.sv
```

Build a local dataset without any API calls:

```bash
sva data build examples/data/dataset_input.json -o examples/out/dataset.jsonl --workers 1
```

See [examples/README.md](examples/README.md) for runnable scripts covering every CLI subcommand.

## CLI Reference

| Command | Purpose | Notes |
| --- | --- | --- |
| `sva parse` | Parse one property/assertion surface into text or JSON | Accepts inline text or a file path |
| `sva formal check` | Check whether one property implies another | Requires `ebmc` or `vcf` on `PATH` |
| `sva formal equivalent` | Check bidirectional equivalence | Requires `ebmc` or `vcf` |
| `sva formal relationship` | Report implication in both directions | Requires `ebmc` or `vcf` |
| `sva timing validate` | Parse and validate timing DSL | Pure Python |
| `sva timing render` | Render timing DSL to SVG or PNG | PNG requires `cairosvg` |
| `sva timing emit-sva` | Emit parameterized SVA from timing DSL | Pure Python |
| `sva timing extract-sva` | Recover timing DSL from SVA files | Pure Python extraction |
| `sva timing bundle-sva` | Group related SVA files into shared scenarios | Pure Python extraction |
| `sva generate` | Generate random or stratified SVA modules | `--validate` uses Verible if installed |
| `sva describe svad` | Generate a structured natural-language description | Pure Python |
| `sva describe cot` | Generate chain-of-thought style reasoning | Pure Python |
| `sva data build` | Build JSONL datasets from raw SVA entries | Works offline; optional LLM for SVAD |
| `sva data benchmark` | Evaluate SVAD -> SVA generation results | Requires `.[llm]`, `OPENAI_API_KEY`, and a model |

## Module Overview

| Domain | Package | What it does | Docs |
| --- | --- | --- | --- |
| Parser | `sva_toolkit.sva` | Lexer, parser, AST, emitter, transforms, visitors | [docs/sva-parse.md](docs/sva-parse.md) |
| Formal | `sva_toolkit.formal` | Practical property normalization plus EBMC and VC Formal adapters | [docs/sva-formal.md](docs/sva-formal.md) |
| Timing | `sva_toolkit.timing` | Timing DSL parser, bridge layers, WaveDrom-style rendering, SVA extraction/emission | [docs/sva-timing.md](docs/sva-timing.md) |
| Generate | `sva_toolkit.generate` | Type-directed generation, stratified sampling, templates, NL helpers | [docs/sva-generate.md](docs/sva-generate.md) |
| Describe | `sva_toolkit.describe` | SVAD translation and CoT generation from parsed SVA | [docs/sva-describe.md](docs/sva-describe.md) |
| Data | `sva_toolkit.data` | Dataset building, caching, benchmarking, multiprocessing | [docs/sva-data.md](docs/sva-data.md) |
| Runtime | `sva_toolkit.runtime` | Tool discovery, process helpers, LLM client, shared config | [docs/architecture.md](docs/architecture.md) |

## Documentation Map

- [Architecture](docs/architecture.md)
- [SVA Parser](docs/sva-parse.md)
- [Formal Verification](docs/sva-formal.md)
- [Timing Diagrams](docs/sva-timing.md)
- [SVA Generation](docs/sva-generate.md)
- [Description Engine](docs/sva-describe.md)
- [Data Workflows](docs/sva-data.md)

## Development Setup

Install the development environment:

```bash
python -m pip install -e ".[dev]"
```

Common checks:

```bash
pytest -q
ruff check src tests
python -m build
```

When editing docs or examples, keep command lines aligned with the unified `sva` CLI and avoid embedding real credentials. Timing and formal features should continue to degrade gracefully when optional dependencies are missing.

## Contributing

Contributions should preserve the V3 package boundaries:

- Keep CLI examples on the `sva ...` entry point instead of reviving V1/V2 standalone commands.
- Treat `src/` modules as the source of truth for behavior and update docs/examples when command surfaces change.
- Prefer placeholders such as `OPENAI_API_KEY=...` or `SVA_TOOLKIT_MODEL=...` over real endpoints or secrets.
- Add or update tests when changing implementation; examples in this directory are documentation assets, not test fixtures.
