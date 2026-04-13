# Description Engine

## Purpose

`sva_toolkit.describe` turns SVA into documentation-oriented text. V3 bundles two outputs:

- `SVADTranslator` for structured natural-language descriptions
- `SVACoTBuilder` for step-by-step chain-of-thought style explanations

Both operate on top of the V3 parser instead of the old V1 parsing stack.

## CLI Commands

Generate SVAD text:

```bash
sva describe svad examples/inputs/parse/req_ack.sv
```

Generate JSON-wrapped SVAD:

```bash
sva describe svad examples/inputs/parse/req_ack.sv --format json
```

Generate CoT markdown/text:

```bash
sva describe cot examples/inputs/parse/req_ack.sv --format markdown
```

Generate JSON-wrapped CoT:

```bash
sva describe cot examples/inputs/parse/req_ack.sv --format json
```

## Usage Examples

Translate one property:

```python
from sva_toolkit.describe import SVADTranslator

translator = SVADTranslator()
print(translator.translate("assert property (@(posedge clk) req |-> ##1 ack);"))
```

Build chain-of-thought reasoning:

```python
from sva_toolkit.describe import SVACoTBuilder

builder = SVACoTBuilder()
print(builder.build("assert property (@(posedge clk) disable iff (!rst_n) $rose(req) |=> ##1 ack);"))
```

## API Reference

Public classes:

- `SVADTranslator`
- `SVACoTBuilder`
- `CoTSection`

Key methods:

- `SVADTranslator.translate(sva_code: str) -> str`
- `SVACoTBuilder.build(sva_code: str) -> str`
- `SVACoTBuilder.build_from_structure(structure) -> str`

## V3 Notes

- `SVADTranslator` produces the most compact documentation-oriented output.
- `SVACoTBuilder` is better suited for dataset enrichment and model-reasoning traces.
- The CLI supports `text`, `json`, and `markdown` output selectors, although both engines currently render text-first content internally.

## Related Docs

- [SVA parser](sva-parse.md)
- [Data workflows](sva-data.md)
- [Examples](../examples/README.md)
