# SVA Parser

## Purpose

`sva_toolkit.sva` is the core parser package for V3. It replaces V1's Verible-centered parsing workflow with a native lexer, parser, AST, emitter, analysis, transforms, and visitors layer that other V3 modules can reuse directly.

## CLI Commands

Parse inline text:

```bash
sva parse "assert property (@(posedge clk) req |-> ##1 ack);" --format text
```

Parse a file and emit JSON:

```bash
sva parse examples/inputs/parse/req_ack.sv --format json
```

The `sva parse` command accepts either raw property text or a file path. Output can be `text` or `json`.

## Usage Examples

Parse a property body:

```python
from sva_toolkit.sva import parse_property_body

node = parse_property_body("req |=> ##1 ack")
print(type(node).__name__)
```

Round-trip text through the parser and emitter:

```python
from sva_toolkit.sva import emit_property_text, parse_property_text

spec = parse_property_text("assert property (@(posedge clk) req |-> ##1 ack);")
print(emit_property_text(spec))
```

Parse expressions or sequences independently:

```python
from sva_toolkit.sva import parse_expr, parse_sequence

expr = parse_expr("$rose(req) && ready")
seq = parse_sequence("req ##[1:3] ack")
```

## API Reference

Primary functions:

- `parse_expr(text: str) -> ExprNode`
- `parse_sequence(text: str) -> SequenceNode`
- `parse_property_body(text: str) -> PropertyNode`
- `parse_property_text(text: str) -> PropertySpec`
- `emit_expr(node: ExprNode) -> str`
- `emit_sequence(node: SequenceNode) -> str`
- `emit_property_body(node: PropertyNode) -> str`
- `emit_property_text(spec: PropertySpec) -> str`

Common AST types re-exported from `sva_toolkit.sva`:

- `PropertySpec`
- `ClockingEvent`
- `ImplicationProperty`
- `DelaySequence`
- `RepeatSequence`
- `SequenceBinary`
- `UnaryExpr`
- `BinaryExpr`
- `CallExpr`
- `Identifier`
- `Literal`

## Notes for V3

- Parser behavior is property-centric: it targets assertion/property surfaces rather than full RTL module parsing.
- Other V3 domains consume this parser instead of duplicating syntax handling.
- Verible is no longer a required dependency for parsing. It remains optional elsewhere for validation-only workflows.

## Related Docs

- [Architecture](architecture.md)
- [Formal verification](sva-formal.md)
- [Description engine](sva-describe.md)
