# Formal Verification

## Purpose

`sva_toolkit.formal` provides a thin normalized property model and a unified service for implication, equivalence, and relationship checks. V3 keeps the V2 service shape while routing practical SVA text through the V3 parser and runtime tool-discovery layer.

## CLI Commands

Check implication:

```bash
sva formal check "req |-> ##1 ack" "req |-> ##[1:2] ack" --backend auto --depth 20 --timeout 300
```

Check equivalence:

```bash
sva formal equivalent "req |-> ##1 ack" "req |-> ##1 ack" --backend auto
```

Report both implication directions:

```bash
sva formal relationship "req |-> ##[1:3] ack" "req |-> ##2 ack" --backend auto
```

These commands require at least one formal backend executable on `PATH`:

- `ebmc`
- `vcf`

If neither is available, the CLI reports that no backend could be selected.

## Usage Examples

Programmatic implication checking:

```python
from sva_toolkit.formal import FormalService

service = FormalService(backend="auto", depth=20, timeout=300)
result = service.check_implication("req |-> ack", "req |-> ##1 ack")
print(result.result.value, result.message)
```

Practical property normalization:

```python
from sva_toolkit.formal import normalize_property, parse_property

parsed = parse_property("assert property (@(posedge clk) disable iff (!rst_n) req |-> ack);")
normalized = normalize_property(parsed)
print(normalized.clock_name, normalized.reset_expr)
```

## API Reference

Primary public objects:

- `FormalService`
- `FormalProperty`
- `CheckResult`
- `ImplicationResult`
- `parse_property(text: str) -> FormalProperty`
- `normalize_property(property: FormalProperty) -> FormalProperty`

`FormalService` methods:

- `check_implication(antecedent: str, consequent: str) -> CheckResult`
- `check_equivalence(sva1: str, sva2: str) -> CheckResult`
- `get_relationship(sva1: str, sva2: str) -> tuple[bool, bool]`

Backends:

- `EbmcBackend`
- `VcformalBackend`

## Operational Notes

- `backend="auto"` prefers VC Formal when `vcf` is available, then falls back to EBMC.
- Syntax and structural normalization happen before backend selection.
- Backend availability is determined through `sva_toolkit.runtime.ToolRegistry`.
- Failed formal checks can surface counterexamples or backend logs in `CheckResult`.

## Related Docs

- [Architecture](architecture.md)
- [SVA parser](sva-parse.md)
- [Data workflows](sva-data.md)
