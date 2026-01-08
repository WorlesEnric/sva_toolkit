# SVA-Gen: Type-Directed SystemVerilog Assertion Generator with SVAD

SVA-Gen is a programmatic engine designed to generate syntactically legal and "rich" SystemVerilog Assertions (SVA). Unlike simple string-based generators, SVA-Gen uses a Type-Directed Synthesis approach to ensure that temporal logic, boolean expressions, and property implications are always correctly nested.

## Key Feature: SVAD (SVA Descriptions)

Every generated SVA property comes with a **natural language description (SVAD)** that explains what the assertion checks in plain English. This is possible because the type system captures the semantic structure of SVA through well-typed AST nodes, allowing compositional translation to natural language.

## Installation

The generator is part of the sva_toolkit package:

```bash
pip install -e .
```

## CLI Usage

### Generate SVA Properties

```bash
# Generate JSON output with SVA-SVAD pairs (recommended)
sva-gen generate -j -o output.json

# Generate 20 SVA-SVAD pairs
sva-gen generate -n 20 -j -o assertions.json

# Generate SystemVerilog module (for testing/validation)
sva-gen generate -n 10 -o module.sv

# Use specific signals
sva-gen generate -s req -s ack -s valid -s ready -j

# Use predefined signal presets
sva-gen generate --preset axi -j

# Disable validation
sva-gen generate --no-validate -j

# Reproducible generation with seed
sva-gen generate --seed 42 -j
```

#### Output Formats

**JSON Mode (Primary)**: Generates SVA-SVAD pairs as JSON
```json
{
  "properties": [
    {
      "name": "p_gen_0",
      "sva": "$rose(valid) |-> ack",
      "svad": "whenever signal 'valid' rises from 0 to 1, then in the same cycle, signal 'ack'",
      "property_block": "property p_gen_0;\n  @(posedge clk) $rose(valid) |-> ack;\nendproperty"
    }
  ],
  ...
}
```

**SystemVerilog Module Mode**: Generates a complete SV module (for syntax validation/testing)

### Validate SVA Syntax

```bash
# Validate a file
sva-gen validate-file generated.sv

# Validate inline code
sva-gen validate "property p; @(posedge clk) a |-> b; endproperty"
```

### Stress Test

Run stress test to find type system leaks:

```bash
# Run 100 batches of 10 assertions each
sva-gen stress-test -n 100 -s 10
```

### List Signal Presets

```bash
sva-gen list-presets
```

## Programmatic Usage

```python
from sva_toolkit.gen import SVASynthesizer, SVAProperty

# Create synthesizer with advanced features enabled
synthesizer = SVASynthesizer(
    signals=["req", "ack", "gnt", "valid"],
    max_depth=4,
    clock_signal="clk",
    enable_advanced_features=True  # Enables ternary, until, if-else, etc.
)

# Generate a single property with SVAD
prop: SVAProperty = synthesizer.synthesize(name="p_handshake")
print(f"SVA: {prop.sva_code}")
print(f"SVAD: {prop.svad}")
print(f"Property Block:\n{prop.property_block}")

# Generate multiple properties
result = synthesizer.generate_validated(
    module_name="my_assertions",
    num_assertions=10
)

if result.validation.is_valid:
    print("All assertions are syntactically valid!")
    for prop in result.properties:
        print(f"\n{prop.name}:")
        print(f"  SVA:  {prop.sva_code}")
        print(f"  SVAD: {prop.svad}")
else:
    print(f"Validation error: {result.validation.error_message}")
```

### Direct AST to SVAD Conversion

You can also convert SVA AST nodes directly to natural language:

```python
from sva_toolkit.gen.types_sva import *

# Build an SVA property
req = Signal("req")
ack = Signal("ack")
rose_req = UnarySysFunction("$rose", req)
implication = Implication(rose_req, "|=>", ack)

# Get both SVA code and natural language
print(str(implication))                    # SVA: $rose(req) |=> ack
print(implication.to_natural_language())   # SVAD: whenever signal 'req' rises from 0 to 1, then in the next cycle, signal 'ack'
```

### Using Individual Node Types

```python
from sva_toolkit.gen.types_sva import *

# Create signals
req = Signal("req")
ack = Signal("ack")

# Build expressions with operators
xor_expr = BinaryOp(req, "^", ack)  # Bitwise XOR
case_eq = BinaryOp(req, "===", ack)  # 4-value equality
ternary = TernaryOp(req, ack, Signal("gnt"))  # Conditional

# Use system functions
changed = UnarySysFunction("$changed", req)
past_val = PastFunction(ack, depth=3)  # $past(ack, 3)

# Build sequences with advanced features
seq = SequenceDelay(req, "##[0:$]", ack)  # Unbounded delay
first = SequenceFirstMatch(seq)

# Build properties with constructs
prop_until = PropertyUntil(req, "until", ack)
prop_if = PropertyIfElse(req, ack, Signal("ready"))

# Get both SVA and SVAD
print(str(prop_until))                    # SVA: (req until ack)
print(prop_until.to_natural_language())   # SVAD: signal 'req' until signal 'ack'

print(str(prop_if))                       # SVA: if (req) ack else ready
print(prop_if.to_natural_language())      # SVAD: if signal 'req', then signal 'ack', otherwise signal 'ready'
```

## Project Structure

- `types_sva.py`: The core Type System. Defines classes for Boolean, Sequence, and Property.
- `generator.py`: The recursive synthesis engine that builds expression trees based on type rules.
- `templates.py`: SystemVerilog boilerplate to wrap properties into a module.
- `utils.py`: Randomization helpers and weighting logic.
- `cli.py`: The Click-based CLI entry point.

## How It Works

The generator follows the SVA promotion hierarchy:

```
Expressions (Signals, Arithmetic, Bitwise, Unary, Ternary) ->
Booleans (Comparisons, System Functions, Logical Ops) ->
Sequences (Temporal delays ##, repetitions [*], binary ops, first_match, ended) ->
Properties (Implications |->, disable iff, not, if-else, until, and/or).
```

### SVAD Generation

Each node in the type system implements both:
- `__str__()` - Generates SVA code
- `to_natural_language()` - Generates natural language description

The natural language generation is compositional, meaning complex properties are described by combining the descriptions of their sub-expressions. This ensures SVAD is always synchronized with the SVA code.

**Example**:
```python
# SVA: $rose(req) |=> ##[1:3] ack
# SVAD: "whenever signal 'req' rises from 0 to 1, then in the next cycle,
#        between 1 and 3 cycles later, signal 'ack'"
```

### Supported Syntax Features

**Expression Layer:**
- Signals and signal references
- Unary operators: `!`, `~`, `-`, `+`
- Binary arithmetic: `+`, `-`, `*`, `/`, `%`
- Binary bitwise: `&`, `|`, `^`, `^~`, `~^`
- Ternary conditional: `? :`
- System functions: `$past(sig)`, `$past(sig, depth)`

**Boolean Layer:**
- Logical operators: `&&`, `||`, `!`
- Relational operators: `<`, `<=`, `>`, `>=`
- Equality operators: `==`, `!=`, `===`, `!==`
- System functions: `$rose`, `$fell`, `$stable`, `$changed`, `$onehot`, `$onehot0`, `$isunknown`, `$countones`
- Sequence ended: `sequence.ended`

**Sequence Layer:**
- Timing delays: `##1`, `##[1:5]`, `##[0:$]` (unbounded)
- Repetition operators: `[*n]`, `[=n]`, `[->n]` with ranges and unbounded `[*0:$]`
- Binary operators: `intersect`, `throughout`, `and`, `or`
- First match: `first_match(sequence)`

**Property Layer:**
- Implications: `|->`, `|=>`
- Disable clause: `disable iff (condition)`
- Negation: `not property`
- Conditional: `if (condition) property else property`
- Temporal until: `until`, `until_with`
- Binary operators: `and`, `or`

## Syntax Validation

Generated SVAs are validated using Verible. The validation works by:

1. Wrapping generated properties in a dummy module
2. Running `verible-verilog-syntax` on the module
3. Checking the return code for syntax errors

If validation fails, it indicates a "type system leak" - a rule that allows invalid syntax.

## Customization

The refactored type system makes it easy to add new SVA constructs:

### Adding New Operators

To add a new operator (e.g., a custom bitwise operation):

1. Add it to the appropriate enum in `types_sva.py`:
```python
class BinaryOperator(Enum):
    # ... existing operators
    MY_NEW_OP = "~&"  # NAND operator
```

2. Update the generator in `generator.py`:
```python
BITWISE_OPS: List[str] = ["&", "|", "^", "^~", "~^", "~&"]
```

### Adding New Node Types

To add a new SVA construct (e.g., `s_eventually`):

1. Create a new class in `types_sva.py`:
```python
class PropertyEventually(SVANode):
    """Represents s_eventually operator."""
    def __init__(self, prop: SVANode) -> None:
        super().__init__(TYPE_PROPERTY)
        self.prop = prop

    def __str__(self) -> str:
        return f"s_eventually {self.prop}"
```

2. Update the `generate_property` method in `generator.py`:
```python
def generate_property(self, depth: int = 0) -> SVANode:
    choice = random.random()
    # ... existing cases
    elif choice < 0.95:
        return PropertyEventually(self.generate_sequence(depth + 1))
```

### Type Safety with Enums

The refactored system uses enums for type safety. You can use either enums or strings:

```python
# Using enum (type-safe)
op = BinaryOp(sig_a, BinaryOperator.BITWISE_XOR, sig_b)

# Using string (backward compatible)
op = BinaryOp(sig_a, "^", sig_b)
```

## Signal Presets

Available presets:
- `default`: req, ack, gnt, valid, ready, data_en, busy
- `handshake`: req, ack, valid, ready, grant, request
- `fifo`: push, pop, full, empty, almost_full, almost_empty
- `axi`: awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready
