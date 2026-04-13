# SVA Chain-of-Thought Builder (`sva-cot`)

## Overview

The SVA Chain-of-Thought (CoT) Builder is an automated reasoning pipeline that generates structured natural language explanations of SystemVerilog Assertion semantics. By decomposing complex temporal properties into human-readable step-by-step interpretations, the tool bridges the gap between formal specifications and intuitive understanding, facilitating both human comprehension and machine learning applications.

## Theoretical Motivation

Chain-of-Thought reasoning, inspired by cognitive science and educational psychology, involves breaking down complex problem-solving into explicit intermediate steps. For formal verification:

1. **Cognitive load reduction**: Complex temporal logic becomes accessible through structured narratives
2. **Pedagogical value**: Engineers learning SVA benefit from explicit semantic decomposition
3. **Machine learning enhancement**: CoT annotations improve large language model (LLM) performance on reasoning tasks
4. **Verification debugging**: Step-by-step explanations aid in identifying specification errors
5. **Documentation generation**: Automated creation of human-readable property documentation

The CoT builder employs a template-based generation approach combined with syntactic analysis to produce consistent, technically accurate explanations.

## Architecture

The CoT generation pipeline consists of the following components:

1. **Syntactic parsing**: The SVA code is parsed using the AST parser to extract structural components (signals, temporal operators, delays, implications)
2. **Semantic segmentation**: The property is decomposed into logical sections: overview, clocking, antecedent, temporal relationships, consequent, and conclusion
3. **Template instantiation**: Each section is populated using domain-specific templates that map SVA constructs to natural language
4. **Narrative synthesis**: Individual sections are concatenated into a coherent markdown-formatted explanation
5. **Validation**: The generated CoT is optionally cross-checked against the original property for consistency

## Output Format

The generated Chain-of-Thought follows a standardized structure:

1. **Property Overview**: High-level summary of the assertion's purpose
2. **Clock and Reset Context**: Description of synchronization and reset behavior
3. **Antecedent Analysis**: Explanation of the triggering condition
4. **Temporal Relationship**: Interpretation of delay operators and sequence structures
5. **Consequent Analysis**: Description of the expected outcome
6. **Overall Interpretation**: Synthesis of the complete temporal behavior

All output is formatted in GitHub-flavored Markdown for compatibility with documentation systems.

## Use Cases

1. **LLM training data**: Generating (SVA, CoT) pairs for supervised fine-tuning of code-reasoning models
2. **Assertion documentation**: Automated generation of human-readable property specifications
3. **Educational material**: Creating instructional content for SystemVerilog assertion training
4. **Verification debugging**: Producing natural language explanations to aid in identifying specification errors
5. **Cross-domain communication**: Translating formal properties for non-expert stakeholders

## Command Reference

### Generate Chain-of-Thought from String

Generate CoT explanation for an SVA property provided as a command-line argument:

```bash
sva-cot build "assert property (@(posedge clk) disable iff (rst) req |-> ##[1:3] ack);"
```

Output: Rendered markdown with syntax highlighting in the terminal.

### Generate CoT with Raw Markdown Output

Output unrendered markdown text:

```bash
sva-cot build "property p; @(posedge clk) valid |=> data_ready; endproperty" --raw
```

### Save CoT to File

Write the generated explanation to a markdown file:

```bash
sva-cot build "assert property (@(posedge clk) req |-> ##1 gnt);" --output explanation.md
```

### Generate CoT from File

Process SVA code from a SystemVerilog file:

```bash
sva-cot build-file assertions.sv --output documentation.md
```

This command reads the SVA property from the file and writes the CoT explanation to `documentation.md`.

### JSON Output

Export structured CoT data in JSON format:

```bash
sva-cot build "property p; a ##1 b; endproperty" --json-output
```

JSON schema:
```json
{
  "sva_code": "property p; a ##1 b; endproperty",
  "cot": "# Chain-of-Thought Explanation\n\n...",
  "sections": [
    {
      "title": "Property Overview",
      "content": "..."
    },
    {
      "title": "Temporal Analysis",
      "content": "..."
    }
  ]
}
```

### Extract Sections

Display individual CoT sections as separate panels:

```bash
sva-cot sections "assert property (@(posedge clk) req && valid |-> ##2 ack);"
```

This command shows each reasoning section independently, useful for debugging or selective documentation.

## Integration Examples

### Python Integration for Dataset Generation

```python
import subprocess
import json

def generate_cot(sva_code: str) -> str:
    """Generate Chain-of-Thought explanation for SVA code."""
    result = subprocess.run(
        ["sva-cot", "build", sva_code, "--json-output"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"CoT generation failed: {result.stderr}")

    data = json.loads(result.stdout)
    return data["cot"]

# Example: Build training dataset
training_data = []
sva_properties = [
    "req |-> ##1 ack",
    "valid && ready |=> data_out",
    "@(posedge clk) disable iff (rst) a ##[1:3] b"
]

for sva in sva_properties:
    cot = generate_cot(sva)
    training_data.append({
        "input": f"Explain the following SVA: {sva}",
        "output": cot
    })

# Save as JSONL for LLM fine-tuning
with open("sva_cot_dataset.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")
```

### Batch Documentation Generation

```bash
#!/bin/bash
# generate_docs.sh - Create documentation for all assertions

INPUT_DIR="rtl/assertions"
OUTPUT_DIR="docs/properties"

mkdir -p "$OUTPUT_DIR"

for sv_file in "$INPUT_DIR"/*.sv; do
    base_name=$(basename "$sv_file" .sv)
    echo "Generating documentation for $base_name"

    sva-cot build-file "$sv_file" \
        --output "$OUTPUT_DIR/${base_name}.md"
done

echo "Documentation generation complete!"
```

### Integration with Markdown Documentation Systems

```python
# docs_builder.py - Integrate CoT into mkdocs structure
import subprocess
from pathlib import Path

def build_property_docs(sva_dir: Path, docs_dir: Path):
    """Generate markdown docs from SVA files."""
    docs_dir.mkdir(parents=True, exist_ok=True)

    for sva_file in sva_dir.glob("*.sv"):
        md_file = docs_dir / f"{sva_file.stem}.md"

        # Generate CoT
        result = subprocess.run(
            ["sva-cot", "build-file", str(sva_file), "--output", str(md_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ Generated: {md_file}")
        else:
            print(f"✗ Failed: {sva_file}")

    # Update mkdocs.yml navigation
    update_mkdocs_nav(docs_dir)

build_property_docs(
    sva_dir=Path("verification/properties"),
    docs_dir=Path("docs/generated")
)
```

### CoT Quality Validation

```python
# validate_cot.py - Ensure CoT contains key elements
import subprocess
import json

def validate_cot_quality(sva_code: str) -> dict:
    """Check if generated CoT meets quality criteria."""
    result = subprocess.run(
        ["sva-cot", "build", sva_code, "--json-output"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)
    sections = data["sections"]

    criteria = {
        "has_overview": any("overview" in s["title"].lower() for s in sections),
        "has_temporal": any("temporal" in s["title"].lower() for s in sections),
        "min_length": len(data["cot"]) > 100,
        "structured": len(sections) >= 3
    }

    return {
        "sva": sva_code,
        "quality_score": sum(criteria.values()) / len(criteria),
        "criteria": criteria
    }

# Example validation
result = validate_cot_quality("req |-> ##[1:3] ack")
print(f"Quality Score: {result['quality_score']:.2%}")
```

## Output Example

For the input:
```systemverilog
assert property (@(posedge clk) disable iff (rst) req |-> ##[1:3] ack);
```

The generated Chain-of-Thought explanation includes:

```markdown
# Chain-of-Thought Explanation

## Property Overview
This assertion verifies a request-acknowledge handshake protocol with variable latency.

## Clock and Reset Context
- Synchronized to: positive edge of 'clk'
- Reset behavior: assertion disabled when 'rst' is active

## Antecedent (Trigger Condition)
The property is triggered when: 'req' signal is asserted

## Temporal Relationship
After the trigger, the system must respond within 1 to 3 clock cycles (bounded delay)

## Consequent (Expected Response)
The expected response is: 'ack' signal becomes asserted

## Overall Interpretation
When a request is issued (req=1), an acknowledgment (ack=1) must occur within 1 to 3 clock cycles. The property is only checked on rising clock edges and is disabled during reset.
```

## Customization and Extension

The CoT builder supports customization through:

1. **Template modification**: Edit the template system to adjust explanation style
2. **Section filtering**: Select specific sections for targeted documentation
3. **Multi-language support**: Extend templates to generate explanations in multiple languages
4. **Domain-specific vocabularies**: Customize terminology for application-specific contexts (e.g., networking, automotive)

## Limitations

1. **Template-based generation**: May not capture all semantic nuances of complex properties
2. **Natural language ambiguity**: Generated text approximates formal semantics but is not a substitute for formal analysis
3. **SVA coverage**: Limited to constructs supported by the AST parser
4. **Context dependency**: Explanations assume basic knowledge of digital logic and timing concepts

## Quality Assurance

For mission-critical applications, validate generated CoT explanations through:

1. **Expert review**: Human verification engineers review auto-generated documentation
2. **Cross-validation**: Compare CoT against formal verification results
3. **Consistency checking**: Ensure multiple runs produce identical output for the same input
4. **Semantic alignment**: Verify that key terms (signals, operators) appear correctly in explanations

## References

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)
- SystemVerilog Assertions: A Practical Guide (Vijayaraghavan & Ramanathan)
- Natural Language Generation for Formal Specifications (survey paper)
