#!/usr/bin/env python3
"""
Example usage of SVA Toolkit.

This script demonstrates the main features of the toolkit.
"""

import json
from pathlib import Path

# Note: Run `pip install -e .` from the project root before running this script


def example_ast_parser():
    """Demonstrate AST parsing functionality."""
    print("=" * 60)
    print("Example 1: AST Parser")
    print("=" * 60)
    
    from sva_toolkit.ast_parser import SVAASTParser
    
    parser = SVAASTParser()
    
    sva_code = """
    property ReqAckHandshake;
        @(posedge clk) disable iff (!rst_n)
        req |-> ##[1:3] ack;
    endproperty
    assert property (ReqAckHandshake);
    """
    
    structure = parser.parse(sva_code)
    
    print(f"Property Name: {structure.property_name}")
    print(f"Clock: {structure.clock_edge}({structure.clock_signal})")
    print(f"Reset: {structure.reset_signal} (Active Low: {structure.reset_active_low})")
    print(f"Implication Type: {structure.implication_type.value}")
    print(f"Antecedent: {structure.antecedent}")
    print(f"Consequent: {structure.consequent}")
    print(f"Signals: {[s.name for s in structure.signals]}")
    print(f"Delays: {[(d.min_cycles, d.max_cycles) for d in structure.delays]}")
    print()


def example_cot_builder():
    """Demonstrate CoT builder functionality."""
    print("=" * 60)
    print("Example 2: Chain-of-Thought Builder")
    print("=" * 60)
    
    from sva_toolkit.cot_builder import SVACoTBuilder
    
    builder = SVACoTBuilder()
    
    sva_code = """
    property BurstTransfer;
        @(posedge clk) disable iff (!rst_n)
        $rose(burst_start) |-> (data_valid)[*4] ##1 burst_done;
    endproperty
    """
    
    cot = builder.build(sva_code)
    
    # Print first 1500 characters
    print(cot[:1500])
    print("...(truncated)")
    print()


def example_dataset_builder_cot_only():
    """Demonstrate dataset building with CoT only (no LLM required)."""
    print("=" * 60)
    print("Example 3: Dataset Builder (CoT Only)")
    print("=" * 60)
    
    from sva_toolkit.dataset_builder import DatasetBuilder
    
    builder = DatasetBuilder()  # No LLM client needed for CoT
    
    input_data = [
        {"SVA": "property p1; @(posedge clk) req |-> ##[1:3] ack; endproperty"},
        {"SVA": "property p2; @(posedge clk) $rose(valid) |=> data_ready; endproperty"},
    ]
    
    entries = builder.build_dataset(
        input_data,
        generate_svad=False,  # Skip SVAD (requires LLM)
        generate_cot=True
    )
    
    print(f"Processed {len(entries)} entries")
    for i, entry in enumerate(entries):
        print(f"\nEntry {i+1}:")
        print(f"  SVA: {entry.SVA[:50]}...")
        print(f"  CoT: {entry.CoT[:100] if entry.CoT else 'None'}...")
    
    # Validate dataset
    report = builder.validate_dataset(entries)
    print(f"\nValidation Report:")
    print(f"  CoT Coverage: {report['cot_coverage']*100:.0f}%")
    print()


def example_utilities():
    """Demonstrate utility functions."""
    print("=" * 60)
    print("Example 4: Utility Functions")
    print("=" * 60)
    
    from sva_toolkit.utils import (
        clean_sva_code,
        validate_sva_syntax,
        extract_signals_from_expression,
        parse_delay_spec,
    )
    
    # Clean SVA code
    messy_code = "  property   p;   a  |->   b;   endproperty  "
    clean_code = clean_sva_code(messy_code)
    print(f"Cleaned code: '{clean_code}'")
    
    # Validate syntax
    valid_code = "property p; (a && b) |-> c; endproperty"
    is_valid, error = validate_sva_syntax(valid_code)
    print(f"Valid syntax: {is_valid}")
    
    invalid_code = "property p; (a && b |-> c; endproperty"  # Missing )
    is_valid, error = validate_sva_syntax(invalid_code)
    print(f"Invalid syntax: {is_valid}, Error: {error}")
    
    # Extract signals
    expr = "valid && ready && (data_out == expected)"
    signals = extract_signals_from_expression(expr)
    print(f"Extracted signals: {signals}")
    
    # Parse delay
    delay = parse_delay_spec("##[1:5]")
    print(f"Parsed delay: min={delay['min']}, max={delay['max']}")
    print()


def example_with_llm(base_url: str, model: str, api_key: str):
    """
    Demonstrate LLM-powered features.
    
    This example requires a valid LLM configuration.
    """
    print("=" * 60)
    print("Example 5: LLM-Powered Features")
    print("=" * 60)
    
    from sva_toolkit.dataset_builder import DatasetBuilder
    
    builder = DatasetBuilder.from_llm_config(
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    
    sva_code = """
    property ReqAckHandshake;
        @(posedge clk) disable iff (!rst_n)
        req |-> ##[1:3] ack;
    endproperty
    """
    
    svad = builder.generate_svad(sva_code)
    print(f"Generated SVAD:\n{svad}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SVA Toolkit - Example Usage")
    print("=" * 60 + "\n")
    
    # Examples that don't require external tools
    example_ast_parser()
    example_cot_builder()
    example_dataset_builder_cot_only()
    example_utilities()
    
    # LLM example (commented out - requires configuration)
    example_with_llm(
        base_url="https://api.siliconflow.cn/v1",
        model="Pro/deepseek-ai/DeepSeek-V3.2-Exp",
        api_key="sk-anwluomxfwjhiwpoyjhjnmwnfqobzbdjaigihjwjcvncjehq"
    )
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor more examples, see the CLI tools:")
    print("  sva-ast --help")
    print("  sva-cot --help")
    print("  sva-dataset --help")
    print("  sva-benchmark --help")
    print("  sva-implication --help")


if __name__ == "__main__":
    main()
