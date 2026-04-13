"""
SVA Templates - SystemVerilog module templates for wrapping generated assertions.

This module provides template functions for generating complete SystemVerilog
modules containing SVA properties and assertions.
"""

from typing import List, Optional


def generate_sv_module(
    module_name: str,
    signals: List[str],
    properties: List[str],
    clock_signal: str = "clk",
    reset_signal: str = "rst_n",
    include_assertions: bool = True
) -> str:
    """
    @brief Generate a complete SystemVerilog module wrapping SVA properties.
    @param module_name Name of the module
    @param signals List of signal names to declare
    @param properties List of property code blocks
    @param clock_signal Clock signal name
    @param reset_signal Reset signal name
    @param include_assertions Whether to generate assert property statements
    @return Complete SystemVerilog module code
    """
    signal_decls = "\n  ".join([f"logic {sig};" for sig in signals])
    # Generate assert property statements for each property
    if include_assertions:
        assertions = "\n  ".join([
            f"assert_p{i}: assert property (p_gen_{i});"
            for i in range(len(properties))
        ])
    else:
        assertions = "// No assertion instances"
    # Combine properties
    properties_block = "\n\n  ".join(properties)
    template = f"""module {module_name} (
    input logic {clock_signal},
    input logic {reset_signal}
);

  // Signal Declarations
  {signal_decls}

  // Generated Properties
  {properties_block}

  // Assertion Instances
  {assertions}

endmodule
"""
    return template


def generate_minimal_wrapper(
    properties: List[str],
    clock_signal: str = "clk"
) -> str:
    """
    @brief Generate a minimal module wrapper for syntax validation.
    @param properties List of property code blocks
    @param clock_signal Clock signal name
    @return Minimal SystemVerilog module for validation
    """
    properties_block = "\n".join(properties)
    return f"""module sva_check (
    input logic {clock_signal}
);
  logic dummy;
{properties_block}
endmodule
"""


def generate_assertion_only(
    property_name: str,
    property_body: str,
    clock_signal: str = "clk",
    clock_edge: str = "posedge"
) -> str:
    """
    @brief Generate a standalone assertion without property declaration.
    @param property_name Name for the assertion label
    @param property_body Property body expression
    @param clock_signal Clock signal name
    @param clock_edge Clock edge (posedge/negedge)
    @return Inline assertion statement
    """
    return (
        f"{property_name}: assert property "
        f"(@({clock_edge} {clock_signal}) {property_body});"
    )


def generate_cover_property(
    property_name: str,
    property_body: str,
    clock_signal: str = "clk",
    clock_edge: str = "posedge"
) -> str:
    """
    @brief Generate a cover property statement.
    @param property_name Name for the cover label
    @param property_body Property body expression
    @param clock_signal Clock signal name
    @param clock_edge Clock edge (posedge/negedge)
    @return Cover property statement
    """
    return (
        f"{property_name}: cover property "
        f"(@({clock_edge} {clock_signal}) {property_body});"
    )


def generate_assume_property(
    property_name: str,
    property_body: str,
    clock_signal: str = "clk",
    clock_edge: str = "posedge"
) -> str:
    """
    @brief Generate an assume property statement.
    @param property_name Name for the assume label
    @param property_body Property body expression
    @param clock_signal Clock signal name
    @param clock_edge Clock edge (posedge/negedge)
    @return Assume property statement
    """
    return (
        f"{property_name}: assume property "
        f"(@({clock_edge} {clock_signal}) {property_body});"
    )
