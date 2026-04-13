"""
Common utility functions for SVA Toolkit.
"""

import re
import os
import tempfile
import shutil
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


def clean_sva_code(code: str) -> str:
    """
    Clean and normalize SVA code.
    
    Args:
        code: Raw SVA code
        
    Returns:
        Cleaned SVA code
    """
    # Remove excessive whitespace
    code = re.sub(r'\s+', ' ', code)
    
    # Normalize line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    return code


def extract_sva_from_markdown(text: str) -> Optional[str]:
    """
    Extract SVA code from markdown code blocks.
    
    Args:
        text: Text potentially containing markdown code blocks
        
    Returns:
        Extracted SVA code or None
    """
    # Try to find systemverilog/verilog code blocks
    patterns = [
        r'```(?:systemverilog|verilog|sv)\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def wrap_sva_in_module(sva_code: str, module_name: str = "sva_wrapper") -> str:
    """
    Wrap SVA code in a module if not already wrapped.
    
    Args:
        sva_code: SVA code
        module_name: Name for the wrapper module
        
    Returns:
        SVA code wrapped in module
    """
    if 'module' in sva_code.lower():
        return sva_code
    
    return f"module {module_name};\n{sva_code}\nendmodule"


def extract_signals_from_expression(expr: str) -> List[str]:
    """
    Extract signal names from a Verilog expression.
    
    Args:
        expr: Verilog expression
        
    Returns:
        List of signal names
    """
    # Remove strings
    expr = re.sub(r'"[^"]*"', '', expr)
    
    # Remove numbers
    expr = re.sub(r"\b\d+'[bhd][\da-fA-F_]+\b", '', expr)
    expr = re.sub(r'\b\d+\b', '', expr)
    
    # Find identifiers
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)
    
    # Filter out keywords
    keywords = {
        'property', 'endproperty', 'sequence', 'endsequence',
        'assert', 'assume', 'cover', 'disable', 'iff',
        'posedge', 'negedge', 'or', 'and', 'not', 'if', 'else',
        'throughout', 'within', 'intersect', 'first_match',
        'always', 'eventually', 'nexttime', 'until', 'until_with',
        'module', 'endmodule', 'input', 'output', 'wire', 'reg', 'logic',
    }
    
    return [ident for ident in identifiers 
            if ident.lower() not in keywords and not ident.startswith('$')]


def create_temp_dir(prefix: str = "sva_toolkit_") -> str:
    """
    Create a temporary directory.
    
    Args:
        prefix: Prefix for the directory name
        
    Returns:
        Path to the temporary directory
    """
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_temp_dir(path: str) -> None:
    """
    Clean up a temporary directory.
    
    Args:
        path: Path to the directory
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def validate_sva_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Basic SVA syntax validation.
    
    Args:
        code: SVA code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for balanced parentheses
    paren_count = 0
    bracket_count = 0
    
    for char in code:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        
        if paren_count < 0:
            return False, "Unbalanced parentheses: extra ')'"
        if bracket_count < 0:
            return False, "Unbalanced brackets: extra ']'"
    
    if paren_count != 0:
        return False, f"Unbalanced parentheses: {paren_count} unclosed '('"
    if bracket_count != 0:
        return False, f"Unbalanced brackets: {bracket_count} unclosed '['"
    
    # Check for property/endproperty matching
    if 'property' in code.lower():
        prop_count = len(re.findall(r'\bproperty\b', code, re.IGNORECASE))
        endprop_count = len(re.findall(r'\bendproperty\b', code, re.IGNORECASE))
        if prop_count != endprop_count:
            return False, f"Mismatched property/endproperty: {prop_count} vs {endprop_count}"
    
    # Check for sequence/endsequence matching
    if 'sequence' in code.lower():
        seq_count = len(re.findall(r'\bsequence\b', code, re.IGNORECASE))
        endseq_count = len(re.findall(r'\bendsequence\b', code, re.IGNORECASE))
        if seq_count != endseq_count:
            return False, f"Mismatched sequence/endsequence: {seq_count} vs {endseq_count}"
    
    return True, None


def format_sva_code(code: str, indent: int = 4) -> str:
    """
    Format SVA code with proper indentation.
    
    Args:
        code: SVA code to format
        indent: Number of spaces for indentation
        
    Returns:
        Formatted SVA code
    """
    lines = code.strip().split('\n')
    formatted_lines = []
    current_indent = 0
    indent_str = ' ' * indent
    
    # Keywords that increase indent
    indent_increase = {'property', 'sequence', 'module', 'begin'}
    # Keywords that decrease indent
    indent_decrease = {'endproperty', 'endsequence', 'endmodule', 'end'}
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
        
        # Check for indent decrease
        first_word = stripped.split()[0].lower() if stripped.split() else ''
        if first_word in indent_decrease and current_indent > 0:
            current_indent -= 1
        
        # Add the line with current indentation
        formatted_lines.append(indent_str * current_indent + stripped)
        
        # Check for indent increase
        if first_word in indent_increase:
            current_indent += 1
    
    return '\n'.join(formatted_lines)


def parse_delay_spec(spec: str) -> Dict[str, Any]:
    """
    Parse a delay specification string.
    
    Args:
        spec: Delay specification (e.g., "##1", "##[1:3]", "##[1:$]")
        
    Returns:
        Dictionary with delay information
    """
    result = {
        'type': 'exact',
        'min': 0,
        'max': 0,
        'unbounded': False,
    }
    
    # Exact delay: ##N
    exact_match = re.match(r'##(\d+)', spec)
    if exact_match:
        n = int(exact_match.group(1))
        result['min'] = n
        result['max'] = n
        return result
    
    # Range delay: ##[N:M] or ##[N:$]
    range_match = re.match(r'##\[(\d+):(\d+|\$)\]', spec)
    if range_match:
        result['type'] = 'range'
        result['min'] = int(range_match.group(1))
        max_str = range_match.group(2)
        if max_str == '$':
            result['unbounded'] = True
            result['max'] = None
        else:
            result['max'] = int(max_str)
        return result
    
    return result


def generate_signal_declaration(
    name: str,
    width: int = 1,
    direction: str = "input",
    sig_type: str = "wire"
) -> str:
    """
    Generate a Verilog signal declaration.
    
    Args:
        name: Signal name
        width: Bit width
        direction: input/output/inout
        sig_type: wire/reg/logic
        
    Returns:
        Signal declaration string
    """
    if width > 1:
        return f"{direction} {sig_type} [{width-1}:0] {name}"
    return f"{direction} {sig_type} {name}"


def merge_datasets(datasets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of datasets to merge
        
    Returns:
        Merged dataset
    """
    merged = []
    seen_svas = set()
    
    for dataset in datasets:
        for item in dataset:
            sva = item.get('SVA', '')
            # Use cleaned SVA as dedup key
            clean_sva = clean_sva_code(sva)
            if clean_sva and clean_sva not in seen_svas:
                seen_svas.add(clean_sva)
                merged.append(item)
    
    return merged


def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train, val, test) datasets
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    data = dataset.copy()
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return data[:train_end], data[train_end:val_end], data[val_end:]
