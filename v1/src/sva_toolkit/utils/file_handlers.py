"""
File handling utilities for SVA Toolkit.
"""

import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load JSON from file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to output file
        indent: JSON indentation
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_sva_file(filepath: Union[str, Path]) -> str:
    """
    Load SVA code from a file.
    
    Args:
        filepath: Path to SVA/SV file
        
    Returns:
        SVA code content
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_sva_file(code: str, filepath: Union[str, Path]) -> None:
    """
    Save SVA code to a file.
    
    Args:
        code: SVA code
        filepath: Path to output file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)


def load_dataset(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a dataset from JSON file.
    
    Args:
        filepath: Path to dataset JSON file
        
    Returns:
        List of dataset entries
    """
    data = load_json(filepath)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        raise ValueError(f"Invalid dataset format in {filepath}")


def save_dataset(
    data: List[Dict[str, Any]],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a dataset to JSON file.
    
    Args:
        data: Dataset entries
        filepath: Path to output file
        metadata: Optional metadata to include
    """
    if metadata:
        output = {
            'metadata': metadata,
            'data': data
        }
    else:
        output = data
    
    save_json(output, filepath)


def find_sva_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> List[Path]:
    """
    Find all SVA/SV files in a directory.
    
    Args:
        directory: Directory to search
        extensions: File extensions to include (default: ['.sv', '.sva', '.svh'])
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.sv', '.sva', '.svh']
    
    directory = Path(directory)
    files = []
    
    if recursive:
        for ext in extensions:
            files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            files.extend(directory.glob(f'*{ext}'))
    
    return sorted(files)


def create_sby_config(
    module_file: str,
    module_name: str,
    mode: str = "prove",
    depth: int = 20,
    engine: str = "smtbmc"
) -> str:
    """
    Create a SymbiYosys configuration file content.
    
    Args:
        module_file: Name of the module file
        module_name: Name of the top module
        mode: Verification mode (prove, cover, live, bmc)
        depth: Proof depth
        engine: Verification engine
        
    Returns:
        SBY configuration file content
    """
    return f"""[options]
mode {mode}
depth {depth}

[engines]
{engine}

[script]
read -formal {module_file}
prep -top {module_name}

[files]
{module_file}
"""


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Try to find pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def get_templates_dir() -> Path:
    """
    Get the templates directory.
    
    Returns:
        Path to templates directory
    """
    root = get_project_root()
    return root / 'templates'
