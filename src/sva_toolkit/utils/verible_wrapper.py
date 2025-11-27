"""
Verible wrapper utilities for SVA Toolkit.

Provides helper functions for working with Verible tools.
"""

import subprocess
import json
import shutil
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class VeribleConfig:
    """Configuration for Verible tools."""
    syntax_path: str = "verible-verilog-syntax"
    format_path: str = "verible-verilog-format"
    lint_path: str = "verible-verilog-lint"
    timeout: int = 30


class VeribleWrapper:
    """
    Wrapper for Verible tools.
    
    Provides convenient access to Verible's parsing, formatting, and linting
    capabilities.
    """

    def __init__(self, config: Optional[VeribleConfig] = None):
        """
        Initialize the Verible wrapper.
        
        Args:
            config: Verible configuration
        """
        self.config = config or VeribleConfig()
        self._check_installation()

    def _check_installation(self) -> None:
        """Check if Verible tools are installed."""
        if not shutil.which(self.config.syntax_path):
            raise RuntimeError(
                f"Verible syntax tool not found: {self.config.syntax_path}\n"
                "Please install Verible from https://github.com/chipsalliance/verible"
            )

    def parse_to_json(self, code: str) -> Dict[str, Any]:
        """
        Parse Verilog/SystemVerilog code and return JSON AST.
        
        Args:
            code: Code to parse
            
        Returns:
            JSON AST from Verible
        """
        result = subprocess.run(
            [self.config.syntax_path, "--export_json", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self.config.timeout
        )
        
        if result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {}
        return {}

    def parse_to_tree(self, code: str) -> str:
        """
        Parse code and return text tree representation.
        
        Args:
            code: Code to parse
            
        Returns:
            Text tree representation
        """
        result = subprocess.run(
            [self.config.syntax_path, "--printtree", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self.config.timeout
        )
        
        return result.stdout

    def get_tokens(self, code: str) -> str:
        """
        Get token stream from code.
        
        Args:
            code: Code to tokenize
            
        Returns:
            Token stream representation
        """
        result = subprocess.run(
            [self.config.syntax_path, "--printtokens", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self.config.timeout
        )
        
        return result.stdout

    def format_code(self, code: str) -> str:
        """
        Format Verilog/SystemVerilog code.
        
        Args:
            code: Code to format
            
        Returns:
            Formatted code
        """
        if not shutil.which(self.config.format_path):
            return code  # Return original if formatter not available
        
        result = subprocess.run(
            [self.config.format_path, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self.config.timeout
        )
        
        if result.returncode == 0:
            return result.stdout
        return code

    def lint(self, code: str) -> List[Dict[str, Any]]:
        """
        Lint Verilog/SystemVerilog code.
        
        Args:
            code: Code to lint
            
        Returns:
            List of lint findings
        """
        if not shutil.which(self.config.lint_path):
            return []
        
        result = subprocess.run(
            [self.config.lint_path, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self.config.timeout
        )
        
        findings = []
        for line in result.stdout.split('\n'):
            if line.strip():
                findings.append({'message': line})
        
        return findings

    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Check if code has valid syntax.
        
        Args:
            code: Code to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        result = subprocess.run(
            [self.config.syntax_path, "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=self.config.timeout
        )
        
        if result.returncode == 0:
            return True, ""
        
        return False, result.stderr


def find_verible() -> Optional[str]:
    """
    Find Verible installation path.
    
    Returns:
        Path to verible-verilog-syntax or None
    """
    return shutil.which("verible-verilog-syntax")


def get_verible_version() -> Optional[str]:
    """
    Get Verible version string.
    
    Returns:
        Version string or None
    """
    verible_path = find_verible()
    if not verible_path:
        return None
    
    try:
        result = subprocess.run(
            [verible_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception:
        return None
