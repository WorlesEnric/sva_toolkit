"""
SVA AST Parser - Wrapper around Verible for extracting structured SVA information.
"""

import json
import subprocess
import re
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ImplicationType(Enum):
    """Type of implication in SVA."""
    OVERLAPPING = "|->"  # Overlapping implication
    NON_OVERLAPPING = "|=>"  # Non-overlapping implication
    NONE = None  # No implication (simple property)


class TemporalOperator(Enum):
    """Temporal operators in SVA."""
    DELAY = "##"
    REPETITION_CONSECUTIVE = "[*"
    REPETITION_GOTO = "[->"
    REPETITION_NON_CONSECUTIVE = "[="
    THROUGHOUT = "throughout"
    WITHIN = "within"
    INTERSECT = "intersect"
    AND = "and"
    OR = "or"
    NOT = "not"
    FIRST_MATCH = "first_match"
    IF_ELSE = "if"


@dataclass
class Signal:
    """Represents a signal in SVA."""
    name: str
    is_clock: bool = False
    is_reset: bool = False
    width: Optional[int] = None
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Signal):
            return self.name == other.name
        return False


@dataclass
class BuiltinFunction:
    """Represents a built-in function call in SVA."""
    name: str  # e.g., $rose, $fell, $stable, $past, $onehot
    arguments: List[str] = field(default_factory=list)


@dataclass
class DelayRange:
    """Represents a delay specification."""
    min_cycles: int
    max_cycles: Optional[int] = None  # None means exact delay, $ means unbounded
    is_unbounded: bool = False


@dataclass 
class Sequence:
    """Represents a sequence in SVA."""
    expression: str
    delays: List[DelayRange] = field(default_factory=list)
    repetitions: List[str] = field(default_factory=list)
    sub_sequences: List["Sequence"] = field(default_factory=list)


@dataclass
class SVAStructure:
    """Structured representation of an SVA property/assertion."""
    
    # Original code
    raw_code: str
    
    # Property information
    property_name: Optional[str] = None
    is_assertion: bool = True
    is_assumption: bool = False
    is_cover: bool = False
    
    # Clock and reset
    clock_signal: Optional[str] = None
    clock_edge: str = "posedge"  # posedge or negedge
    reset_signal: Optional[str] = None
    reset_active_low: bool = True
    disable_condition: Optional[str] = None
    
    # Implication
    implication_type: ImplicationType = ImplicationType.NONE
    antecedent: Optional[str] = None
    consequent: Optional[str] = None
    
    # Signals and functions
    signals: Set[Signal] = field(default_factory=set)
    builtin_functions: List[BuiltinFunction] = field(default_factory=list)
    
    # Temporal information
    delays: List[DelayRange] = field(default_factory=list)
    temporal_operators: List[TemporalOperator] = field(default_factory=list)
    
    # Raw Verible AST (for advanced usage)
    verible_ast: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "raw_code": self.raw_code,
            "property_name": self.property_name,
            "is_assertion": self.is_assertion,
            "is_assumption": self.is_assumption,
            "is_cover": self.is_cover,
            "clock_signal": self.clock_signal,
            "clock_edge": self.clock_edge,
            "reset_signal": self.reset_signal,
            "reset_active_low": self.reset_active_low,
            "disable_condition": self.disable_condition,
            "implication_type": self.implication_type.value if self.implication_type else None,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "signals": [{"name": s.name, "is_clock": s.is_clock, "is_reset": s.is_reset} 
                       for s in self.signals],
            "builtin_functions": [{"name": f.name, "arguments": f.arguments} 
                                 for f in self.builtin_functions],
            "delays": [{"min": d.min_cycles, "max": d.max_cycles, "unbounded": d.is_unbounded}
                      for d in self.delays],
            "temporal_operators": [op.value for op in self.temporal_operators],
        }


class SVAASTParser:
    """
    Parser for SystemVerilog Assertions using Verible.
    
    This class wraps Verible's native parsing and extracts key structural
    information about SVA properties and assertions.
    """

    # Built-in SVA functions
    BUILTIN_FUNCTIONS = {
        "$rose", "$fell", "$stable", "$changed", "$past",
        "$onehot", "$onehot0", "$isunknown", "$countones",
        "$sampled", "$bits"
    }
    
    # Common clock signal patterns
    CLOCK_PATTERNS = re.compile(r'\b(clk|clock|CLK|CLOCK)\w*\b', re.IGNORECASE)
    
    # Common reset signal patterns
    RESET_PATTERNS = re.compile(r'\b(rst|reset|RST|RESET)\w*\b', re.IGNORECASE)

    def __init__(self, verible_path: str = None, require_verible: bool = False):
        """
        Initialize the parser.
        
        Args:
            verible_path: Path to verible-verilog-syntax binary
            require_verible: If True, raise error if Verible not found
        """
        if verible_path is None:
            import pathlib
            # Determine paths relative to this file
            # src/sva_toolkit/ast_parser/parser.py -> root is 4 levels up
            project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            verible_path = project_root / "3rd_party" / "verible_bin" / "verible-verilog-syntax"
        self.verible_path = str(verible_path)
        self.verible_available = self._check_verible()
        
        if require_verible and not self.verible_available:
            raise RuntimeError(
                f"Verible not found at '{self.verible_path}'. "
                "Please install Verible or provide the correct path."
            )

    def _check_verible(self) -> bool:
        """Check if Verible is accessible."""
        try:
            result = subprocess.run(
                [self.verible_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _run_verible(self, code: str) -> Dict[str, Any]:
        """
        Run Verible on the given code and return the JSON AST.
        
        Args:
            code: SystemVerilog code to parse
            
        Returns:
            Parsed JSON AST from Verible (empty dict if Verible unavailable)
        """
        if not self.verible_available:
            return {}
        
        # Wrap code in a module if it's just an assertion
        if not code.strip().startswith("module"):
            wrapped_code = f"module sva_wrapper;\n{code}\nendmodule"
        else:
            wrapped_code = code
            
        try:
            result = subprocess.run(
                [self.verible_path, "--export_json", "-"],
                input=wrapped_code,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout.strip():
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        
        return {}

    def _extract_signals(self, code: str, property_body: Optional[str] = None) -> Set[Signal]:
        """
        Extract all signal references from SVA code.
        
        Args:
            code: SVA code (used for clock/reset extraction)
            property_body: Optional extracted property body to limit signal extraction
            
        Returns:
            Set of Signal objects
        """
        signals = set()
        
        # Use property body if provided, otherwise use full code
        target_code = property_body if property_body else code
        
        # Remove strings and comments
        clean_code = re.sub(r'"[^"]*"', '', target_code)
        clean_code = re.sub(r'//.*$', '', clean_code, flags=re.MULTILINE)
        clean_code = re.sub(r'/\*.*?\*/', '', clean_code, flags=re.DOTALL)
        
        # Remove built-in function names (but keep their arguments)
        # Replace $funcname( with just ( to keep the arguments
        clean_code = re.sub(r'\$[a-zA-Z_][a-zA-Z0-9_]*\s*\(', '(', clean_code)
        
        # Remove keywords and built-in functions
        keywords = {
            'property', 'endproperty', 'sequence', 'endsequence',
            'assert', 'assume', 'cover', 'disable', 'iff',
            'posedge', 'negedge', 'or', 'and', 'not', 'if', 'else',
            'throughout', 'within', 'intersect', 'first_match',
            'always', 'eventually', 'nexttime', 'until', 'until_with',
            's_always', 's_eventually', 's_nexttime', 's_until', 's_until_with',
            'strong', 'weak', 'accept_on', 'reject_on', 'sync_accept_on', 'sync_reject_on',
            'module', 'endmodule', 'input', 'output', 'wire', 'reg', 'logic',
            'error', 'warning', 'info', 'fatal',  # System tasks
        }
        
        # Find all identifiers (excluding numbers, keywords, and built-ins)
        identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(identifier_pattern, clean_code)
        
        for match in matches:
            if match.lower() not in keywords and not match.startswith('$'):
                is_clock = bool(self.CLOCK_PATTERNS.match(match))
                is_reset = bool(self.RESET_PATTERNS.match(match))
                signals.add(Signal(name=match, is_clock=is_clock, is_reset=is_reset))
        
        return signals

    def _extract_builtin_functions(self, code: str) -> List[BuiltinFunction]:
        """
        Extract built-in function calls from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            List of BuiltinFunction objects
        """
        functions = []
        
        for func_name in self.BUILTIN_FUNCTIONS:
            # Pattern to match function calls with arguments
            pattern = rf'\{func_name}\s*\(([^)]*)\)'
            matches = re.findall(pattern, code, re.IGNORECASE)
            
            for args_str in matches:
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                functions.append(BuiltinFunction(name=func_name, arguments=args))
        
        return functions

    def _extract_clock_info(self, code: str) -> Tuple[Optional[str], str]:
        """
        Extract clock signal and edge from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            Tuple of (clock_signal, clock_edge)
        """
        # Pattern for @(posedge/negedge signal)
        pattern = r'@\s*\(\s*(posedge|negedge)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
        match = re.search(pattern, code)
        
        if match:
            return match.group(2), match.group(1)
        
        return None, "posedge"

    def _extract_disable_condition(self, code: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Extract disable condition from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            Tuple of (disable_condition, reset_signal, is_active_low)
        """
        # Pattern for disable iff (condition)
        pattern = r'disable\s+iff\s*\(\s*([^)]+)\s*\)'
        match = re.search(pattern, code)
        
        if match:
            condition = match.group(1).strip()
            # Check if it's a negated reset (active low)
            is_active_low = condition.startswith('!') or condition.startswith('~')
            
            # Extract reset signal name
            reset_match = re.search(r'[!~]?\s*([a-zA-Z_][a-zA-Z0-9_]*)', condition)
            reset_signal = reset_match.group(1) if reset_match else None
            
            return condition, reset_signal, is_active_low
        
        return None, None, True

    def _extract_property_body(self, code: str) -> str:
        """
        Extract only the property body from SVA code.
        
        Handles:
        - Named property: property name; ... endproperty
        - Inline assertion: assert property (...)
        
        Args:
            code: SVA code
            
        Returns:
            Property body content only
        """
        # Try to extract named property body (between property...endproperty)
        property_match = re.search(
            r'property\s+\w+\s*;([\s\S]*?)endproperty',
            code,
            re.IGNORECASE
        )
        if property_match:
            return property_match.group(1).strip()
        
        # Try to extract inline assertion property body
        # Matches: assert/assume/cover property (...)
        inline_match = re.search(
            r'(?:assert|assume|cover)\s+property\s*\(([^;]+)\)',
            code,
            re.IGNORECASE
        )
        if inline_match:
            return inline_match.group(1).strip()
        
        # Fallback: return original code
        return code

    def _extract_implication(self, code: str) -> Tuple[ImplicationType, Optional[str], Optional[str]]:
        """
        Extract implication type and antecedent/consequent from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            Tuple of (implication_type, antecedent, consequent)
        """
        # First extract the property body to avoid including assertion statements
        property_body = self._extract_property_body(code)
        
        # Look for non-overlapping implication first (|=>)
        if '|=>' in property_body:
            parts = property_body.split('|=>', 1)
            if len(parts) == 2:
                # Clean up antecedent (remove clock, disable iff)
                antecedent = parts[0]
                antecedent = re.sub(r'@\s*\([^)]+\)', '', antecedent)
                antecedent = re.sub(r'disable\s+iff\s*\([^)]+\)', '', antecedent)
                antecedent = antecedent.strip().strip(';').strip()
                
                # Clean up consequent
                consequent = parts[1].strip().strip(';').strip()
                
                return ImplicationType.NON_OVERLAPPING, antecedent, consequent
        
        # Look for overlapping implication (|->)
        if '|->' in property_body:
            parts = property_body.split('|->', 1)
            if len(parts) == 2:
                antecedent = parts[0]
                antecedent = re.sub(r'@\s*\([^)]+\)', '', antecedent)
                antecedent = re.sub(r'disable\s+iff\s*\([^)]+\)', '', antecedent)
                antecedent = antecedent.strip().strip(';').strip()
                
                consequent = parts[1].strip().strip(';').strip()
                
                return ImplicationType.OVERLAPPING, antecedent, consequent
        
        return ImplicationType.NONE, None, None

    def _extract_delays(self, code: str) -> List[DelayRange]:
        """
        Extract delay specifications from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            List of DelayRange objects
        """
        delays = []
        
        # Pattern for ##N (exact delay)
        exact_pattern = r'##(\d+)'
        for match in re.finditer(exact_pattern, code):
            n = int(match.group(1))
            delays.append(DelayRange(min_cycles=n, max_cycles=n))
        
        # Pattern for ##[N:M] (range delay)
        range_pattern = r'##\[(\d+):(\d+|\$)\]'
        for match in re.finditer(range_pattern, code):
            min_cycles = int(match.group(1))
            max_str = match.group(2)
            if max_str == '$':
                delays.append(DelayRange(min_cycles=min_cycles, is_unbounded=True))
            else:
                delays.append(DelayRange(min_cycles=min_cycles, max_cycles=int(max_str)))
        
        return delays

    def _extract_temporal_operators(self, code: str) -> List[TemporalOperator]:
        """
        Extract temporal operators from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            List of TemporalOperator values
        """
        operators = []
        
        if '##' in code:
            operators.append(TemporalOperator.DELAY)
        if '[*' in code:
            operators.append(TemporalOperator.REPETITION_CONSECUTIVE)
        if '[->' in code:
            operators.append(TemporalOperator.REPETITION_GOTO)
        if '[=' in code:
            operators.append(TemporalOperator.REPETITION_NON_CONSECUTIVE)
        if 'throughout' in code.lower():
            operators.append(TemporalOperator.THROUGHOUT)
        if 'within' in code.lower():
            operators.append(TemporalOperator.WITHIN)
        if 'intersect' in code.lower():
            operators.append(TemporalOperator.INTERSECT)
        if re.search(r'\band\b', code, re.IGNORECASE):
            operators.append(TemporalOperator.AND)
        if re.search(r'\bor\b', code, re.IGNORECASE):
            operators.append(TemporalOperator.OR)
        if re.search(r'\bnot\b', code, re.IGNORECASE):
            operators.append(TemporalOperator.NOT)
        if 'first_match' in code.lower():
            operators.append(TemporalOperator.FIRST_MATCH)
        
        return operators

    def _extract_property_name(self, code: str) -> Optional[str]:
        """
        Extract property name from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            Property name or None
        """
        pattern = r'property\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        match = re.search(pattern, code)
        return match.group(1) if match else None

    def _determine_assertion_type(self, code: str) -> Tuple[bool, bool, bool]:
        """
        Determine if code is assert, assume, or cover.
        
        Args:
            code: SVA code
            
        Returns:
            Tuple of (is_assertion, is_assumption, is_cover)
        """
        is_assertion = 'assert' in code.lower()
        is_assumption = 'assume' in code.lower()
        is_cover = 'cover' in code.lower()
        
        # Default to assertion if none specified
        if not any([is_assertion, is_assumption, is_cover]):
            is_assertion = True
            
        return is_assertion, is_assumption, is_cover

    def parse(self, code: str) -> SVAStructure:
        """
        Parse SVA code and extract structured information.
        
        Args:
            code: SVA code (property, assertion, or assumption)
            
        Returns:
            SVAStructure containing extracted information
        """
        # Get Verible AST
        verible_ast = self._run_verible(code)
        
        # Extract property body first (for cleaner signal/operator extraction)
        property_body = self._extract_property_body(code)
        
        # Extract components
        signals = self._extract_signals(code, property_body)
        builtin_functions = self._extract_builtin_functions(property_body)
        clock_signal, clock_edge = self._extract_clock_info(code)
        disable_condition, reset_signal, reset_active_low = self._extract_disable_condition(code)
        implication_type, antecedent, consequent = self._extract_implication(code)
        delays = self._extract_delays(property_body)
        temporal_operators = self._extract_temporal_operators(property_body)
        property_name = self._extract_property_name(code)
        is_assertion, is_assumption, is_cover = self._determine_assertion_type(code)
        
        # Mark clock and reset signals
        for signal in signals:
            if clock_signal and signal.name == clock_signal:
                signal.is_clock = True
            if reset_signal and signal.name == reset_signal:
                signal.is_reset = True
        
        return SVAStructure(
            raw_code=code,
            property_name=property_name,
            is_assertion=is_assertion,
            is_assumption=is_assumption,
            is_cover=is_cover,
            clock_signal=clock_signal,
            clock_edge=clock_edge,
            reset_signal=reset_signal,
            reset_active_low=reset_active_low,
            disable_condition=disable_condition,
            implication_type=implication_type,
            antecedent=antecedent,
            consequent=consequent,
            signals=signals,
            builtin_functions=builtin_functions,
            delays=delays,
            temporal_operators=temporal_operators,
            verible_ast=verible_ast,
        )

    def parse_file(self, filepath: str) -> List[SVAStructure]:
        """
        Parse a SystemVerilog file and extract all SVA structures.
        
        Args:
            filepath: Path to the .sv file
            
        Returns:
            List of SVAStructure objects
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        structures = []
        
        # Find all property definitions
        property_pattern = r'property\s+\w+[\s\S]*?endproperty'
        for match in re.finditer(property_pattern, content):
            structures.append(self.parse(match.group()))
        
        # Find all inline assertions/assumptions/covers
        inline_pattern = r'(assert|assume|cover)\s+property\s*\([^;]+\)\s*;'
        for match in re.finditer(inline_pattern, content):
            structures.append(self.parse(match.group()))
        
        return structures

    def get_all_signals(self, code: str) -> List[str]:
        """
        Get all signal names from SVA code.
        
        Args:
            code: SVA code
            
        Returns:
            List of signal names
        """
        structure = self.parse(code)
        return [s.name for s in structure.signals]

    def to_json(self, structure: SVAStructure) -> str:
        """
        Convert SVAStructure to JSON string.
        
        Args:
            structure: SVAStructure object
            
        Returns:
            JSON string representation
        """
        return json.dumps(structure.to_dict(), indent=2)
