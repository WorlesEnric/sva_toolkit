"""
SVA Implication Checker - Verify implication relationships between SVA pairs.

Uses EBMC for formal verification.
"""

import os
import re
import tempfile
import subprocess
import shutil
from typing import Optional, Set, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sva_toolkit.ast_parser.parser import SVAASTParser, SVAStructure, ImplicationType


class ImplicationResult(Enum):
    """Result of implication check."""
    IMPLIES = "implies"  # SVA1 implies SVA2
    NOT_IMPLIES = "not_implies"  # SVA1 does not imply SVA2
    EQUIVALENT = "equivalent"  # SVA1 and SVA2 are equivalent
    ERROR = "error"  # Internal tool/verification error
    SYNTAX_ERROR = "syntax_error"  # SVA syntax error in input
    TIMEOUT = "timeout"  # Verification timed out


class SVASyntaxError(Exception):
    """Exception raised when SVA syntax is invalid."""
    def __init__(self, message: str, log: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.log = log


@dataclass
class CheckResult:
    """Result of an implication check."""
    result: ImplicationResult
    message: str
    counterexample: Optional[str] = None
    log: Optional[str] = None
    module: Optional[str] = None  # Generated Verilog module for this check


class SVAImplicationChecker:
    """
    Checker for verifying implication relationships between SVA pairs.
    
    Uses EBMC to formally verify whether SVA1 implies SVA2.
    The approach:
    1. Create a trivial RTL module with all relevant signals
    2. Assume SVA1 holds
    3. Assert SVA2 should hold
    
    If the assertion passes, SVA1 implies SVA2.
    """

    # Built-in SVA functions that need special handling
    BUILTIN_FUNCTIONS = {
        "$rose", "$fell", "$stable", "$changed", "$past",
        "$onehot", "$onehot0", "$isunknown", "$countones",
        "$sampled", "$bits"
    }
    
    # Functions that EBMC supports natively
    EBMC_SUPPORTED_FUNCTIONS = {
        "$rose", "$fell", "$stable", "$past", "$onehot", "$onehot0"
    }
    
    # SVA constructs not supported by EBMC
    UNSUPPORTED_CONSTRUCTS = {
        "if_else_property": r'\bif\s*\([^)]+\)\s*[^|]*\|[-=]>.*\belse\b',
    }

    MODULE_TEMPLATE = """
module sva_checker (
    input wire {clock_name},
    input wire rst_n,
{signal_declarations}
);

{past_signal_declarations}
{past_signal_logic}

    // ----------------------------------------------------------------
    // INLINE SVA WRAPPERS
    // ----------------------------------------------------------------

    // 1. Assume the Antecedent (User SVA 1)
    //    We assume this property holds true to constrain the state space.
    assume property (@({clock_edge} {clock_name}) disable iff (!rst_n)
        {antecedent}
    );

    // 2. Assert the Consequent (User SVA 2)
    //    We check if this holds true under the assumption above.
    assert property (@({clock_edge} {clock_name}) disable iff (!rst_n)
        {consequent}
    );

    // 3. Cover the Antecedent
    //    (Optional) Check if the antecedent is even possible.
    cover_antecedent: cover property (
        @({clock_edge} {clock_name}) disable iff (!rst_n)
        {antecedent}
    );

endmodule
"""

    def __init__(
        self,
        ebmc_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        depth: int = 20,
        keep_files: bool = False,
        verbose: bool = False,
        require_ebmc: bool = False,
        timeout: int = 300,
    ):
        """
        Initialize the implication checker.
        
        Args:
            ebmc_path: Path to ebmc binary (default: search in PATH or bundled in 3rd_party)
            work_dir: Working directory for verification files (default: temp dir)
            depth: Proof depth for bounded model checking
            keep_files: Keep generated files after verification
            verbose: Print debug information
            require_ebmc: If True, raise error if EBMC not found (default: False)
            timeout: Timeout in seconds for each EBMC verification (default: 300)
        """
        # Determine paths relative to this file
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        default_ebmc = project_root / "3rd_party" / "ebmc" / "bin" / "ebmc"

        if ebmc_path:
            self.ebmc_path = ebmc_path
        elif default_ebmc.exists():
            self.ebmc_path = str(default_ebmc)
        else:
            # Try to find ebmc in PATH
            self.ebmc_path = shutil.which("ebmc") or "ebmc"
        
        self.work_dir = work_dir
        self.depth = depth
        self.keep_files = keep_files
        self.verbose = verbose
        self.timeout = timeout
        self.parser = SVAASTParser()
        self.ebmc_available = False
        
        # Only verify tools if explicitly required
        if require_ebmc:
            self._verify_tools()
            self.ebmc_available = True
        else:
            # Check silently
            self.ebmc_available = self._check_ebmc_available()

    def _check_ebmc_available(self) -> bool:
        """Check if EBMC is available without raising an error."""
        try:
            result = subprocess.run(
                [self.ebmc_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _verify_tools(self) -> None:
        """Verify that required tools are accessible."""
        try:
            result = subprocess.run(
                [self.ebmc_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"EBMC not found at '{self.ebmc_path}'. "
                "Please install EBMC or provide the correct path."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("EBMC verification timed out.")

    def _parse_sva(self, sva_code: str) -> SVAStructure:
        """
        Parse SVA code using the AST parser.
        
        Args:
            sva_code: SVA code string
            
        Returns:
            Parsed SVA structure
        """
        return self.parser.parse(sva_code)

    def _extract_clock_spec(self, sva_code: str) -> Tuple[str, str]:
        """
        Extract the clock specification from SVA code.
        
        Args:
            sva_code: SVA code string
            
        Returns:
            Tuple of (clock_edge, clock_name). Defaults to ("posedge", "clk") if not found.
        """
        code = sva_code.strip().strip('`').strip()
        # Match @(posedge clk_name) or @(negedge clk_name)
        clock_match = re.search(
            r'@\s*\(\s*(posedge|negedge)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
            code,
            re.IGNORECASE
        )
        if clock_match:
            edge = clock_match.group(1).lower()
            clock_name = clock_match.group(2)
            return (edge, clock_name)
        return ("posedge", "clk")

    def _extract_property_body(self, sva_code: str) -> str:
        """
        Extract the core property body from SVA code, handling various formats.
        
        This handles:
        - Markdown backticks (inline code formatting)
        - Simple inline expressions: "req |-> ##1 gnt"
        - Property declarations with property/endproperty
        - Assert/assume/cover property wrappers
        - Full SVA with both property definition and assertion
        - Clock specifications
        - Disable conditions
        
        Args:
            sva_code: SVA code
            
        Returns:
            The core property expression
        """
        code = sva_code.strip()
        # Remove markdown backticks that may surround the code (e.g., from LLM output)
        code = code.strip('`')
        code = code.strip()
        # First, check if there's a property ... endproperty block
        # This should take priority over assert property references
        property_match = re.search(
            r'property\s+\w+\s*;?\s*(.*?)\s*endproperty',
            code,
            re.DOTALL | re.IGNORECASE
        )
        if property_match:
            # Extract the property body
            code = property_match.group(1).strip()
            
            # Remove clock specification @(posedge clk) or @(negedge clk)
            code = re.sub(r'@\s*\(\s*(posedge|negedge)\s+\w+\s*\)', '', code)
            
            # Remove disable iff clause
            code = re.sub(r'disable\s+iff\s*\([^)]+\)', '', code)
            
            # Clean up whitespace
            code = ' '.join(code.split())
            
            return code.strip().strip(';').strip()
        
        # Remove trailing else $error(...) clauses
        code = re.sub(r'\s+else\s+\$\w+\s*\([^;]*\)\s*;?\s*$', '', code, flags=re.DOTALL)
        
        # Remove labeled assert: "label: assert property (...)"
        code = re.sub(r'^\s*\w+\s*:\s*', '', code)
        
        # Check for assert/assume/cover property wrapper with inline property
        # Match patterns like: assert property (@(posedge clk) expr);
        wrapper_match = re.search(
            r'(assert|assume|cover)\s+property\s*\(\s*(.*?)\s*\)\s*;?\s*$',
            code,
            re.DOTALL | re.IGNORECASE
        )
        if wrapper_match:
            code = wrapper_match.group(2).strip()
        
        # Remove property name declaration without endproperty
        code = re.sub(r'^\s*property\s+\w+\s*;?\s*', '', code, flags=re.IGNORECASE)
        code = re.sub(r'\s*endproperty\s*$', '', code, flags=re.IGNORECASE)
        
        # Remove clock specification @(posedge clk) or @(negedge clk)
        code = re.sub(r'@\s*\(\s*(posedge|negedge)\s+\w+\s*\)', '', code)
        
        # Remove disable iff clause
        code = re.sub(r'disable\s+iff\s*\([^)]+\)', '', code)
        
        # Clean up whitespace
        code = ' '.join(code.split())
        
        return code.strip().strip(';').strip()

    def _extract_builtin_calls(self, expr: str) -> List[Tuple[str, List[str]]]:
        """
        Extract all built-in function calls from an expression.
        
        Args:
            expr: SVA expression
            
        Returns:
            List of (function_name, arguments) tuples
        """
        calls = []
        
        for func_name in self.BUILTIN_FUNCTIONS:
            # Match function calls with potentially nested parentheses
            pattern = rf'\{func_name}\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)'
            for match in re.finditer(pattern, expr, re.IGNORECASE):
                args_str = match.group(1)
                # Split arguments carefully (handling nested expressions)
                args = self._split_arguments(args_str)
                calls.append((func_name, args))
        
        return calls

    def _split_arguments(self, args_str: str) -> List[str]:
        """
        Split function arguments, respecting nested parentheses.
        
        Args:
            args_str: Arguments string
            
        Returns:
            List of argument strings
        """
        args = []
        current = []
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            args.append(''.join(current).strip())
        
        return [a for a in args if a]

    def _collect_signals_from_expression(self, expr: str) -> Set[str]:
        """
        Collect all signal names from an expression.
        
        Args:
            expr: SVA expression
            
        Returns:
            Set of signal names
        """
        # Remove strings
        clean_expr = re.sub(r'"[^"]*"', '', expr)
        
        # Remove Verilog numeric literals (e.g., 2'b01, 8'hFF, 32'd100)
        clean_expr = re.sub(r"\d+'[bBhHdDoO][0-9a-fA-FxXzZ_]+", '', clean_expr)
        
        # Remove built-in function names (but keep their arguments)
        # Replace $func_name with just the opening paren to preserve arguments
        for func in self.BUILTIN_FUNCTIONS:
            # Match the function call but don't consume the content
            clean_expr = re.sub(rf'\{func}\s*\(', '(', clean_expr, flags=re.IGNORECASE)
        
        # Keywords to exclude (SVA/Verilog keywords only)
        # Note: Do NOT exclude signal names like 'reset', 'rst', 'clock', 'stable'
        # as these could be user signals. Only exclude 'clk' and 'rst_n' which are
        # the module's predefined ports.
        # Built-in function names like $stable are already handled by the regex
        # substitution above, so we don't need to exclude their base names.
        keywords = {
            'property', 'endproperty', 'sequence', 'endsequence',
            'assert', 'assume', 'cover', 'disable', 'iff',
            'posedge', 'negedge', 'or', 'and', 'not', 'if', 'else',
            'throughout', 'within', 'intersect', 'first_match',
            'clk', 'rst_n',  # Module's predefined ports only
        }
        
        # Find all identifiers
        identifier_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(identifier_pattern, clean_expr)
        
        signals = set()
        for match in matches:
            if match.lower() not in keywords and not match.startswith('$'):
                signals.add(match)
        
        return signals

    def _collect_all_signals(self, *expressions: str) -> Set[str]:
        """
        Collect all signals from multiple expressions.
        
        Args:
            *expressions: SVA expressions
            
        Returns:
            Set of signal names
        """
        all_signals = set()
        for expr in expressions:
            all_signals.update(self._collect_signals_from_expression(expr))
        return all_signals

    def _detect_unsupported_functions(self, *expressions: str) -> Set[str]:
        """
        Detect SVA functions that are not supported by EBMC.
        
        Args:
            *expressions: SVA expressions to check
            
        Returns:
            Set of unsupported function names found in the expressions
        """
        unsupported = set()
        unsupported_funcs = self.BUILTIN_FUNCTIONS - self.EBMC_SUPPORTED_FUNCTIONS
        for expr in expressions:
            for func in unsupported_funcs:
                if re.search(rf'\{func}\s*\(', expr, re.IGNORECASE):
                    unsupported.add(func)
        return unsupported

    def _detect_unsupported_constructs(self, *expressions: str) -> Set[str]:
        """
        Detect SVA constructs that are not supported by EBMC.
        
        Args:
            *expressions: SVA expressions to check
            
        Returns:
            Set of unsupported construct names found in the expressions
        """
        unsupported = set()
        for expr in expressions:
            for construct_name, pattern in self.UNSUPPORTED_CONSTRUCTS.items():
                if re.search(pattern, expr, re.IGNORECASE | re.DOTALL):
                    unsupported.add(construct_name)
        return unsupported

    def _analyze_past_usage(self, expr: str) -> Dict[str, int]:
        """
        Analyze $past function usage and determine required delay depths.
        
        Args:
            expr: SVA expression
            
        Returns:
            Dictionary mapping signal names to their max $past depth
        """
        past_usage = {}
        
        # Match $past(signal) or $past(signal, depth)
        pattern = r'\$past\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:,\s*(\d+))?\s*\)'
        
        for match in re.finditer(pattern, expr, re.IGNORECASE):
            signal = match.group(1)
            depth = int(match.group(2)) if match.group(2) else 1
            
            if signal not in past_usage:
                past_usage[signal] = depth
            else:
                past_usage[signal] = max(past_usage[signal], depth)
        
        return past_usage

    def _generate_past_signal_infrastructure(
        self,
        expr1: str,
        expr2: str,
        clock_name: str = "clk"
    ) -> Tuple[str, str, str, str]:
        """
        Generate Verilog code for $past signal handling.
        
        EBMC may or may not support $past directly. This generates
        explicit shift registers as a fallback.
        
        Args:
            expr1: First expression
            expr2: Second expression
            clock_name: Name of the clock signal to use
            
        Returns:
            Tuple of (declarations, logic, modified_expr1, modified_expr2)
        """
        # Analyze $past usage in both expressions
        past_usage1 = self._analyze_past_usage(expr1)
        past_usage2 = self._analyze_past_usage(expr2)
        
        # Merge usage
        all_past = {}
        for signal, depth in {**past_usage1, **past_usage2}.items():
            if signal not in all_past:
                all_past[signal] = depth
            else:
                all_past[signal] = max(all_past[signal], depth)
        
        if not all_past:
            return "", "", expr1, expr2
        
        declarations = []
        logic = []
        
        for signal, max_depth in all_past.items():
            # Create shift register for each signal that uses $past
            for d in range(1, max_depth + 1):
                declarations.append(f"    reg {signal}_d{d};")
            
            # Generate shift register logic
            logic.append(f"    always @(posedge {clock_name}) begin")
            logic.append(f"        if (!rst_n) begin")
            for d in range(1, max_depth + 1):
                logic.append(f"            {signal}_d{d} <= 1'b0;")
            logic.append(f"        end else begin")
            logic.append(f"            {signal}_d1 <= {signal};")
            for d in range(2, max_depth + 1):
                logic.append(f"            {signal}_d{d} <= {signal}_d{d-1};")
            logic.append(f"        end")
            logic.append(f"    end")
            logic.append("")
        
        # We'll keep $past in the expressions - EBMC should handle it
        # The shift registers above are just for reference/fallback
        modified_expr1 = expr1
        modified_expr2 = expr2
        
        return (
            "\n".join(declarations) if declarations else "",
            "\n".join(logic) if logic else "",
            modified_expr1,
            modified_expr2
        )

    def _generate_signal_declarations(self, signals: Set[str]) -> str:
        """
        Generate Verilog signal declarations.
        
        Args:
            signals: Set of signal names
            
        Returns:
            Signal declarations string
        """
        if not signals:
            return "    // No additional signals"
        
        declarations = []
        for sig in sorted(signals):
            declarations.append(f"    input wire {sig}")
        
        return ",\n".join(declarations)

    def _run_ebmc(
        self,
        module_content: str,
        module_name: str,
        work_dir: str,
    ) -> Tuple[Optional[bool], str]:
        """
        Run EBMC verification.
        
        Args:
            module_content: Verilog module content
            module_name: Name of the module
            work_dir: Working directory
            
        Returns:
            Tuple of (success, log_output)
        """
        module_file = os.path.join(work_dir, f"{module_name}.sv")
        
        # Write module file
        with open(module_file, 'w') as f:
            f.write(module_content)
        
        if self.verbose:
            print("=" * 60)
            print("Generated Module:")
            print("=" * 60)
            print(module_content)
            print("=" * 60)
        
        # Run EBMC with bounded model checking
        cmd = [
            self.ebmc_path,
            module_file,
            "--top", module_name,
            "--bound", str(self.depth),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            log = result.stdout + result.stderr
            if self.verbose:
                print("EBMC Output:")
                print(log)
            # EBMC exit codes:
            # 0 = all properties proved
            # 10 = property violated (counterexample found)
            # Other = error
            success = result.returncode == 0
            return success, log
        except subprocess.TimeoutExpired:
            return None, f"EBMC verification timed out after {self.timeout} seconds"

    def _categorize_error(self, log: str) -> Tuple[bool, bool, str]:
        """
        Categorize the type of error from EBMC log output.
        
        Args:
            log: EBMC log output
            
        Returns:
            Tuple of (is_error, is_syntax_error, error_message)
            - is_error: True if any error occurred (not a verification failure)
            - is_syntax_error: True if the error is due to invalid SVA syntax
            - error_message: Descriptive error message
        """
        log_lower = log.lower()
        # Internal tool errors (EBMC crash, etc.)
        internal_error_indicators = [
            "invariant violation",
            "invariant check failed",
            "segmentation fault",
            "aborted",
            "internal error",
        ]
        for indicator in internal_error_indicators:
            if indicator in log_lower:
                return (True, False, f"EBMC internal error: {indicator}")
        # Check for verification failure indicators (property refuted)
        # "refuted" and "violated" in the context of property verification
        # (but not "invariant violation" which is a tool error)
        if "refuted" in log_lower or ("violated" in log_lower and "invariant" not in log_lower):
            return (False, False, "")
        # SVA/Verilog syntax errors - these indicate problems with the input SVA
        syntax_error_indicators = [
            ("syntax error", "Syntax error in SVA"),
            ("parse error", "Parse error in SVA"),
            ("parsing error", "Parsing error in SVA"),
            ("expected", "Syntax error: unexpected token"),
            ("expecting", "Syntax error: expecting element"),  # EBMC uses "expecting"
            ("unexpected", "Syntax error: unexpected token"),
            ("unterminated", "Syntax error: unterminated construct"),
            ("missing", "Syntax error: missing element"),
            ("invalid", "Syntax error: invalid construct"),
            ("unrecognized", "Syntax error: unrecognized element"),
            ("unknown identifier", "Unknown identifier in SVA"),
            ("unknown preprocessor directive", "Syntax error: unknown preprocessor directive"),
            ("undeclared", "Undeclared identifier in SVA"),
            ("after backtick", "Syntax error: invalid backtick usage"),
        ]
        for indicator, msg in syntax_error_indicators:
            if indicator in log_lower:
                # Extract the actual error line from the log if possible
                error_detail = self._extract_error_detail(log)
                full_msg = f"{msg}: {error_detail}" if error_detail else msg
                return (True, True, full_msg)
        # General errors that could be syntax or tool issues
        general_error_indicators = [
            "error:",
            "conversion error",
            "type error",
            "cannot open",
            "file not found",
        ]
        for indicator in general_error_indicators:
            if indicator in log_lower:
                error_detail = self._extract_error_detail(log)
                # Check if the error message suggests syntax issues
                if any(hint in log_lower for hint in ["line", "column", "token", "character"]):
                    return (True, True, f"SVA syntax error: {error_detail}" if error_detail else "SVA syntax error")
                return (True, False, f"EBMC error: {error_detail}" if error_detail else "EBMC error")
        return (False, False, "")

    def _extract_error_detail(self, log: str) -> str:
        """
        Extract detailed error message from EBMC log.
        
        Args:
            log: EBMC log output
            
        Returns:
            Extracted error detail or empty string
        """
        lines = log.split('\n')
        for line in lines:
            line_lower = line.lower()
            # Look for lines containing error information
            if 'error' in line_lower or 'expected' in line_lower or 'unexpected' in line_lower:
                # Clean up the line
                line = line.strip()
                if line and len(line) < 200:  # Avoid overly long lines
                    return line
        return ""

    def _is_tool_error(self, log: str) -> bool:
        """
        Check if the log indicates a tool error (syntax error, etc.)
        rather than a verification failure.
        
        Args:
            log: EBMC log output
            
        Returns:
            True if it's a tool error, False if it's a verification failure
        """
        is_error, _, _ = self._categorize_error(log)
        return is_error

    def _is_syntax_error(self, log: str) -> Tuple[bool, str]:
        """
        Check if the log indicates a syntax error in the input SVA.
        
        Args:
            log: EBMC log output
            
        Returns:
            Tuple of (is_syntax_error, error_message)
        """
        is_error, is_syntax, msg = self._categorize_error(log)
        return (is_error and is_syntax, msg)

    def _extract_counterexample(self, log: str) -> Optional[str]:
        """
        Extract counterexample from EBMC log.
        
        Args:
            log: EBMC log output
            
        Returns:
            Counterexample trace or None
        """
        lines = log.split('\n')
        cex_lines = []
        in_cex = False
        
        for line in lines:
            if 'counterexample' in line.lower() or 'refuted' in line.lower():
                in_cex = True
            if 'state' in line.lower() or 'cycle' in line.lower():
                in_cex = True
            if in_cex:
                cex_lines.append(line)
                if len(cex_lines) > 50:
                    break
        
        return '\n'.join(cex_lines) if cex_lines else None

    def check_implication(
        self,
        antecedent: str,
        consequent: str,
        verbose: bool = False
    ) -> CheckResult:
        """
        Check if antecedent implies consequent.
        
        This method handles various SVA formats:
        - Simple inline expressions: "req |-> ##1 gnt"
        - Full property declarations with property/endproperty
        - Assert/assume/cover property wrappers
        - Properties with $past, $rose, $fell, etc.
        
        Args:
            antecedent: Antecedent SVA code (can be full property or expression)
            consequent: Consequent SVA code (can be full property or expression)
            
        Returns:
            CheckResult with the verification result
        """
        # Check if EBMC is available
        if not self.ebmc_available:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=f"EBMC not found at '{self.ebmc_path}'. Please install EBMC or provide the correct path.",
                module=None,
            )
        
        # Extract clock specifications from both SVAs
        ant_clock_edge, ant_clock_name = self._extract_clock_spec(antecedent)
        cons_clock_edge, cons_clock_name = self._extract_clock_spec(consequent)
        
        # Check for clock mismatch
        if ant_clock_name != cons_clock_name or ant_clock_edge != cons_clock_edge:
            # Use the antecedent's clock as the primary clock
            if self.verbose:
                print(f"Warning: Clock mismatch - antecedent uses @({ant_clock_edge} {ant_clock_name}), "
                      f"consequent uses @({cons_clock_edge} {cons_clock_name}). Using antecedent's clock.")
        
        clock_edge = ant_clock_edge
        clock_name = ant_clock_name
        
        # Extract property bodies from potentially complex SVA code
        ant_expr = self._extract_property_body(antecedent)
        cons_expr = self._extract_property_body(consequent)
        
        if self.verbose:
            print(f"Clock: @({clock_edge} {clock_name})")
            print(f"Antecedent expression: {ant_expr}")
            print(f"Consequent expression: {cons_expr}")
        
        # Check for unsupported SVA constructs early
        unsupported_constructs = self._detect_unsupported_constructs(ant_expr, cons_expr)
        if unsupported_constructs:
            construct_descriptions = {
                "if_else_property": "conditional property (if...else with implications)",
            }
            descriptions = [
                construct_descriptions.get(c, c) for c in sorted(unsupported_constructs)
            ]
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=f"EBMC Error: Unsupported SVA constructs detected: {', '.join(descriptions)}. "
                        f"EBMC does not support these constructs.",
                module=None,
            )
        
        # Collect all signals, excluding the clock signal (it's declared as module port)
        signals = self._collect_all_signals(ant_expr, cons_expr)
        signals.discard(clock_name)  # Don't redeclare the clock signal
        signal_decls = self._generate_signal_declarations(signals)
        
        # Handle $past and other temporal functions
        past_decls, past_logic, ant_expr, cons_expr = \
            self._generate_past_signal_infrastructure(ant_expr, cons_expr, clock_name)
        
        # Generate module with the correct clock
        module_content = self.MODULE_TEMPLATE.format(
            clock_name=clock_name,
            clock_edge=clock_edge,
            signal_declarations=signal_decls,
            past_signal_declarations=past_decls,
            past_signal_logic=past_logic,
            antecedent=ant_expr,
            consequent=cons_expr,
        )
        if verbose:
            print(module_content)
        # Create working directory
        if self.work_dir:
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = tempfile.mkdtemp(prefix="sva_check_")
        
        try:
            success, log = self._run_ebmc(module_content, "sva_checker", work_dir)
            # Handle timeout case (success is None)
            if success is None:
                return CheckResult(
                    result=ImplicationResult.TIMEOUT,
                    message=log,
                    module=module_content,
                )
            if success:
                return CheckResult(
                    result=ImplicationResult.IMPLIES,
                    message="Antecedent implies consequent (proof passed)",
                    log=log,
                    module=module_content,
                )
            else:
                # Categorize the error type
                is_error, is_syntax_error, error_msg = self._categorize_error(log)
                if is_error:
                    # Check for unsupported functions in the expressions
                    unsupported = self._detect_unsupported_functions(ant_expr, cons_expr)
                    if unsupported:
                        msg = f"EBMC Error: Unsupported SVA functions detected: {', '.join(sorted(unsupported))}. EBMC does not support these functions."
                        result_type = ImplicationResult.ERROR
                    elif is_syntax_error:
                        # This is a syntax error in the input SVA
                        msg = f"SVA Syntax Error: {error_msg}" if error_msg else "SVA Syntax Error: Invalid SVA syntax in input"
                        result_type = ImplicationResult.SYNTAX_ERROR
                    elif "invariant violation" in log.lower() or "invariant check failed" in log.lower():
                        msg = "EBMC Error: Internal tool crash. The SVA may contain unsupported constructs."
                        result_type = ImplicationResult.ERROR
                    else:
                        # Could still be a syntax error we didn't categorize
                        msg = error_msg if error_msg else "EBMC Error: Check the generated module for syntax issues"
                        result_type = ImplicationResult.ERROR
                    return CheckResult(
                        result=result_type,
                        message=msg,
                        log=log,
                        module=module_content,
                    )
                counterexample = self._extract_counterexample(log)
                return CheckResult(
                    result=ImplicationResult.NOT_IMPLIES,
                    message="Antecedent does not imply consequent (counterexample found)",
                    counterexample=counterexample,
                    log=log,
                    module=module_content,
                )
        except subprocess.TimeoutExpired:
            return CheckResult(
                result=ImplicationResult.TIMEOUT,
                message=f"Verification timed out after {self.timeout} seconds",
                module=module_content if 'module_content' in locals() else None,
            )
        except Exception as e:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=f"Verification error: {str(e)}",
                module=module_content if 'module_content' in locals() else None,
            )
        finally:
            if not self.keep_files and not self.work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    def check_equivalence(self, sva1: str, sva2: str) -> CheckResult:
        """
        Check if two SVAs are equivalent (bidirectional implication).
        
        Args:
            sva1: First SVA code
            sva2: Second SVA code
            
        Returns:
            CheckResult with the verification result
        """
        error_results = (ImplicationResult.ERROR, ImplicationResult.SYNTAX_ERROR, ImplicationResult.TIMEOUT)
        # Check sva1 -> sva2
        result1 = self.check_implication(sva1, sva2)
        if result1.result in error_results:
            return result1
        # Check sva2 -> sva1
        result2 = self.check_implication(sva2, sva1)
        if result2.result in error_results:
            return result2
        
        # Both directions must hold for equivalence
        if (result1.result == ImplicationResult.IMPLIES and 
            result2.result == ImplicationResult.IMPLIES):
            return CheckResult(
                result=ImplicationResult.EQUIVALENT,
                message="SVAs are equivalent (bidirectional implication holds)",
                log=f"Forward: {result1.log}\n\nBackward: {result2.log}",
                module=f"=== Forward Check (sva1 -> sva2) ===\n{result1.module or 'N/A'}\n\n=== Backward Check (sva2 -> sva1) ===\n{result2.module or 'N/A'}",
            )
        else:
            directions = []
            if result1.result == ImplicationResult.IMPLIES:
                directions.append("sva1 -> sva2")
            if result2.result == ImplicationResult.IMPLIES:
                directions.append("sva2 -> sva1")
            
            if directions:
                return CheckResult(
                    result=ImplicationResult.NOT_IMPLIES,
                    message=f"SVAs are not equivalent. Only holds: {', '.join(directions)}",
                    log=f"Forward: {result1.log}\n\nBackward: {result2.log}",
                    module=f"=== Forward Check (sva1 -> sva2) ===\n{result1.module or 'N/A'}\n\n=== Backward Check (sva2 -> sva1) ===\n{result2.module or 'N/A'}",
                )
            else:
                return CheckResult(
                    result=ImplicationResult.NOT_IMPLIES,
                    message="SVAs are not equivalent (neither direction holds)",
                    counterexample=result1.counterexample or result2.counterexample,
                    log=f"Forward: {result1.log}\n\nBackward: {result2.log}",
                    module=f"=== Forward Check (sva1 -> sva2) ===\n{result1.module or 'N/A'}\n\n=== Backward Check (sva2 -> sva1) ===\n{result2.module or 'N/A'}",
                )

    def get_implication_relationship(
        self,
        sva1: str,
        sva2: str,
    ) -> Tuple[bool, bool]:
        """
        Determine the implication relationship between two SVAs.
        
        Args:
            sva1: First SVA code
            sva2: Second SVA code
            
        Returns:
            Tuple of (sva1_implies_sva2, sva2_implies_sva1)
            
        Raises:
            SVASyntaxError: If either SVA has invalid syntax
            RuntimeError: If verification encounters an internal error or timeout
        """
        result1 = self.check_implication(sva1, sva2)
        # Check for errors in the first direction
        if result1.result == ImplicationResult.SYNTAX_ERROR:
            raise SVASyntaxError(result1.message, result1.log)
        elif result1.result == ImplicationResult.ERROR:
            raise RuntimeError(f"Verification error: {result1.message}")
        elif result1.result == ImplicationResult.TIMEOUT:
            raise RuntimeError(f"Verification timeout: {result1.message}")
        result2 = self.check_implication(sva2, sva1)
        # Check for errors in the second direction
        if result2.result == ImplicationResult.SYNTAX_ERROR:
            raise SVASyntaxError(result2.message, result2.log)
        elif result2.result == ImplicationResult.ERROR:
            raise RuntimeError(f"Verification error: {result2.message}")
        elif result2.result == ImplicationResult.TIMEOUT:
            raise RuntimeError(f"Verification timeout: {result2.message}")
        return (
            result1.result == ImplicationResult.IMPLIES,
            result2.result == ImplicationResult.IMPLIES,
        )

    def normalize_sva(self, sva_code: str) -> str:
        """
        Normalize SVA code to a standard form.
        
        This extracts the core property expression and removes
        formatting variations.
        
        Args:
            sva_code: SVA code in any format
            
        Returns:
            Normalized property expression
        """
        return self._extract_property_body(sva_code)