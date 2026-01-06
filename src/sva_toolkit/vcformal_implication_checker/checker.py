"""
VCFormal-based SVA Implication Checker.

This checker uses Synopsys VC Formal (vcf) as the primary engine for checking
whether SVA1 implies SVA2 (SVA1 -> SVA2), with optional EBMC cross-validation.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sva_toolkit.implication_checker.checker import CheckResult, ImplicationResult, SVAImplicationChecker


@dataclass(frozen=True)
class CrossValidationEntry:
    """A single cross-validation record between VCFormal and EBMC."""

    entry_id: str
    antecedent: str
    consequent: str
    vcformal_result: str
    ebmc_result: Optional[str]
    aligned: bool
    vcformal_message: str
    ebmc_message: Optional[str]


@dataclass(frozen=True)
class CrossValidationSummary:
    """Aggregated cross-validation statistics."""

    total: int
    aligned: int
    mismatched: int
    vcformal_counts: Dict[str, int]
    ebmc_counts: Dict[str, int]
    ebmc_skipped: int


class VCFormalImplicationChecker:
    """
    Checker for verifying implication relationships between SVA pairs using VC Formal.

    Approach:
    - Build a small SV module that:
      - assumes antecedent property
      - asserts consequent property
    - Run VC Formal proof (check_fv -prove -all)
    - Parse the report to decide implies / not_implies / error / timeout
    """

    MODULE_TEMPLATE: str = """
module sva_checker (
    input wire {clock_name},
    input wire rst_n{signal_declarations}
);
    // 1) Constrain with antecedent
    assume_antecedent: assume property (@({clock_edge} {clock_name}) disable iff (!rst_n)
        {antecedent}
    );
    // 2) Prove consequent under constraint
    assert_consequent: assert property (@({clock_edge} {clock_name}) disable iff (!rst_n)
        {consequent}
    );
endmodule
"""

    TCL_TEMPLATE: str = r"""
set_app_var fml_mode_on true
set_app_var compile_enable_sva true

# Read design (single generated module)
read_file -top {top} -format sverilog -sva {{{sv_file}}}

# Basic clock/reset constraints (properties also use explicit event controls)
catch {{ create_clock -name {clock_name} -period 10 [get_ports {clock_name}] }}
catch {{ create_reset [get_ports rst_n] -sense low }}

# Prove all properties
check_fv -prove -all
report_fv -summary

exit
"""

    def __init__(
        self,
        vcf_path: Optional[str] = None,
        work_dir: Optional[str] = None,
        keep_files: bool = False,
        verbose: bool = False,
        timeout: int = 300,
        require_vcformal: bool = False,
    ) -> None:
        """
        Initialize the VCFormal implication checker.

        Args:
            vcf_path: Path to `vcf` binary (default: find in PATH)
            work_dir: Working directory (default: temp dir per run)
            keep_files: Keep generated files
            verbose: Verbose logging
            timeout: Timeout in seconds for each VCFormal run
            require_vcformal: If True, raise error if VCFormal not found
        """
        self.vcf_path: str = vcf_path or (shutil.which("vcf") or "vcf")
        self.work_dir: Optional[str] = work_dir
        self.keep_files: bool = keep_files
        self.verbose: bool = verbose
        self.timeout: int = timeout
        self.vcformal_available: bool = False
        if require_vcformal:
            self._verify_tools()
            self.vcformal_available = True
        else:
            self.vcformal_available = self._check_vcformal_available()

    def _check_vcformal_available(self) -> bool:
        """Check if VCFormal is available without raising."""
        try:
            subprocess.run([self.vcf_path, "-help"], capture_output=True, text=True, timeout=10)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _verify_tools(self) -> None:
        """Verify VCFormal is accessible."""
        try:
            subprocess.run([self.vcf_path, "-help"], capture_output=True, text=True, timeout=10)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"VC Formal not found at '{self.vcf_path}'. Please install VC Formal or provide the correct path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("VC Formal verification timed out during tool check.") from exc

    def _extract_clock_spec(self, sva_code: str) -> Tuple[str, str]:
        """
        Extract the clock specification from SVA code.

        Returns:
            Tuple of (clock_edge, clock_name). Defaults to ("posedge", "clk") if not found.
        """
        code: str = sva_code.strip().strip("`").strip()
        clock_match = re.search(
            r"@\s*\(\s*(posedge|negedge)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\)",
            code,
            re.IGNORECASE,
        )
        if clock_match:
            return clock_match.group(1).lower(), clock_match.group(2)
        return "posedge", "clk"

    def _extract_property_body(self, sva_code: str) -> str:
        """
        Extract the core property body from SVA code, handling various formats.
        """
        code: str = sva_code.strip().strip("`").strip()
        property_match = re.search(
            r"property\s+\w+\s*;?\s*(.*?)\s*endproperty",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if property_match:
            code = property_match.group(1).strip()
            code = re.sub(r"@\s*\(\s*(posedge|negedge)\s+\w+\s*\)", "", code)
            code = re.sub(r"disable\s+iff\s*\([^)]+\)", "", code)
            code = " ".join(code.split())
            return code.strip().strip(";").strip()
        code = re.sub(r"\s+else\s+\$\w+\s*\([^;]*\)\s*;?\s*$", "", code, flags=re.DOTALL)
        code = re.sub(r"^\s*\w+\s*:\s*", "", code)
        wrapper_match = re.search(
            r"(assert|assume|cover)\s+property\s*\(\s*(.*?)\s*\)\s*;?\s*$",
            code,
            re.DOTALL | re.IGNORECASE,
        )
        if wrapper_match:
            code = wrapper_match.group(2).strip()
        code = re.sub(r"^\s*property\s+\w+\s*;?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*endproperty\s*$", "", code, flags=re.IGNORECASE)
        code = re.sub(r"@\s*\(\s*(posedge|negedge)\s+\w+\s*\)", "", code)
        code = re.sub(r"disable\s+iff\s*\([^)]+\)", "", code)
        code = " ".join(code.split())
        return code.strip().strip(";").strip()

    def _collect_signals_from_expression(self, expr: str) -> Set[str]:
        """
        Collect signal identifiers from an SVA expression.
        """
        clean_expr: str = re.sub(r'"[^"]*"', "", expr)
        clean_expr = re.sub(r"\d+'[bBhHdDoO][0-9a-fA-FxXzZ_]+", "", clean_expr)
        clean_expr = re.sub(r"\$[a-zA-Z_][a-zA-Z0-9_]*\s*\(", "(", clean_expr)
        keywords: Set[str] = {
            "property",
            "endproperty",
            "sequence",
            "endsequence",
            "assert",
            "assume",
            "cover",
            "disable",
            "iff",
            "posedge",
            "negedge",
            "or",
            "and",
            "not",
            "if",
            "else",
            "throughout",
            "within",
            "intersect",
            "first_match",
            "rst_n",
        }
        matches: List[str] = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", clean_expr)
        signals: Set[str] = set()
        for m in matches:
            if m.lower() not in keywords and not m.startswith("$"):
                signals.add(m)
        return signals

    def _generate_signal_declarations(self, signals: Set[str]) -> str:
        """Generate SV input declarations for all signals."""
        if not signals:
            return ""
        decls: str = ",\n" + ",\n".join([f"    input wire {sig}" for sig in sorted(signals)])
        return decls

    def _build_module(
        self,
        antecedent: str,
        consequent: str,
    ) -> Tuple[str, str, str]:
        """
        Build the generated SV module and return (module_content, clock_edge, clock_name).
        """
        ant_edge, ant_clk = self._extract_clock_spec(antecedent)
        cons_edge, cons_clk = self._extract_clock_spec(consequent)
        clock_edge: str = ant_edge
        clock_name: str = ant_clk
        if (ant_edge, ant_clk) != (cons_edge, cons_clk) and self.verbose:
            print(
                f"Warning: Clock mismatch - antecedent uses @({ant_edge} {ant_clk}), "
                f"consequent uses @({cons_edge} {cons_clk}). Using antecedent's clock."
            )
        ant_expr: str = self._extract_property_body(antecedent)
        cons_expr: str = self._extract_property_body(consequent)
        signals: Set[str] = self._collect_signals_from_expression(ant_expr) | self._collect_signals_from_expression(
            cons_expr
        )
        signals.discard(clock_name)
        signal_decls: str = self._generate_signal_declarations(signals)
        module_content: str = self.MODULE_TEMPLATE.format(
            clock_name=clock_name,
            clock_edge=clock_edge,
            signal_declarations=signal_decls,
            antecedent=ant_expr,
            consequent=cons_expr,
        )
        return module_content, clock_edge, clock_name

    def _parse_vcformal_summary(self, log: str) -> Tuple[ImplicationResult, str]:
        """
        Parse VCFormal output and derive implication result.

        Heuristics:
        - If assert_consequent is Proven -> IMPLIES
        - If assert_consequent is Falsified/Failed -> NOT_IMPLIES
        - If Inconclusive/Vacuous/Unknown -> ERROR (treated as non-decisive)
        """
        log_lower: str = log.lower()
        if "error" in log_lower and "report_fv" not in log_lower:
            return ImplicationResult.ERROR, "VCFormal error (see log)"
        patterns: List[Tuple[str, ImplicationResult, str]] = [
            (r"assert_consequent.*\bproven\b", ImplicationResult.IMPLIES, "Antecedent implies consequent (VCFormal proven)"),
            (r"assert_consequent.*\bfalsif", ImplicationResult.NOT_IMPLIES, "Antecedent does not imply consequent (VCFormal falsified)"),
            (r"assert_consequent.*\bfail", ImplicationResult.NOT_IMPLIES, "Antecedent does not imply consequent (VCFormal failed)"),
            (r"assert_consequent.*\binconclusive\b", ImplicationResult.ERROR, "VCFormal inconclusive on assertion"),
            (r"assert_consequent.*\bvacuous\b", ImplicationResult.ERROR, "VCFormal vacuous proof on assertion"),
        ]
        for pat, res, msg in patterns:
            if re.search(pat, log, re.IGNORECASE):
                return res, msg
        if "assert_consequent" in log_lower and ("proven" not in log_lower and "falsif" not in log_lower):
            return ImplicationResult.ERROR, "VCFormal could not determine assertion status (see log)"
        return ImplicationResult.ERROR, "VCFormal output did not include recognizable property status"

    def _run_vcformal(self, module_content: str, clock_name: str, work_dir: str) -> str:
        """Run VCFormal in batch mode and return combined stdout/stderr."""
        sv_file: str = os.path.join(work_dir, "sva_checker.sv")
        tcl_file: str = os.path.join(work_dir, "run_vcformal.tcl")
        with open(sv_file, "w", encoding="utf-8") as f:
            f.write(module_content)
        tcl_content: str = self.TCL_TEMPLATE.format(top="sva_checker", sv_file=sv_file, clock_name=clock_name)
        with open(tcl_file, "w", encoding="utf-8") as f:
            f.write(tcl_content)
        cmd: List[str] = [self.vcf_path, "-f", tcl_file]
        if self.verbose:
            print("=" * 60)
            print("VCFormal SV module:")
            print(module_content)
            print("=" * 60)
            print("VCFormal Tcl script:")
            print(tcl_content)
            print("=" * 60)
            print("Running:", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return (result.stdout or "") + (result.stderr or "")
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"VCFormal timed out after {self.timeout} seconds")

    def check_implication(self, antecedent: str, consequent: str, verbose: bool = False) -> CheckResult:
        """
        Check if antecedent implies consequent using VCFormal.
        """
        if not self.vcformal_available:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=f"VC Formal not found at '{self.vcf_path}'. Please install VC Formal or provide the correct path.",
                module=None,
            )
        old_verbose: bool = self.verbose
        self.verbose = self.verbose or verbose
        module_content, _, clock_name = self._build_module(antecedent, consequent)
        if self.work_dir:
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = tempfile.mkdtemp(prefix="sva_vcformal_")
        try:
            log: str = self._run_vcformal(module_content, clock_name, work_dir)
            result, msg = self._parse_vcformal_summary(log)
            return CheckResult(result=result, message=msg, log=log, module=module_content)
        except TimeoutError as exc:
            return CheckResult(result=ImplicationResult.TIMEOUT, message=str(exc), module=module_content)
        except Exception as exc:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=f"VCFormal verification error: {str(exc)}",
                module=module_content,
            )
        finally:
            self.verbose = old_verbose
            if not self.keep_files and not self.work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    def check_equivalence(self, sva1: str, sva2: str) -> CheckResult:
        """
        Check if two SVAs are equivalent (bidirectional implication).
        """
        error_results = (ImplicationResult.ERROR, ImplicationResult.SYNTAX_ERROR, ImplicationResult.TIMEOUT)
        r1 = self.check_implication(sva1, sva2)
        if r1.result in error_results:
            return r1
        r2 = self.check_implication(sva2, sva1)
        if r2.result in error_results:
            return r2
        if r1.result == ImplicationResult.IMPLIES and r2.result == ImplicationResult.IMPLIES:
            return CheckResult(
                result=ImplicationResult.EQUIVALENT,
                message="SVAs are equivalent (VCFormal bidirectional implication holds)",
                log=f"Forward: {r1.log}\n\nBackward: {r2.log}",
                module=f"=== Forward Check (sva1 -> sva2) ===\n{r1.module or 'N/A'}\n\n=== Backward Check (sva2 -> sva1) ===\n{r2.module or 'N/A'}",
            )
        return CheckResult(
            result=ImplicationResult.NOT_IMPLIES,
            message="SVAs are not equivalent (VCFormal: bidirectional implication does not hold)",
            log=f"Forward: {r1.log}\n\nBackward: {r2.log}",
            module=f"=== Forward Check (sva1 -> sva2) ===\n{r1.module or 'N/A'}\n\n=== Backward Check (sva2 -> sva1) ===\n{r2.module or 'N/A'}",
        )

    def get_implication_relationship(self, sva1: str, sva2: str) -> Tuple[bool, bool]:
        """
        Determine implication relationship (sva1->sva2, sva2->sva1).

        Raises on ERROR/TIMEOUT to match existing checker contract.
        """
        r1 = self.check_implication(sva1, sva2)
        if r1.result == ImplicationResult.SYNTAX_ERROR:
            raise RuntimeError(r1.message)
        if r1.result == ImplicationResult.ERROR:
            raise RuntimeError(r1.message)
        if r1.result == ImplicationResult.TIMEOUT:
            raise RuntimeError(r1.message)
        r2 = self.check_implication(sva2, sva1)
        if r2.result == ImplicationResult.SYNTAX_ERROR:
            raise RuntimeError(r2.message)
        if r2.result == ImplicationResult.ERROR:
            raise RuntimeError(r2.message)
        if r2.result == ImplicationResult.TIMEOUT:
            raise RuntimeError(r2.message)
        return r1.result == ImplicationResult.IMPLIES, r2.result == ImplicationResult.IMPLIES

    def cross_validate_with_ebmc(
        self,
        pairs: List[Dict[str, Any]],
        ebmc_path: Optional[str] = None,
        ebmc_depth: int = 20,
        ebmc_timeout: int = 300,
        require_ebmc: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[CrossValidationEntry], CrossValidationSummary]:
        """
        Cross-validate VCFormal results against EBMC across a batch of pairs.

        Args:
            pairs: list of dicts containing at least {"id": str, "antecedent": str, "consequent": str}
            ebmc_path: optional path to ebmc
            ebmc_depth: EBMC bound
            ebmc_timeout: EBMC timeout
            require_ebmc: if True, require EBMC to be present

        Returns:
            (entries, summary)
        """
        ebmc_checker = SVAImplicationChecker(
            ebmc_path=ebmc_path,
            depth=ebmc_depth,
            timeout=ebmc_timeout,
            require_ebmc=require_ebmc,
        )
        ebmc_available: bool = ebmc_checker.ebmc_available
        entries: List[CrossValidationEntry] = []
        vc_counts: Dict[str, int] = {}
        ebmc_counts: Dict[str, int] = {}
        aligned: int = 0
        mismatched: int = 0
        ebmc_skipped: int = 0
        for i, p in enumerate(pairs):
            entry_id: str = str(p.get("id", f"entry_{i+1}"))
            antecedent: str = str(p.get("antecedent", p.get("sva1", "")))
            consequent: str = str(p.get("consequent", p.get("sva2", "")))
            vc_res = self.check_implication(antecedent, consequent)
            vc_key: str = vc_res.result.value
            vc_counts[vc_key] = vc_counts.get(vc_key, 0) + 1
            ebmc_key: Optional[str] = None
            ebmc_msg: Optional[str] = None
            if ebmc_available:
                ebmc_res = ebmc_checker.check_implication(antecedent, consequent)
                ebmc_key = ebmc_res.result.value
                ebmc_counts[ebmc_key] = ebmc_counts.get(ebmc_key, 0) + 1
                ebmc_msg = ebmc_res.message
            else:
                ebmc_skipped += 1
            is_aligned: bool = (ebmc_key is not None) and (ebmc_key == vc_key)
            if is_aligned:
                aligned += 1
            else:
                if ebmc_key is not None:
                    mismatched += 1
            entries.append(
                CrossValidationEntry(
                    entry_id=entry_id,
                    antecedent=antecedent,
                    consequent=consequent,
                    vcformal_result=vc_key,
                    ebmc_result=ebmc_key,
                    aligned=is_aligned,
                    vcformal_message=vc_res.message,
                    ebmc_message=ebmc_msg,
                )
            )
            if progress_callback is not None:
                progress_callback(i + 1, len(pairs))
        summary = CrossValidationSummary(
            total=len(entries),
            aligned=aligned,
            mismatched=mismatched,
            vcformal_counts=vc_counts,
            ebmc_counts=ebmc_counts,
            ebmc_skipped=ebmc_skipped,
        )
        return entries, summary

