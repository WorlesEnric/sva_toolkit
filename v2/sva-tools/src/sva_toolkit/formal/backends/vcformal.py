from __future__ import annotations

import re
import shutil
from pathlib import Path

from sva_toolkit.formal.model import CheckResult, FormalProperty, ImplicationResult
from sva_toolkit.runtime.process import make_work_dir, run_tool


class VcformalBackend:
    """VC Formal-based formal verification backend."""

    MODULE_TEMPLATE = """module sva_checker(
    input wire {clock_name},
    input wire {reset_name}{signal_ports}
);
    assume_antecedent: assume property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({antecedent}));
    assert_consequent: assert property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({consequent}));
endmodule
"""

    TCL_TEMPLATE = """set_fml_appmode FPV
read_file -top sva_checker -format sverilog -sva {sv_file}
create_clock {clock_name} -period 1
create_reset {reset_name} -sense {reset_sense}
check_fv -block
report_fv -list > {report_file}
exit 0
"""

    _SYNTAX_PATTERNS = (
        "syntax error",
        "parse error",
        "failed to parse",
        "unexpected token",
        "compile error",
        "elaboration error",
    )
    _INTERNAL_PATTERNS = (
        "internal error",
        "traceback",
        "segmentation fault",
        "assertion failed",
        "fatal",
        "exception",
    )
    _STATUS_PATTERN = re.compile(
        r"^\s*\[(?P<index>\d+)\]\s*(?P<entry>.*?\b(?P<status>proven|falsified|vacuous|inconclusive)\b.*)$",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self, tool_path: str | None = None, timeout: int = 300):
        self.tool_path = tool_path or shutil.which("vcf")
        self.timeout = timeout
        self.keep_files = False
        self.verbose = False

    @property
    def available(self) -> bool:
        return self.tool_path is not None and shutil.which(self.tool_path) is not None

    def check_implication(self, antecedent: FormalProperty, consequent: FormalProperty) -> CheckResult:
        compatibility_error = self._validate_properties(antecedent, consequent)
        if compatibility_error is not None:
            return compatibility_error
        if not self.available:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message="VC Formal executable is not available.",
            )

        module_text = self._build_module(antecedent, consequent)
        work_dir = Path(make_work_dir(prefix="sva_vcf_"))
        sv_file = work_dir / "sva_checker.sv"
        report_file = work_dir / "report.txt"
        tcl_file = work_dir / "run.tcl"
        sv_file.write_text(module_text, encoding="utf-8")
        tcl_file.write_text(
            self._build_tcl(
                sv_file=sv_file,
                report_file=report_file,
                clock_name=antecedent.clock_name,
                reset_name=antecedent.reset_name,
                reset_sense=antecedent.reset_sense,
            ),
            encoding="utf-8",
        )

        try:
            try:
                result = run_tool(
                    [self.tool_path or "vcf", "-f", str(tcl_file)],
                    cwd=work_dir,
                    timeout=self.timeout,
                )
            except RuntimeError as exc:
                return CheckResult(
                    result=ImplicationResult.ERROR,
                    message=self._with_artifact_hint(str(exc), work_dir),
                    module=module_text,
                )

            report_text = report_file.read_text(encoding="utf-8") if report_file.exists() else ""
            combined_output = self._combine_output(result.stdout, result.stderr, report_text)

            if result.timed_out:
                return CheckResult(
                    result=ImplicationResult.TIMEOUT,
                    message=self._with_artifact_hint("VC Formal timed out while checking implication.", work_dir),
                    log=combined_output or None,
                    module=module_text,
                )

            status_entry = self._extract_status_entry(report_text or combined_output)
            if status_entry is not None:
                status, status_line = status_entry
                return self._result_from_status(status, status_line, combined_output, module_text, work_dir)

            classified_result, message = self._classify_error(combined_output)
            return CheckResult(
                result=classified_result,
                message=self._with_artifact_hint(message, work_dir),
                log=combined_output or None,
                module=module_text,
            )
        finally:
            if not self.keep_files:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _build_module(self, antecedent: FormalProperty, consequent: FormalProperty) -> str:
        signal_ports = self._render_signal_ports(
            sorted(
                (antecedent.signals | consequent.signals)
                - {antecedent.clock_name, antecedent.reset_name, consequent.clock_name, consequent.reset_name}
            )
        )
        return self.MODULE_TEMPLATE.format(
            clock_name=antecedent.clock_name,
            reset_name=antecedent.reset_name,
            signal_ports=signal_ports,
            clock_edge=antecedent.clock_edge,
            reset_expr=antecedent.reset_expr,
            antecedent=antecedent.body,
            consequent=consequent.body,
        )

    def _build_tcl(
        self,
        *,
        sv_file: Path,
        report_file: Path,
        clock_name: str,
        reset_name: str,
        reset_sense: str,
    ) -> str:
        return self.TCL_TEMPLATE.format(
            sv_file=self._tcl_quote(sv_file),
            report_file=self._tcl_quote(report_file),
            clock_name=clock_name,
            reset_name=reset_name,
            reset_sense=reset_sense,
        )

    def _validate_properties(
        self,
        antecedent: FormalProperty,
        consequent: FormalProperty,
    ) -> CheckResult | None:
        if not antecedent.body:
            return CheckResult(
                result=ImplicationResult.SYNTAX_ERROR,
                message="Antecedent property body is empty after parsing.",
            )
        if not consequent.body:
            return CheckResult(
                result=ImplicationResult.SYNTAX_ERROR,
                message="Consequent property body is empty after parsing.",
            )

        if antecedent.clock_name != consequent.clock_name or antecedent.clock_edge != consequent.clock_edge:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=(
                    "Property clock mismatch: "
                    f"{antecedent.clock_edge} {antecedent.clock_name} vs "
                    f"{consequent.clock_edge} {consequent.clock_name}."
                ),
            )

        if antecedent.reset_name != consequent.reset_name or antecedent.reset_expr != consequent.reset_expr:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message=(
                    "Property reset mismatch: "
                    f"{antecedent.reset_expr} vs {consequent.reset_expr}."
                ),
            )

        return None

    def _render_signal_ports(self, signal_names: list[str]) -> str:
        if not signal_names:
            return ""
        return "".join(f",\n    input wire {signal_name}" for signal_name in signal_names)

    def _combine_output(self, stdout: str, stderr: str, report: str) -> str:
        parts = [part.strip() for part in (stdout, stderr, report) if part.strip()]
        return "\n\n".join(parts)

    def _extract_status_entry(self, text: str) -> tuple[str, str] | None:
        matches = list(self._STATUS_PATTERN.finditer(text))
        if not matches:
            return None

        preferred = [match for match in matches if "assert_consequent" in match.group("entry").lower()]
        chosen = preferred[0] if preferred else matches[0]
        return chosen.group("status").lower(), chosen.group("entry").strip()

    def _result_from_status(
        self,
        status: str,
        status_line: str,
        combined_output: str,
        module_text: str,
        work_dir: Path,
    ) -> CheckResult:
        if status == "proven":
            return CheckResult(
                result=ImplicationResult.IMPLIES,
                message=self._with_artifact_hint("VC Formal proved the implication.", work_dir),
                log=combined_output or status_line,
                module=module_text,
            )
        if status == "vacuous":
            return CheckResult(
                result=ImplicationResult.IMPLIES,
                message=self._with_artifact_hint("VC Formal proved the implication vacuously.", work_dir),
                log=combined_output or status_line,
                module=module_text,
            )
        if status == "falsified":
            return CheckResult(
                result=ImplicationResult.NOT_IMPLIES,
                message=self._with_artifact_hint(
                    "VC Formal found a counterexample to the implication.",
                    work_dir,
                ),
                counterexample=self._extract_counterexample(combined_output, status_line),
                log=combined_output or status_line,
                module=module_text,
            )
        return CheckResult(
            result=ImplicationResult.ERROR,
            message=self._with_artifact_hint("VC Formal returned an inconclusive result.", work_dir),
            log=combined_output or status_line,
            module=module_text,
        )

    def _extract_counterexample(self, output: str, fallback: str) -> str | None:
        if output:
            match = re.search(r"(?is)(counterexample.*)", output)
            if match:
                return match.group(1).strip()

            lines = output.splitlines()
            interesting = [
                line.strip()
                for line in lines
                if any(token in line.lower() for token in ("falsified", "cex", "trace", "step", "state"))
            ]
            if interesting:
                return "\n".join(interesting[:40])
        return fallback.strip() or None

    def _classify_error(self, output: str) -> tuple[ImplicationResult, str]:
        lower_output = output.lower()
        if any(pattern in lower_output for pattern in self._SYNTAX_PATTERNS):
            return ImplicationResult.SYNTAX_ERROR, "VC Formal reported a SystemVerilog syntax error."
        if any(pattern in lower_output for pattern in self._INTERNAL_PATTERNS):
            return ImplicationResult.ERROR, "VC Formal failed with an internal error."
        return ImplicationResult.ERROR, "VC Formal failed to complete formal verification."

    def _tcl_quote(self, path: Path) -> str:
        return "{" + path.resolve().as_posix() + "}"

    def _with_artifact_hint(self, message: str, work_dir: Path) -> str:
        if self.keep_files or self.verbose:
            return f"{message} Artifacts: {work_dir}"
        return message
