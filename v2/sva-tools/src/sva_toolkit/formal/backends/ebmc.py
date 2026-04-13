from __future__ import annotations

import re
import shutil
from pathlib import Path

from sva_toolkit.formal.model import CheckResult, FormalProperty, ImplicationResult
from sva_toolkit.runtime.process import make_work_dir, run_tool


class EbmcBackend:
    """EBMC-based formal verification backend."""

    MODULE_TEMPLATE = """module sva_checker(
    input wire {clock_name},
    input wire {reset_name}{signal_ports}
);
    assume property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({antecedent}));
    assert property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({consequent}));
    cover property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({antecedent}));
endmodule
"""

    _SYNTAX_PATTERNS = (
        "syntax error",
        "parse error",
        "parser error",
        "failed to parse",
        "unexpected token",
        "unexpected end",
        "lexical error",
    )
    _INTERNAL_PATTERNS = (
        "internal error",
        "traceback",
        "exception",
        "assertion failed",
        "segmentation fault",
        "stack trace",
        "terminate called",
    )

    def __init__(self, tool_path: str | None = None, depth: int = 20, timeout: int = 300):
        self.tool_path = tool_path or shutil.which("ebmc")
        self.depth = depth
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
                message="EBMC executable is not available.",
            )

        module_text = self._build_module(antecedent, consequent)
        work_dir = Path(make_work_dir(prefix="sva_ebmc_"))
        sv_file = work_dir / "sva_checker.sv"
        sv_file.write_text(module_text, encoding="utf-8")

        try:
            try:
                result = run_tool(
                    [
                        self.tool_path or "ebmc",
                        "--top",
                        "sva_checker",
                        "--bound",
                        str(self.depth),
                        str(sv_file),
                    ],
                    cwd=work_dir,
                    timeout=self.timeout,
                )
            except RuntimeError as exc:
                return CheckResult(
                    result=ImplicationResult.ERROR,
                    message=self._with_artifact_hint(str(exc), work_dir),
                    module=module_text,
                )

            combined_output = self._combine_output(result.stdout, result.stderr)

            if result.timed_out:
                return CheckResult(
                    result=ImplicationResult.TIMEOUT,
                    message=self._with_artifact_hint("EBMC timed out while checking implication.", work_dir),
                    log=combined_output or None,
                    module=module_text,
                )

            if result.returncode == 0:
                return CheckResult(
                    result=ImplicationResult.IMPLIES,
                    message=self._with_artifact_hint("EBMC proved the implication.", work_dir),
                    log=combined_output or None,
                    module=module_text,
                )

            if result.returncode == 10:
                return CheckResult(
                    result=ImplicationResult.NOT_IMPLIES,
                    message=self._with_artifact_hint(
                        "EBMC found a counterexample to the implication.",
                        work_dir,
                    ),
                    counterexample=self._extract_counterexample(combined_output),
                    log=combined_output or None,
                    module=module_text,
                )

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

    def _combine_output(self, stdout: str, stderr: str) -> str:
        parts = [part.strip() for part in (stdout, stderr) if part.strip()]
        return "\n\n".join(parts)

    def _extract_counterexample(self, output: str) -> str | None:
        if not output:
            return None

        match = re.search(r"(?is)(counterexample.*)", output)
        if match:
            return match.group(1).strip()

        lines = output.splitlines()
        interesting = [
            line.strip()
            for line in lines
            if any(token in line.lower() for token in ("fail", "violat", "trace", "state", "step"))
        ]
        if interesting:
            return "\n".join(interesting[:40])
        return output.strip() or None

    def _classify_error(self, output: str) -> tuple[ImplicationResult, str]:
        lower_output = output.lower()
        if any(pattern in lower_output for pattern in self._SYNTAX_PATTERNS):
            return ImplicationResult.SYNTAX_ERROR, "EBMC reported a SystemVerilog syntax error."
        if any(pattern in lower_output for pattern in self._INTERNAL_PATTERNS):
            return ImplicationResult.ERROR, "EBMC failed with an internal error."
        return ImplicationResult.ERROR, "EBMC failed to complete formal verification."

    def _with_artifact_hint(self, message: str, work_dir: Path) -> str:
        if self.keep_files or self.verbose:
            return f"{message} Artifacts: {work_dir}"
        return message
