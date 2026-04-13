from __future__ import annotations

from sva_toolkit.formal.backends.ebmc import EbmcBackend
from sva_toolkit.formal.backends.vcformal import VcformalBackend
from sva_toolkit.formal.model import CheckResult, ImplicationResult
from sva_toolkit.formal.parse import parse_property


class FormalService:
    """Unified formal verification service."""

    def __init__(
        self,
        backend: str = "auto",
        timeout: int = 300,
        depth: int = 20,
        verbose: bool = False,
        keep_files: bool = False,
    ):
        backend_name = backend.lower()
        if backend_name not in {"auto", "ebmc", "vcformal"}:
            raise ValueError("backend must be one of: 'auto', 'ebmc', 'vcformal'")

        self.backend_name = backend_name
        self.timeout = timeout
        self.depth = depth
        self.verbose = verbose
        self.keep_files = keep_files

        self._vcformal = VcformalBackend(timeout=timeout)
        self._ebmc = EbmcBackend(depth=depth, timeout=timeout)
        self._configure_backend(self._vcformal)
        self._configure_backend(self._ebmc)

    def check_implication(self, antecedent: str, consequent: str) -> CheckResult:
        try:
            antecedent_property = parse_property(antecedent)
            consequent_property = parse_property(consequent)
        except Exception as exc:
            return CheckResult(
                result=ImplicationResult.SYNTAX_ERROR,
                message=f"Failed to parse property text: {exc}",
            )

        backend = self._select_backend()
        if backend is None:
            return CheckResult(
                result=ImplicationResult.ERROR,
                message="No formal backend is available. Install 'vcf' or 'ebmc' and retry.",
            )

        return backend.check_implication(antecedent_property, consequent_property)

    def check_equivalence(self, sva1: str, sva2: str) -> CheckResult:
        forward = self.check_implication(sva1, sva2)
        if forward.result not in {ImplicationResult.IMPLIES, ImplicationResult.NOT_IMPLIES}:
            return forward

        reverse = self.check_implication(sva2, sva1)
        if reverse.result not in {ImplicationResult.IMPLIES, ImplicationResult.NOT_IMPLIES}:
            return reverse

        if forward.result is ImplicationResult.IMPLIES and reverse.result is ImplicationResult.IMPLIES:
            return CheckResult(
                result=ImplicationResult.EQUIVALENT,
                message="The properties are formally equivalent.",
                log=self._combine_logs(forward, reverse),
            )

        if forward.result is ImplicationResult.IMPLIES:
            message = "The properties are not equivalent: the first implies the second, but not vice versa."
            counterexample = reverse.counterexample
        elif reverse.result is ImplicationResult.IMPLIES:
            message = "The properties are not equivalent: the second implies the first, but not vice versa."
            counterexample = forward.counterexample
        else:
            message = "The properties are not equivalent in either direction."
            counterexample = forward.counterexample or reverse.counterexample

        return CheckResult(
            result=ImplicationResult.NOT_IMPLIES,
            message=message,
            counterexample=counterexample,
            log=self._combine_logs(forward, reverse),
        )

    def get_relationship(self, sva1: str, sva2: str) -> tuple[bool, bool]:
        forward = self.check_implication(sva1, sva2)
        reverse = self.check_implication(sva2, sva1)
        return self._as_boolean(forward), self._as_boolean(reverse)

    def _select_backend(self) -> VcformalBackend | EbmcBackend | None:
        if self.backend_name == "vcformal":
            return self._vcformal
        if self.backend_name == "ebmc":
            return self._ebmc
        if self._vcformal.available:
            return self._vcformal
        if self._ebmc.available:
            return self._ebmc
        return None

    def _configure_backend(self, backend: VcformalBackend | EbmcBackend) -> None:
        backend.keep_files = self.keep_files
        backend.verbose = self.verbose

    def _combine_logs(self, forward: CheckResult, reverse: CheckResult) -> str | None:
        logs: list[str] = []
        if forward.log:
            logs.append("forward:\n" + forward.log)
        if reverse.log:
            logs.append("reverse:\n" + reverse.log)
        return "\n\n".join(logs) or None

    def _as_boolean(self, result: CheckResult) -> bool:
        if result.result is ImplicationResult.IMPLIES:
            return True
        if result.result is ImplicationResult.NOT_IMPLIES:
            return False
        if result.result is ImplicationResult.TIMEOUT:
            raise TimeoutError(result.message)
        raise RuntimeError(result.message)
