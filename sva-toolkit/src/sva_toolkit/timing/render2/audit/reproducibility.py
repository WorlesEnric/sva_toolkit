"""Renderer reproducibility audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import TimingScene
from sva_toolkit.timing.render2.spec import RenderSpec


@dataclass(frozen=True)
class ReproReport:
    renderer_id: str
    attempts: int
    output_kind: str | None
    digests: tuple[str, ...]
    passed: bool
    reason: str | None


def audit_renderer_reproducibility(
    adapter,
    scene: TimingScene,
    spec: RenderSpec,
    *,
    attempts: int = 2,
) -> ReproReport:
    attempts = max(2, int(attempts))
    renderer_id = str(getattr(adapter, "id", spec.renderer_id))
    outputs: list[tuple[str, bytes]] = []

    for _ in range(attempts):
        try:
            result = adapter.render(scene, spec)
        except Exception as exc:
            return ReproReport(
                renderer_id=renderer_id,
                attempts=attempts,
                output_kind=None,
                digests=(),
                passed=False,
                reason=f"render_failed:{type(exc).__name__}",
            )
        output = _canonical_output(result)
        if output is None:
            return ReproReport(
                renderer_id=renderer_id,
                attempts=attempts,
                output_kind=None,
                digests=(),
                passed=False,
                reason="no_render_output",
            )
        outputs.append(output)

    output_kinds = {kind for kind, _payload in outputs}
    digests = tuple(hashlib.sha256(payload).hexdigest() for _kind, payload in outputs)
    passed = len(output_kinds) == 1 and len(set(digests)) == 1
    return ReproReport(
        renderer_id=renderer_id,
        attempts=attempts,
        output_kind=outputs[0][0],
        digests=digests,
        passed=passed,
        reason=None if passed else "non_reproducible_output",
    )


def _canonical_output(result: RenderResult) -> tuple[str, bytes] | None:
    if result.svg_text is not None:
        return "svg", result.svg_text.encode("utf-8")
    if result.png_bytes is not None:
        return "png", result.png_bytes
    if result.ascii_text is not None:
        return "ascii", result.ascii_text.encode("utf-8")
    return None


__all__ = ["ReproReport", "audit_renderer_reproducibility"]
