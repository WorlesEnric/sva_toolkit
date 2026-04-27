"""Target visibility audit for render2 results."""

from __future__ import annotations

from dataclasses import dataclass

from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import TimingScene


@dataclass(frozen=True)
class TargetVisibilityReport:
    required_signals: frozenset[str]
    rendered_signals: frozenset[str]
    missing_signals: frozenset[str]
    passed: bool
    reasons: tuple[str, ...]


def audit_target_visibility(scene: TimingScene, render_result: RenderResult) -> TargetVisibilityReport:
    """Ensure every target signal has a visible rendered label."""

    required = _required_signal_names(scene)
    rendered_text = frozenset(text.text for text in render_result.visibility.rendered_text)
    rendered = frozenset(signal for signal in required if _signal_is_rendered(signal, rendered_text))
    missing = required - rendered
    return TargetVisibilityReport(
        required_signals=required,
        rendered_signals=rendered,
        missing_signals=missing,
        passed=not missing,
        reasons=() if not missing else ("target_not_visible",),
    )


def _required_signal_names(scene: TimingScene) -> frozenset[str]:
    if scene.visible_target is not None:
        names = {scene.visible_target.clocking.signal}
        names.update(signal.name for signal in scene.visible_target.signals)
        return frozenset(names)
    return frozenset(lane.name for lane in scene.lanes)


def _signal_is_rendered(signal: str, rendered_text: frozenset[str]) -> bool:
    return signal in rendered_text or any(text.startswith(f"{signal}[") for text in rendered_text)
