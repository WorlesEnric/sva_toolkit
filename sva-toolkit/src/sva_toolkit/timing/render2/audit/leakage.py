"""Rendered-text leakage audit for timing renders."""

from __future__ import annotations

from dataclasses import dataclass
import re

from sva_toolkit.timing.render2.decorations import DecorationKind
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.render2.scene import LaneType, TimingScene
from sva_toolkit.timing.visual import VisibilityClass


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CANONICAL_REF_RE = re.compile(r"^[aw]\d+$")
_KEYWORDS = frozenset(
    {
        "after",
        "and",
        "anchor",
        "at",
        "before",
        "between",
        "bit",
        "bus",
        "clock",
        "compressed",
        "cut",
        "diagram",
        "disable",
        "edge",
        "fall",
        "false",
        "from",
        "high",
        "iff",
        "in",
        "lane",
        "lookback",
        "low",
        "negedge",
        "not",
        "omitted",
        "param",
        "posedge",
        "property",
        "rise",
        "show",
        "stable",
        "ticks",
        "until",
        "window",
    }
)


@dataclass(frozen=True)
class LeakageAuditReport:
    leaked_tokens: frozenset[str]
    suspicious_tokens: frozenset[str]
    debug_overlay_tokens: frozenset[str]
    allowed_tokens: frozenset[str]
    rendered_tokens: frozenset[str]
    target_tokens: frozenset[str]
    passed: bool
    reasons: tuple[str, ...]

    def format(self) -> str:
        """Return a compact CLI-friendly summary."""

        status = "passed" if self.passed else "failed"
        return "\n".join(
            (
                f"leakage: {status}",
                f"leaked_tokens: {', '.join(sorted(self.leaked_tokens)) or '-'}",
                f"debug_overlay_tokens: {', '.join(sorted(self.debug_overlay_tokens)) or '-'}",
                f"suspicious_tokens: {', '.join(sorted(self.suspicious_tokens)) or '-'}",
                f"reasons: {', '.join(self.reasons) or '-'}",
            )
        )


def audit_rendered_text(
    scene: TimingScene,
    render_result: RenderResult,
    *,
    target_dsl_text: str,
) -> LeakageAuditReport:
    """Audit rendered SVG text against the target visual DSL token set."""

    target_tokens = tokenize_target_dsl(target_dsl_text)
    rendered_tokens = frozenset(
        token
        for text in render_result.visibility.rendered_text
        for token in _identifier_tokens(text.text, include_keywords=True)
    )
    allowed_tokens = canonical_visual_tokens(scene)
    debug_overlay_tokens = _debug_overlay_tokens(render_result)
    leaked_tokens = frozenset((rendered_tokens & target_tokens) - allowed_tokens)
    suspicious_tokens = _suspicious_tokens(scene, rendered_tokens, allowed_tokens)

    reasons: list[str] = []
    if leaked_tokens:
        reasons.append("render_text_leakage")
    if debug_overlay_tokens:
        reasons.append("debug_overlay_text")

    return LeakageAuditReport(
        leaked_tokens=leaked_tokens,
        suspicious_tokens=suspicious_tokens,
        debug_overlay_tokens=debug_overlay_tokens,
        allowed_tokens=allowed_tokens,
        rendered_tokens=rendered_tokens,
        target_tokens=target_tokens,
        passed=not leaked_tokens and not debug_overlay_tokens,
        reasons=tuple(reasons),
    )


def tokenize_target_dsl(target_dsl_text: str) -> frozenset[str]:
    """Tokenize target DSL text into identifier-like non-keyword tokens."""

    return frozenset(_identifier_tokens(target_dsl_text, include_keywords=False))


def canonical_visual_tokens(scene: TimingScene) -> frozenset[str]:
    """Return identifier tokens that are allowed to appear as visual text."""

    raw_tokens: set[str] = {scene.name, scene.clocking_signal}
    for lane in scene.lanes:
        raw_tokens.add(lane.name)
        if lane.width_bits:
            raw_tokens.add(f"{lane.name}[{lane.width_bits}]")
        if lane.lane_type == LaneType.BUS:
            raw_tokens.update(str(run.value) for run in lane.runs if _printable_bus_value(run.value))

    if scene.visible_target is not None:
        raw_tokens.add(scene.visible_target.name)
        raw_tokens.add(scene.visible_target.clocking.signal)
        for signal in scene.visible_target.signals:
            raw_tokens.add(signal.name)
            raw_tokens.add(signal.display_name)

    for decoration in scene.decorations:
        if decoration.text and decoration.visibility_class == VisibilityClass.VISIBLE_TEXT:
            raw_tokens.add(decoration.text)
        if decoration.kind in {DecorationKind.NUISANCE_TEXT, DecorationKind.CAPTION} and decoration.text:
            raw_tokens.add(decoration.text)

    return frozenset(
        token
        for raw_token in raw_tokens
        for token in _identifier_tokens(raw_token, include_keywords=True)
        if token
    )


def _debug_overlay_tokens(render_result: RenderResult) -> frozenset[str]:
    tokens: set[str] = set()
    tokens.update(
        token
        for text in render_result.visibility.debug_overlay_tokens
        for token in _identifier_tokens(text, include_keywords=True)
    )
    for text in render_result.visibility.rendered_text:
        if text.role == "debug_overlay" or text.visibility_class == VisibilityClass.DEBUG_OVERLAY.value:
            tokens.update(_identifier_tokens(text.text, include_keywords=True))
    return frozenset(tokens)


def _suspicious_tokens(
    scene: TimingScene,
    rendered_tokens: frozenset[str],
    allowed_tokens: frozenset[str],
) -> frozenset[str]:
    suspicious = {token for token in rendered_tokens if _CANONICAL_REF_RE.match(token) and token not in allowed_tokens}
    semantic = scene.semantic_document
    if semantic is not None:
        semantic_names = {anchor.name for anchor in semantic.anchors}
        semantic_names.update(window.name for window in semantic.windows)
        suspicious.update((rendered_tokens & semantic_names) - allowed_tokens)
    return frozenset(suspicious)


def _identifier_tokens(text: str, *, include_keywords: bool) -> tuple[str, ...]:
    tokens = tuple(match.group(0) for match in _IDENT_RE.finditer(text))
    if include_keywords:
        return tokens
    return tuple(token for token in tokens if token.lower() not in _KEYWORDS)


def _printable_bus_value(value: object) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in {"x", "z", "?", "unknown", "highz"}
