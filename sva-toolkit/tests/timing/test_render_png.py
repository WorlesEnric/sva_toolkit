from __future__ import annotations

import builtins

import pytest

from sva_toolkit.timing.core.conditions import parse_dsl_condition
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ClockingSpec,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.render.png import render_diagram_png


def test_render_diagram_png_raises_runtime_error_when_cairosvg_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    original_import = builtins.__import__

    def _missing_cairosvg(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cairosvg":
            raise ImportError("No module named 'cairosvg'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _missing_cairosvg)

    with pytest.raises(RuntimeError) as exc_info:
        render_diagram_png(_concrete_document(), tmp_path / "diagram.png")

    message = str(exc_info.value).lower()
    assert "cairosvg" in message
    assert "svg" in message


def test_render_diagram_png_writes_non_empty_png_output(tmp_path) -> None:
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>')
    except (ImportError, OSError) as exc:
        pytest.skip(f"cairosvg native dependencies are unavailable: {exc}")

    output_path = tmp_path / "diagram.png"
    render_diagram_png(_concrete_document(), output_path)

    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _concrete_document() -> ScenarioDocument:
    return ScenarioDocument(
        name="png_wave",
        clocking=ClockingSpec(edge="posedge", signal="clk"),
        signals=(
            SignalDecl(name="req", kind=SignalKind.BIT, samples=("0", "1", "1", "0")),
            SignalDecl(name="gnt", kind=SignalKind.BIT, samples=("0", "0", "1", "0")),
        ),
        anchors=(
            Anchor(name="start", condition=parse_dsl_condition("rise(req)"), role=AnchorRole.TRIGGER),
            Anchor(name="grant", condition=parse_dsl_condition("high(gnt)"), role=AnchorRole.RESPONSE),
        ),
        windows=(
            TimeWindow(
                name="grant_window",
                start_anchor="start",
                end_anchor="grant",
                bound=TimeBound(kind=WindowBoundKind.RANGE, min_delay="1", max_delay="2"),
            ),
        ),
        ticks=4,
    )
