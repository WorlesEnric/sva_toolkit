from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    LaneScene,
    LaneType,
    RenderSpec,
    SampleRun,
    TickModel,
    TimingScene,
    build_timing_scene,
    sample_native_render_spec,
)
from sva_toolkit.timing.render2.audit.leakage import audit_rendered_text
from sva_toolkit.timing.render2.result import RenderResult
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples" / "td"


def document_and_scene():
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document
    return document, build_timing_scene(visual, semantic_document=document)


def adapter_spec(renderer_id: str, *, extras: dict[str, str] | None = None) -> RenderSpec:
    spec = sample_native_render_spec(random.Random(19), profile="clean-native")
    return replace(
        spec,
        renderer_id=renderer_id,
        profile=f"clean-{renderer_id}",
        annotations=replace(
            spec.annotations,
            policy=AnnotationPolicy.NONE,
            semantic_guides_enabled=False,
            nuisance_text_count=0,
        ),
        page=replace(spec.page, enabled=False),
        extras=extras or {},
    )


def assert_basic_render_result(result: RenderResult) -> None:
    assert result.svg_text or result.png_bytes or result.ascii_text
    assert result.visibility.rendered_text


def assert_leakage_passes(scene: TimingScene, result: RenderResult) -> None:
    report = audit_rendered_text(scene, result, target_dsl_text=emit_timing_dsl(scene.semantic_document))
    assert report.passed, report.format()
    assert "a0" not in report.rendered_tokens


def high_z_scene() -> TimingScene:
    return TimingScene(
        name="high_z",
        clocking_edge="posedge",
        clocking_signal="clk",
        lanes=(
            LaneScene(
                name="clk",
                lane_type=LaneType.CLOCK,
                runs=(SampleRun(0, 0, "1"), SampleRun(1, 1, "0")),
            ),
            LaneScene(
                name="bus",
                lane_type=LaneType.HIGH_Z,
                runs=(SampleRun(0, 1, "z", is_high_z=True),),
            ),
        ),
        ticks=TickModel(total_ticks=2),
        cuts=(),
        events=(),
        constraints=(),
    )
