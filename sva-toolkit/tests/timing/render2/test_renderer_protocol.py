from __future__ import annotations

from dataclasses import dataclass

import pytest

from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    AnnotationSpec,
    BBox,
    DegradationSpec,
    DiagramLayout,
    FontSpec,
    LaneScene,
    LaneType,
    LayoutSpec,
    PageSpec,
    Point,
    RasterSpec,
    RenderResult,
    RenderSpec,
    RendererRegistry,
    SampleRun,
    Stroke,
    StyleSpec,
    TickModel,
    TimingRenderer,
    TimingScene,
    VisualVisibilityReport,
)


def test_registry_returns_supporting_renderer_and_filters_missing_capabilities() -> None:
    scene = _scene()
    spec = _spec("stub")
    registry = RendererRegistry()
    good = _StubRenderer("stub", frozenset({"clock", "bit", "vector_text"}))
    missing = _StubRenderer("missing_caps", frozenset({"bit", "vector_text"}))

    registry.register(good)
    registry.register(missing)

    assert isinstance(good, TimingRenderer)
    assert registry.supporting(scene, spec) == (good,)
    assert registry.get("stub").render(scene, spec).render_spec == spec


def test_registry_get_missing_raises_key_error() -> None:
    registry = RendererRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


@dataclass(frozen=True)
class _StubRenderer:
    id: str
    capabilities: frozenset[str]

    def supports(self, scene: TimingScene, spec: RenderSpec) -> bool:
        del scene
        return spec.renderer_id == "stub"

    def render(self, scene: TimingScene, spec: RenderSpec) -> RenderResult:
        del scene
        return RenderResult(
            svg_text="<svg />",
            png_bytes=None,
            layout=DiagramLayout(
                width=100,
                height=50,
                plot_origin=Point(10, 10),
                tick_width=20,
                lane_height=14,
                lane_pitch=18,
                bbox_by_role={"lane_label": (BBox(0, 0, 10, 10),)},
            ),
            visibility=VisualVisibilityReport(
                rendered_text=(),
                target_tokens_visible=frozenset(),
                nuisance_tokens=frozenset(),
                debug_overlay_tokens=frozenset(),
                leaked_tokens=frozenset(),
                occluded_lane_fractions={},
                minimum_contrast=7.0,
            ),
            render_spec=spec,
            warnings=("stub renderer",),
        )


def _scene() -> TimingScene:
    return TimingScene(
        name="tiny",
        clocking_edge="posedge",
        clocking_signal="clk",
        lanes=(
            LaneScene(
                name="clk",
                lane_type=LaneType.CLOCK,
                runs=(SampleRun(0, 0, "1"), SampleRun(1, 1, "0")),
            ),
            LaneScene(
                name="req",
                lane_type=LaneType.BIT,
                runs=(SampleRun(0, 0, "0"), SampleRun(1, 1, "1")),
            ),
        ),
        ticks=TickModel(total_ticks=2),
        cuts=(),
        events=(),
        constraints=(),
    )


def _spec(renderer_id: str) -> RenderSpec:
    return RenderSpec(
        renderer_id=renderer_id,
        style=StyleSpec(
            family="test",
            palette=("#000000", "#ffffff"),
            primary_font=FontSpec(),
            label_font=FontSpec(),
            waveform_stroke=Stroke(),
            grid_stroke=Stroke(color="#cccccc"),
            grid_mode="none",
            bus_style="empty",
            unknown_style="gray_block",
            cut_style="ellipsis",
            transition_shape="sharp",
            color_mode="monochrome",
        ),
        layout=LayoutSpec(
            lane_height=14,
            lane_pitch=18,
            tick_width=20,
            label_position="left",
            label_alignment="end",
            margin=BBox(0, 0, 0, 0),
        ),
        annotations=AnnotationSpec(
            policy=AnnotationPolicy.NONE,
            measurement_label_style="none",
            helper_line_density=0.0,
            nuisance_text_count=0,
            semantic_guides_enabled=False,
        ),
        page=PageSpec(
            enabled=False,
            caption_above=False,
            caption_below=False,
            surrounding_paragraph=False,
            table_border=False,
            page_header=False,
            page_footer=False,
            crop_mode="tight",
        ),
        raster=RasterSpec(dpi=96, antialias=True, output_format="png"),
        degradation=DegradationSpec(family="clean"),
        seed=1,
        profile="debug-current",
    )
