from __future__ import annotations

import json

import pytest

from sva_toolkit.timing.render2 import (
    AnnotationPolicy,
    AnnotationSpec,
    BBox,
    DegradationSpec,
    DiagramLayout,
    FontSpec,
    LayoutSpec,
    Line,
    PageSpec,
    Point,
    RasterSpec,
    RenderResult,
    RenderSpec,
    Stroke,
    StyleSpec,
    TextPrimitive,
    VisualVisibilityReport,
)
from sva_toolkit.timing.render2.serialization import from_dict, result_from_dict, result_to_dict, spec_from_dict, spec_to_dict, to_dict


def test_round_trips_every_sub_spec() -> None:
    for cls, spec in (
        (StyleSpec, _style_spec()),
        (LayoutSpec, _layout_spec()),
        (AnnotationSpec, _annotation_spec()),
        (PageSpec, _page_spec()),
        (RasterSpec, _raster_spec()),
        (DegradationSpec, _degradation_spec()),
    ):
        assert from_dict(cls, to_dict(spec)) == spec


def test_complete_render_spec_survives_json_round_trip() -> None:
    spec = _render_spec()

    payload = json.loads(json.dumps(spec_to_dict(spec), sort_keys=True))

    assert spec_from_dict(payload) == spec
    assert dict(spec_from_dict(payload).extras) == {"adapter": "native", "variant": "dense"}


def test_render_result_round_trips_ascii_text() -> None:
    spec = _render_spec()
    result = RenderResult(
        svg_text=None,
        png_bytes=None,
        layout=DiagramLayout(width=10, height=10, plot_origin=Point(0, 0), tick_width=1, lane_height=1, lane_pitch=1),
        visibility=VisualVisibilityReport(
            rendered_text=(TextPrimitive("req", BBox(0, 0, 5, 5), "lane_label", "visible_text"),),
            target_tokens_visible=frozenset({"req"}),
            nuisance_tokens=frozenset(),
            debug_overlay_tokens=frozenset(),
            leaked_tokens=frozenset(),
            occluded_lane_fractions={"req": 0.0},
            minimum_contrast=1.0,
        ),
        render_spec=spec,
        ascii_text="req  ___/--",
    )

    assert result_from_dict(result_to_dict(result)).ascii_text == "req  ___/--"


def test_invalid_primitive_role_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid primitive role"):
        Line(role="not_a_role")


def _render_spec() -> RenderSpec:
    return RenderSpec(
        renderer_id="stub",
        style=_style_spec(),
        layout=_layout_spec(),
        annotations=_annotation_spec(),
        page=_page_spec(),
        raster=_raster_spec(),
        degradation=_degradation_spec(),
        seed=123,
        profile="debug-current",
        extras={"adapter": "native", "variant": "dense"},
    )


def _style_spec() -> StyleSpec:
    return StyleSpec(
        family="datasheet_bw_dense_grid",
        palette=("#000000", "#ffffff", "#dddddd"),
        primary_font=FontSpec(size_px=12),
        label_font=FontSpec(size_px=11, weight="600"),
        waveform_stroke=Stroke(width=1.5),
        grid_stroke=Stroke(color="#cccccc", width=0.5, dasharray=(2.0, 1.0)),
        grid_mode="major_minor",
        bus_style="filled",
        unknown_style="x_hatch",
        cut_style="zigzag",
        transition_shape="sharp",
        color_mode="monochrome",
    )


def _layout_spec() -> LayoutSpec:
    return LayoutSpec(
        lane_height=28.0,
        lane_pitch=32.0,
        tick_width=40.0,
        label_position="left",
        label_alignment="end",
        margin=BBox(10, 12, 14, 16),
        grouped_lanes=True,
        multiline_labels=False,
    )


def _annotation_spec() -> AnnotationSpec:
    return AnnotationSpec(
        policy=AnnotationPolicy.GEOMETRIC_GUIDES,
        measurement_label_style="datasheet",
        helper_line_density=0.25,
        nuisance_text_count=2,
        semantic_guides_enabled=True,
    )


def _page_spec() -> PageSpec:
    return PageSpec(
        enabled=False,
        caption_above=True,
        caption_below=False,
        surrounding_paragraph=False,
        table_border=True,
        page_header=False,
        page_footer=False,
        crop_mode="tight",
    )


def _raster_spec() -> RasterSpec:
    return RasterSpec(dpi=150, antialias=True, output_format="png", jpeg_quality=90)


def _degradation_spec() -> DegradationSpec:
    return DegradationSpec(
        family="clean",
        blur_sigma=0.1,
        noise_sigma=0.2,
        contrast=0.95,
        brightness=1.05,
        rotation_deg=0.5,
        perspective=0.0,
        jpeg_quality=92,
        morphology="none",
        augraphy_pipeline=None,
    )
