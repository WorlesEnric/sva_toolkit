"""Renderer-independent timing scene and renderer protocol API."""

import importlib.util

from sva_toolkit.timing.render2.decorations import (
    AnnotationPolicy,
    Decoration,
    DecorationKind,
    DecorationStyle,
    select_decorations,
)
from sva_toolkit.timing.render2.primitives import (
    ALLOWED_PRIMITIVE_ROLES,
    BBox,
    Fill,
    FontSpec,
    Group,
    Line,
    Path,
    Point,
    Polyline,
    Primitive,
    Rect,
    Stroke,
    Text,
)
from sva_toolkit.timing.render2.protocol import (
    DEFAULT_REGISTRY,
    KNOWN_CAPABILITIES,
    RendererRegistry,
    TimingRenderer,
)
from sva_toolkit.timing.render2.result import (
    DiagramLayout,
    RenderResult,
    TextPrimitive,
    VisibilityReport,
    VisualVisibilityReport,
)
from sva_toolkit.timing.render2.scene import (
    CutRegion,
    LaneScene,
    LaneType,
    SampleRun,
    TickModel,
    TimingScene,
    VisualConstraint,
    VisualEvent,
)
from sva_toolkit.timing.render2.scene_builder import build_timing_scene
from sva_toolkit.timing.render2.serialization import (
    from_dict,
    result_from_dict,
    result_to_dict,
    scene_from_dict,
    scene_to_dict,
    spec_from_dict,
    spec_to_dict,
    to_dict,
)
from sva_toolkit.timing.render2.spec import (
    AnnotationSpec,
    DegradationSpec,
    LayoutSpec,
    PageSpec,
    RasterSpec,
    RenderSpec,
    StyleSpec,
)
from sva_toolkit.timing.render2.native.renderer import NativeSvgRenderer
from sva_toolkit.timing.render2.native.sampler import sample_native_render_spec
from sva_toolkit.timing.render2.profiles import (
    ALL_PROFILES,
    PROFILE_ASCII_RFC,
    PROFILE_BY_ID,
    PROFILE_CLEAN_WAVEDROM,
    PROFILE_DATASHEET_NATIVE,
    PROFILE_DEBUG_CURRENT,
    PROFILE_DOCUMENT_NATIVE,
    PROFILE_GTKWAVE_OOD,
    PROFILE_NATIVE_RANDOM,
    PROFILE_OOD_NATIVE,
    PROFILE_PLANTUML_OOD,
    PROFILE_SET_BY_ID,
    PROFILE_SET_TEST_OOD,
    PROFILE_SET_TRAIN_V2,
    PROFILE_SET_VAL_SEEN_STYLE,
    PROFILE_SET_VAL_UNSEEN_STYLE,
    PROFILE_TIKZ_DATASHEET,
    PROFILE_UNDULATE_RANDOM,
    ProfileSet,
    RenderProfile,
)
from sva_toolkit.timing.render2.spec_sampler import sample_profile, sample_render_spec
from sva_toolkit.timing.render2.visual_coverage import VisualCoverageTracker
from sva_toolkit.timing.render2.adapters.ascii import ASCIIAdapter
from sva_toolkit.timing.render2.adapters.gtkwave import GTKWaveAdapter
from sva_toolkit.timing.render2.adapters.plantuml import PlantUMLAdapter
from sva_toolkit.timing.render2.adapters.registry_bootstrap import bootstrap_external_renderers
from sva_toolkit.timing.render2.adapters.tikz_timing import TikzTimingAdapter
from sva_toolkit.timing.render2.adapters.undulate import UndulateAdapter
from sva_toolkit.timing.render2.adapters.wavedrom import WaveDromAdapter
from sva_toolkit.timing.render2.compose import ComposedRecord, compose_record
from sva_toolkit.timing.render2.degrade import DegradationOperation, DegradationPipeline
from sva_toolkit.timing.render2.page_composer import PageComposer
from sva_toolkit.timing.render2.pipeline import RenderOutcome, render
from sva_toolkit.timing.render2.rasterize import rasterize_svg


try:
    DEFAULT_REGISTRY.get("native_svg")
except KeyError:
    DEFAULT_REGISTRY.register(NativeSvgRenderer())

if importlib.util.find_spec("wavedrom") is not None:
    try:
        DEFAULT_REGISTRY.get("wavedrom")
    except KeyError:
        DEFAULT_REGISTRY.register(WaveDromAdapter())

bootstrap_external_renderers(DEFAULT_REGISTRY)

__all__ = [
    "ALLOWED_PRIMITIVE_ROLES",
    "ALL_PROFILES",
    "ASCIIAdapter",
    "DEFAULT_REGISTRY",
    "KNOWN_CAPABILITIES",
    "AnnotationPolicy",
    "AnnotationSpec",
    "BBox",
    "ComposedRecord",
    "CutRegion",
    "Decoration",
    "DecorationKind",
    "DecorationStyle",
    "DegradationOperation",
    "DegradationPipeline",
    "DegradationSpec",
    "DiagramLayout",
    "Fill",
    "FontSpec",
    "GTKWaveAdapter",
    "Group",
    "LaneScene",
    "LaneType",
    "LayoutSpec",
    "Line",
    "NativeSvgRenderer",
    "PROFILE_ASCII_RFC",
    "PROFILE_BY_ID",
    "PROFILE_CLEAN_WAVEDROM",
    "PROFILE_DATASHEET_NATIVE",
    "PROFILE_DEBUG_CURRENT",
    "PROFILE_DOCUMENT_NATIVE",
    "PROFILE_GTKWAVE_OOD",
    "PROFILE_NATIVE_RANDOM",
    "PROFILE_OOD_NATIVE",
    "PROFILE_PLANTUML_OOD",
    "PROFILE_SET_BY_ID",
    "PROFILE_SET_TEST_OOD",
    "PROFILE_SET_TRAIN_V2",
    "PROFILE_SET_VAL_SEEN_STYLE",
    "PROFILE_SET_VAL_UNSEEN_STYLE",
    "PROFILE_TIKZ_DATASHEET",
    "PROFILE_UNDULATE_RANDOM",
    "PageComposer",
    "PageSpec",
    "PlantUMLAdapter",
    "Path",
    "Point",
    "Polyline",
    "Primitive",
    "ProfileSet",
    "RasterSpec",
    "Rect",
    "RenderResult",
    "RenderOutcome",
    "RenderProfile",
    "RenderSpec",
    "RendererRegistry",
    "SampleRun",
    "Stroke",
    "StyleSpec",
    "Text",
    "TextPrimitive",
    "TickModel",
    "TikzTimingAdapter",
    "TimingRenderer",
    "TimingScene",
    "UndulateAdapter",
    "VisibilityReport",
    "VisualConstraint",
    "VisualEvent",
    "VisualCoverageTracker",
    "VisualVisibilityReport",
    "WaveDromAdapter",
    "build_timing_scene",
    "bootstrap_external_renderers",
    "compose_record",
    "from_dict",
    "rasterize_svg",
    "result_from_dict",
    "result_to_dict",
    "render",
    "scene_from_dict",
    "scene_to_dict",
    "select_decorations",
    "sample_profile",
    "sample_native_render_spec",
    "sample_render_spec",
    "spec_from_dict",
    "spec_to_dict",
    "to_dict",
]
