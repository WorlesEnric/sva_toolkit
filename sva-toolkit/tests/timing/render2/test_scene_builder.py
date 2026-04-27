from __future__ import annotations

from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2 import LaneType, build_timing_scene
from sva_toolkit.timing.visual import lower_to_visual_document


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "td"


def test_handshake_scene_uses_lowered_anchor_names_and_ticks() -> None:
    document = parse_diagram((EXAMPLES_DIR / "01_simple_handshake.td").read_text())
    visual = lower_to_visual_document(document).visual_document

    scene = build_timing_scene(visual, semantic_document=document)

    assert len(scene.lanes) == 3
    assert scene.ticks.total_ticks == 6
    assert tuple(event.name for event in scene.events) == ("a0", "a1")
    assert tuple(event.tick for event in scene.events) == (4, 2)
    assert scene.constraints == ()
    assert scene.visible_target is visual
    assert scene.semantic_document is document


def test_symbolic_pipeline_bus_lanes_group_equal_values() -> None:
    document = parse_diagram((EXAMPLES_DIR / "07_symbolic_pipeline.td").read_text())
    visual = lower_to_visual_document(document).visual_document

    scene = build_timing_scene(visual)

    bus_lanes = {lane.name: lane for lane in scene.lanes if lane.lane_type == LaneType.BUS}
    assert set(bus_lanes) == {"din", "dout"}
    assert any(run.value == "din" and run.start_tick == 1 and run.end_tick == 2 for run in bus_lanes["din"].runs)
    assert bus_lanes["dout"].runs[0].is_unknown
    assert bus_lanes["dout"].runs[0].start_tick == 0
    assert bus_lanes["dout"].runs[0].end_tick == scene.ticks.total_ticks - 1


def test_cuts_translate_to_cut_regions() -> None:
    document = parse_diagram((EXAMPLES_DIR / "04_hold_until_ready.td").read_text())
    visual = lower_to_visual_document(document).visual_document

    scene = build_timing_scene(visual)

    assert tuple(cut.meaning for cut in scene.cuts) == ("omitted_history", "omitted_future")
    assert scene.cuts[0].start_tick == 0
    assert scene.cuts[0].end_tick == 0
    assert scene.cuts[1].start_tick == 6
    assert scene.cuts[1].end_tick == 7


def test_scene_build_is_deterministic_and_structural() -> None:
    document = parse_diagram((EXAMPLES_DIR / "07_symbolic_pipeline.td").read_text())
    visual = lower_to_visual_document(document).visual_document

    first = build_timing_scene(visual)
    second = build_timing_scene(visual)

    assert first == second
    assert hash(first) == hash(second)
