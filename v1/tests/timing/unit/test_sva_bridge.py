"""Tests for SVA extraction, bundling, and DSL emission."""

from textwrap import dedent

from sva_toolkit.timing import bundle_sva_scenarios, emit_timing_dsl, extract_sva_scenario, parse_diagram
from sva_toolkit.timing.core.scenario import ExtractionStatus


def test_extract_sva_exact_range_delay_builds_symbolic_window():
    document = extract_sva_scenario(
        dedent(
            """
            property p_ready_after_wait;
              @(posedge ACLK) disable iff (!ARESETn)
                (valid && !ready) |-> ##[0:READY_WAIT_MAX] (valid && ready);
            endproperty
            """
        )
    )

    assert document.effective_status == ExtractionStatus.EXACT
    assert document.clocking.signal == "ACLK"
    assert document.window_map["p_ready_after_wait__delay_window"].bound.max_delay == "READY_WAIT_MAX"
    assert document.properties[0].body == "(valid && !ready) |-> ##[0:READY_WAIT_MAX] (valid && ready)"


def test_extract_sva_hold_until_marks_plain_until_as_lossy():
    document = extract_sva_scenario(
        "property p; @(posedge clk) req |-> data until done; endproperty"
    )

    assert document.effective_status == ExtractionStatus.LOSSY
    assert any("hold-until skeleton" in note for note in document.properties[0].notes)
    assert any(constraint.relation == "high" for constraint in document.lane_constraints)


def test_extract_sva_unsupported_control_wrapper_is_explicit():
    document = extract_sva_scenario(
        "property p; @(posedge clk) accept_on(err) req |-> ack; endproperty"
    )

    assert document.effective_status == ExtractionStatus.UNSUPPORTED
    assert "unsupported control wrapper" in document.properties[0].notes[0]


def test_emit_timing_dsl_from_extracted_sva_includes_status_and_windows():
    document = extract_sva_scenario(
        "property p; @(posedge clk) req |-> ##[1:$] ack; endproperty"
    )

    dsl_text = emit_timing_dsl(document)

    assert "cut p__history_cut = before trigger omitted;" in dsl_text
    assert "window p__delay_window = between trigger and p__response [1:$];" in dsl_text
    assert "cut p__future_cut = after p__response omitted" in dsl_text
    reparsed = parse_diagram(dsl_text)
    assert {cut.meaning.value for cut in reparsed.cuts} == {"omitted_history", "omitted_future"}


def test_extract_sva_adds_focus_cuts_for_bounded_response_windows():
    document = extract_sva_scenario(
        "property p; @(posedge clk) req |-> ##[1:3] ack; endproperty"
    )

    dsl_text = emit_timing_dsl(document)
    reparsed = parse_diagram(dsl_text)

    assert "cut p__history_cut = before trigger omitted;" in dsl_text
    assert "cut p__future_cut = after p__response omitted;" in dsl_text
    assert {cut.anchor for cut in reparsed.cuts} == {"trigger", "p__response"}


def test_bundle_sva_scenarios_merges_related_documents_and_preserves_properties():
    first = extract_sva_scenario("property p_a; @(posedge clk) req |-> ##1 ack; endproperty")
    second = extract_sva_scenario("property p_b; @(posedge clk) req |-> ready; endproperty")

    bundled = bundle_sva_scenarios([first, second], overlap_threshold=0.2)

    assert len(bundled) == 1
    assert bundled[0].bundle.source_names == ("p_a", "p_b")
    assert len(bundled[0].properties) == 2


def test_bundle_sva_scenarios_renames_duplicate_anchor_identifiers_for_valid_tdg():
    first = extract_sva_scenario("property p_a; @(posedge clk) req |-> ##1 ack; endproperty")
    second = extract_sva_scenario("property p_b; @(posedge clk) req |-> ##[0:2] ready; endproperty")

    bundled = bundle_sva_scenarios([first, second], overlap_threshold=0.2)

    assert len(bundled) == 1
    dsl_text = emit_timing_dsl(bundled[0])
    reparsed = parse_diagram(dsl_text)

    anchor_names = [anchor.name for anchor in reparsed.anchors]
    assert len(anchor_names) == len(set(anchor_names))
    assert "trigger" in anchor_names
    assert "p_b__trigger" in anchor_names
