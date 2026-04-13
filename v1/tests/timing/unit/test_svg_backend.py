"""Tests for the timing SVG renderers."""

from xml.etree import ElementTree as ET

from sva_toolkit.timing import extract_sva_scenario, parse_diagram, render_diagram_svg

from .test_parser import SAMPLE_DIAGRAM, SYMBOLIC_DIAGRAM


SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _parse_svg(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_svg_renderer_keeps_legacy_concrete_rendering_path():
    diagram = parse_diagram(SAMPLE_DIAGRAM)

    svg = render_diagram_svg(diagram)
    root = _parse_svg(svg)

    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert any("WaveDrom" in (element.get("class") or "") for element in root.findall(".//svg:svg", SVG_NS))
    assert "req_ack" in svg


def test_svg_renderer_draws_symbolic_windows_cuts_and_footer():
    diagram = parse_diagram(SYMBOLIC_DIAGRAM)

    svg = render_diagram_svg(diagram)
    root = _parse_svg(svg)

    assert "READY_WAIT_MAX" in svg
    assert "EXACT" in svg
    assert root.find(".//svg:path[@class='cut-line']", SVG_NS) is not None
    assert root.find(".//svg:text[@class='window-label']", SVG_NS) is not None
    assert "ready_after_wait: exact" in svg


def test_svg_renderer_marks_lossy_extracted_sva():
    document = extract_sva_scenario("property p; @(posedge clk) req |-> data until done; endproperty")

    svg = render_diagram_svg(document)

    assert "LOSSY" in svg
    assert "hold-until skeleton" in svg
