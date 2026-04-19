from __future__ import annotations

from sva_toolkit.describe import CoTSection, SVACoTBuilder


def test_cot_builder_returns_markdown_for_raw_sva_code() -> None:
    builder = SVACoTBuilder()

    rendered = builder.build(
        """
        property req_ack;
            @(posedge clk) req |-> ##[1:3] ack;
        endproperty
        """
    )

    assert "# SVA Generation Chain-of-Thought" in rendered
    assert "Step 1: Interface & Clock Domain Analysis" in rendered
    assert "Step 2: Semantic Mapping" in rendered
    assert "Step 3: Sequence Construction" in rendered
    assert "Step 4: Property Assembly" in rendered
    assert "Step 5: Final SVA Code" in rendered
    assert "req_ack" in rendered
    assert "##[1:3]" in rendered or "1 to 3" in rendered


def test_cot_builder_exposes_structured_sections() -> None:
    builder = SVACoTBuilder()

    sections = builder.get_cot_sections(
        """
        assert property (@(posedge clk) disable iff (!rst_n)
            $rose(req) |=> ##1 $stable(data));
        """
    )

    assert sections
    assert all(isinstance(section, CoTSection) for section in sections)
    assert sections[0].title == "Header"
    assert any("Interface" in section.title for section in sections)
    assert any("Semantic" in section.title for section in sections)
    assert any("Property Assembly" == section.title for section in sections)
    assert any("disable iff" in section.content.lower() or "rst_n" in section.content for section in sections)
    assert any("$rose" in section.content for section in sections)
    assert any("$stable" in section.content for section in sections)
