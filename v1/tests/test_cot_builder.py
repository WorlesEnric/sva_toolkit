"""Tests for SVA CoT Builder."""

import pytest
from sva_toolkit.cot_builder import SVACoTBuilder


class TestSVACoTBuilder:
    """Tests for SVACoTBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a builder instance."""
        return SVACoTBuilder()

    def test_build_simple_property(self, builder):
        """Test building CoT for a simple property."""
        sva = """
        property req_ack;
            @(posedge clk) req |-> ##[1:3] ack;
        endproperty
        """
        cot = builder.build(sva)
        
        assert "# SVA Generation Chain-of-Thought" in cot
        assert "Step 1:" in cot
        assert "Step 2:" in cot
        assert "Step 3:" in cot
        assert "Step 4:" in cot
        assert "Step 5:" in cot
        assert "req_ack" in cot

    def test_build_with_builtin_functions(self, builder):
        """Test CoT generation with built-in functions."""
        sva = """
        property edge_detect;
            @(posedge clk) $rose(req) |-> ##1 $stable(data);
        endproperty
        """
        cot = builder.build(sva)
        
        assert "$rose" in cot
        assert "$stable" in cot
        assert "Edge/Change Detection" in cot or "Built-in Functions" in cot

    def test_build_with_disable_iff(self, builder):
        """Test CoT generation with disable iff."""
        sva = """
        property safe_req;
            @(posedge clk) disable iff (!rst_n)
            req |-> ##1 ack;
        endproperty
        """
        cot = builder.build(sva)
        
        assert "disable iff" in cot.lower() or "Disable Condition" in cot
        assert "rst_n" in cot

    def test_build_non_overlapping(self, builder):
        """Test CoT generation for non-overlapping implication."""
        sva = """
        property delayed;
            @(posedge clk) start |=> done;
        endproperty
        """
        cot = builder.build(sva)
        
        assert "non-overlapping" in cot.lower() or "|=>" in cot

    def test_get_cot_sections(self, builder):
        """Test getting CoT as sections."""
        sva = """
        property test_prop;
            @(posedge clk) a |-> b;
        endproperty
        """
        sections = builder.get_cot_sections(sva)
        
        assert len(sections) >= 5
        section_titles = [s.title for s in sections]
        assert "Interface & Clock Domain Analysis" in section_titles or any("Interface" in t for t in section_titles)

    def test_temporal_operators_documented(self, builder):
        """Test that temporal operators are documented in CoT."""
        sva = """
        property complex;
            @(posedge clk) (req ##1 ack)[*3] |-> ##[0:5] done;
        endproperty
        """
        cot = builder.build(sva)
        
        # Should mention delay and repetition
        assert "##" in cot or "delay" in cot.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
