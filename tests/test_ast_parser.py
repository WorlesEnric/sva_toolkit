"""Tests for SVA AST Parser."""

import pytest
from sva_toolkit.ast_parser import SVAASTParser, SVAStructure
from sva_toolkit.ast_parser.parser import ImplicationType, TemporalOperator


class TestSVAASTParser:
    """Tests for SVAASTParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return SVAASTParser()

    def test_parse_simple_property(self, parser):
        """Test parsing a simple property."""
        sva = """
        property req_ack;
            @(posedge clk) req |-> ##[1:3] ack;
        endproperty
        """
        structure = parser.parse(sva)
        
        assert structure.property_name == "req_ack"
        assert structure.clock_signal == "clk"
        assert structure.clock_edge == "posedge"
        assert structure.implication_type == ImplicationType.OVERLAPPING
        assert "req" in structure.antecedent
        assert "ack" in structure.consequent

    def test_parse_with_disable_iff(self, parser):
        """Test parsing property with disable iff."""
        sva = """
        property safe_req;
            @(posedge clk) disable iff (!rst_n)
            req |-> ##1 ack;
        endproperty
        """
        structure = parser.parse(sva)
        
        assert structure.disable_condition == "!rst_n"
        assert structure.reset_signal == "rst_n"
        assert structure.reset_active_low is True

    def test_parse_non_overlapping_implication(self, parser):
        """Test parsing non-overlapping implication."""
        sva = """
        property delayed_resp;
            @(posedge clk) start |=> done;
        endproperty
        """
        structure = parser.parse(sva)
        
        assert structure.implication_type == ImplicationType.NON_OVERLAPPING

    def test_extract_builtin_functions(self, parser):
        """Test extraction of built-in functions."""
        sva = """
        property edge_detect;
            @(posedge clk) $rose(req) |-> ##1 $stable(data);
        endproperty
        """
        structure = parser.parse(sva)
        
        func_names = [f.name for f in structure.builtin_functions]
        assert "$rose" in func_names
        assert "$stable" in func_names

    def test_extract_signals(self, parser):
        """Test extraction of signals."""
        sva = """
        property data_valid;
            @(posedge clk) valid && ready |-> ##1 data_out == expected;
        endproperty
        """
        structure = parser.parse(sva)
        
        signal_names = [s.name for s in structure.signals]
        assert "valid" in signal_names
        assert "ready" in signal_names
        assert "data_out" in signal_names
        assert "expected" in signal_names

    def test_extract_delays(self, parser):
        """Test extraction of delay specifications."""
        sva = """
        property timing;
            @(posedge clk) req |-> ##[1:5] ack ##2 done;
        endproperty
        """
        structure = parser.parse(sva)
        
        assert len(structure.delays) >= 2
        # Check for range delay
        range_delay = next((d for d in structure.delays if d.max_cycles == 5), None)
        assert range_delay is not None
        assert range_delay.min_cycles == 1

    def test_extract_temporal_operators(self, parser):
        """Test extraction of temporal operators."""
        sva = """
        property complex;
            @(posedge clk) (req ##1 ack)[*3] |-> done;
        endproperty
        """
        structure = parser.parse(sva)
        
        assert TemporalOperator.DELAY in structure.temporal_operators
        assert TemporalOperator.REPETITION_CONSECUTIVE in structure.temporal_operators

    def test_to_dict(self, parser):
        """Test conversion to dictionary."""
        sva = """
        property simple;
            @(posedge clk) a |-> b;
        endproperty
        """
        structure = parser.parse(sva)
        d = structure.to_dict()
        
        assert "property_name" in d
        assert "clock_signal" in d
        assert "implication_type" in d
        assert d["property_name"] == "simple"

    def test_get_all_signals(self, parser):
        """Test get_all_signals method."""
        sva = "property p; @(posedge clk) x && y |-> z; endproperty"
        signals = parser.get_all_signals(sva)
        
        assert "x" in signals
        assert "y" in signals
        assert "z" in signals


class TestSVAStructure:
    """Tests for SVAStructure class."""

    def test_to_dict_complete(self):
        """Test complete to_dict conversion."""
        from sva_toolkit.ast_parser.parser import Signal, BuiltinFunction, DelayRange
        
        structure = SVAStructure(
            raw_code="property p; a |-> b; endproperty",
            property_name="p",
            clock_signal="clk",
            implication_type=ImplicationType.OVERLAPPING,
            antecedent="a",
            consequent="b",
            signals={Signal(name="a"), Signal(name="b")},
            builtin_functions=[BuiltinFunction(name="$rose", arguments=["a"])],
            delays=[DelayRange(min_cycles=1, max_cycles=3)],
        )
        
        d = structure.to_dict()
        
        assert d["property_name"] == "p"
        assert d["clock_signal"] == "clk"
        assert d["implication_type"] == "|->"
        assert len(d["signals"]) == 2
        assert len(d["builtin_functions"]) == 1
        assert len(d["delays"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
