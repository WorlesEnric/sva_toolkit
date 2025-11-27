"""Tests for SVA Implication Checker with real SVA cases."""

import pytest
from unittest.mock import patch, MagicMock
import shutil

from sva_toolkit.implication_checker import SVAImplicationChecker
from sva_toolkit.implication_checker.checker import (
    ImplicationResult,
    CheckResult,
)


class TestSVAImplicationCheckerUnit:
    """Unit tests for SVAImplicationChecker helper methods."""

    @pytest.fixture
    def mock_checker(self):
        """Create a checker with mocked subprocess calls."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="PROVED",
                stderr=""
            )
            checker = SVAImplicationChecker()
            checker._mock_run = mock_run
            yield checker

    def test_extract_property_removes_clock_and_disable(self, mock_checker):
        """Test that clock and disable iff are properly removed."""
        sva = "@(posedge clk) disable iff (!rst_n) req |-> ##[1:3] gnt"
        expr = mock_checker._extract_property_expression(sva)
        assert "@" not in expr
        assert "disable" not in expr.lower()
        assert "req" in expr
        assert "gnt" in expr

    def test_extract_property_from_assert_property(self, mock_checker):
        """Test extraction from assert property statement."""
        sva = "assert property (@(posedge clk) req |-> gnt);"
        expr = mock_checker._extract_property_expression(sva)
        assert "assert" not in expr.lower()
        assert "req" in expr
        assert "gnt" in expr

    def test_extract_property_preserves_parens_in_bare_expression(self, mock_checker):
        """Test that parens in bare property expression are preserved."""
        # Regression test for bug where trailing ')' was stripped from bare expressions
        sva = "req |-> ##1 (gnt && !error)"
        expr = mock_checker._extract_property_expression(sva)
        assert expr == "req |-> ##1 (gnt && !error)"

    def test_collect_signals_from_complex_sva(self, mock_checker):
        """Test signal collection from complex SVA expressions."""
        sva1 = "valid && ready && data[7:0] == 8'hFF |-> ##1 ack"
        sva2 = "start_tx |-> ##[1:10] done_tx && !error"
        signals = mock_checker._collect_signals(sva1, sva2)
        
        assert "valid" in signals
        assert "ready" in signals
        assert "ack" in signals
        assert "start_tx" in signals
        assert "done_tx" in signals
        assert "error" in signals

    def test_collect_signals_excludes_keywords(self, mock_checker):
        """Test that clock/reset keywords are excluded from signals."""
        sva = "clk && rst_n && valid |-> ready"
        signals = mock_checker._collect_signals(sva)
        
        assert "clk" not in signals
        assert "rst_n" not in signals
        assert "valid" in signals
        assert "ready" in signals


@pytest.fixture
def checker():
    """Create checker, skip if EBMC not installed."""
    if not shutil.which("ebmc"):
        pytest.skip("EBMC not installed")
    return SVAImplicationChecker(depth=15, keep_files=False)


class TestReflexiveImplication:
    """Test reflexive property: A |= A."""

    def test_identical_simple_sequence(self, checker):
        """Identical sequences should imply each other."""
        sva = "req |-> ##1 gnt"
        result = checker.check_implication(sva, sva)
        assert result.result == ImplicationResult.IMPLIES

    def test_identical_bounded_sequence(self, checker):
        """Identical bounded sequences should imply each other."""
        sva = "start |-> ##[1:5] done"
        result = checker.check_implication(sva, sva)
        assert result.result == ImplicationResult.IMPLIES


class TestSequenceDelayImplication:
    """Test implication relationships with sequence delays."""

    def test_exact_delay_implies_range(self, checker):
        """Exact delay ##N should imply range ##[1:M] where N is in range."""
        # req ##2 gnt implies req ##[1:3] gnt (2 is within [1,3])
        sva_exact = "req ##2 gnt"
        sva_range = "req ##[1:3] gnt"
        result = checker.check_implication(sva_exact, sva_range)
        assert result.result == ImplicationResult.IMPLIES

    def test_range_does_not_imply_exact(self, checker):
        """Range ##[1:M] should NOT imply exact ##N."""
        sva_range = "req ##[1:3] gnt"
        sva_exact = "req ##2 gnt"
        result = checker.check_implication(sva_range, sva_exact)
        assert result.result == ImplicationResult.NOT_IMPLIES

    def test_narrow_range_implies_wider_range(self, checker):
        """Narrower delay range should imply wider range."""
        sva_narrow = "req ##[2:3] gnt"
        sva_wide = "req ##[1:5] gnt"
        result = checker.check_implication(sva_narrow, sva_wide)
        assert result.result == ImplicationResult.IMPLIES

    def test_wider_range_does_not_imply_narrow(self, checker):
        """Wider delay range should NOT imply narrower range."""
        sva_wide = "req ##[1:5] gnt"
        sva_narrow = "req ##[2:3] gnt"
        result = checker.check_implication(sva_wide, sva_narrow)
        assert result.result == ImplicationResult.NOT_IMPLIES

    def test_zero_delay_vs_one_delay(self, checker):
        """##0 (same cycle) vs ##1 (next cycle)."""
        sva_same_cycle = "req ##0 gnt"
        sva_next_cycle = "req ##1 gnt"
        # These should NOT imply each other
        result = checker.check_implication(sva_same_cycle, sva_next_cycle)
        assert result.result == ImplicationResult.NOT_IMPLIES


class TestAntecedentStrengthening:
    """Test that stronger antecedents imply weaker consequents."""

    def test_stronger_antecedent_implies(self, checker):
        """Stronger antecedent (A && B) should imply weaker antecedent (A)."""
        # (req && valid) |-> gnt  IMPLIES  req |-> gnt? NO!
        # Actually the opposite: if we ASSUME (req && valid) |-> gnt,
        # we cannot conclude req |-> gnt (req alone might not get gnt)
        sva_strong = "(req && valid) |-> ##1 gnt"
        sva_weak = "req |-> ##1 gnt"
        result = checker.check_implication(sva_strong, sva_weak)
        assert result.result == ImplicationResult.NOT_IMPLIES

    def test_weaker_antecedent_implies_stronger(self, checker):
        """Property with weaker antecedent implies property with stronger antecedent."""
        # If req |-> gnt holds for ALL req, then (req && valid) |-> gnt also holds
        sva_weak_ante = "req |-> ##1 gnt"
        sva_strong_ante = "(req && valid) |-> ##1 gnt"
        result = checker.check_implication(sva_weak_ante, sva_strong_ante)
        assert result.result == ImplicationResult.IMPLIES


class TestConsequentStrengthening:
    """Test consequent strength relationships."""

    def test_stronger_consequent_implies_weaker(self, checker):
        """Stronger consequent implies weaker consequent."""
        # (gnt && !error) is stronger than just gnt
        sva_strong_cons = "req |-> ##1 (gnt && !error)"
        sva_weak_cons = "req |-> ##1 gnt"
        result = checker.check_implication(sva_strong_cons, sva_weak_cons)
        assert result.result == ImplicationResult.IMPLIES

    def test_weaker_consequent_does_not_imply_stronger(self, checker):
        """Weaker consequent does NOT imply stronger consequent."""
        sva_weak_cons = "req |-> ##1 gnt"
        sva_strong_cons = "req |-> ##1 (gnt && !error)"
        result = checker.check_implication(sva_weak_cons, sva_strong_cons)
        assert result.result == ImplicationResult.NOT_IMPLIES


class TestEquivalenceChecks:
    """Test equivalence (bidirectional implication)."""

    def test_identical_properties_equivalent(self, checker):
        """Identical properties should be equivalent."""
        sva = "req |-> ##[1:3] gnt"
        result = checker.check_equivalence(sva, sva)
        assert result.result == ImplicationResult.EQUIVALENT

    def test_different_delays_not_equivalent(self, checker):
        """Different delay ranges should not be equivalent."""
        sva1 = "req |-> ##1 gnt"
        sva2 = "req |-> ##2 gnt"
        result = checker.check_equivalence(sva1, sva2)
        assert result.result == ImplicationResult.NOT_IMPLIES

    def test_reordered_conjunction_equivalent(self, checker):
        """Reordered conjunctions should be equivalent."""
        sva1 = "(a && b) |-> ##1 c"
        sva2 = "(b && a) |-> ##1 c"
        result = checker.check_equivalence(sva1, sva2)
        assert result.result == ImplicationResult.EQUIVALENT


class TestOverlappingImplication:
    """Test overlapping vs non-overlapping implication operators."""

    def test_overlapping_implication(self, checker):
        """Test overlapping implication |->."""
        # a |-> b means if a is true, b must be true in the SAME cycle
        sva1 = "req |-> gnt"
        sva2 = "req |-> ##0 gnt"  # Same as above
        result = checker.check_equivalence(sva1, sva2)
        assert result.result == ImplicationResult.EQUIVALENT

    def test_non_overlapping_implication(self, checker):
        """Test non-overlapping implication |=>."""
        # a |=> b is equivalent to a |-> ##1 b
        sva1 = "req |=> gnt"
        sva2 = "req |-> ##1 gnt"
        result = checker.check_equivalence(sva1, sva2)
        assert result.result == ImplicationResult.EQUIVALENT


class TestRepetitionOperators:
    """Test SVA repetition operators."""

    def test_exact_repetition_implies_range(self, checker):
        """Exact repetition should imply range containing it."""
        sva_exact = "req ##1 data[*3] ##1 done"
        sva_range = "req ##1 data[*2:4] ##1 done"
        result = checker.check_implication(sva_exact, sva_range)
        assert result.result == ImplicationResult.IMPLIES

    def test_consecutive_repetition(self, checker):
        """Test consecutive repetition [*n]."""
        # a[*2] means a must be true for 2 consecutive cycles
        sva1 = "start ##1 busy[*3] ##1 done"
        sva2 = "start ##1 busy[*2:4] ##1 done"
        result = checker.check_implication(sva1, sva2)
        assert result.result == ImplicationResult.IMPLIES


class TestProtocolAssertions:
    """Test real-world protocol-like assertions."""

    def test_handshake_protocol_implication(self, checker):
        """Test AXI-like handshake: valid && ready implies transfer."""
        # Stronger handshake implies weaker
        sva_strong = "(valid && ready) |-> ##1 (data_valid && !error)"
        sva_weak = "(valid && ready) |-> ##1 data_valid"
        result = checker.check_implication(sva_strong, sva_weak)
        assert result.result == ImplicationResult.IMPLIES

    def test_request_grant_timing(self, checker):
        """Test request-grant timing constraints."""
        # Tighter timing implies looser timing
        sva_tight = "req |-> ##[1:2] gnt"
        sva_loose = "req |-> ##[1:5] gnt"
        result = checker.check_implication(sva_tight, sva_loose)
        assert result.result == ImplicationResult.IMPLIES

    def test_bus_protocol_data_stability(self, checker):
        """Test data stability during transfer."""
        # If data is stable for more cycles, it implies stability for fewer
        sva_long_stable = "valid |-> data_out[*4]"
        sva_short_stable = "valid |-> data_out[*2]"
        result = checker.check_implication(sva_long_stable, sva_short_stable)
        assert result.result == ImplicationResult.IMPLIES


class TestGetImplicationRelationship:
    """Test the get_implication_relationship method."""

    def test_one_way_implication(self, checker):
        """Test when only one direction implies."""
        sva1 = "req ##2 gnt"  # Exact delay
        sva2 = "req ##[1:3] gnt"  # Range
        
        implies_forward, implies_backward = checker.get_implication_relationship(sva1, sva2)
        
        assert implies_forward is True  # sva1 -> sva2
        assert implies_backward is False  # sva2 -> sva1

    def test_equivalence_relationship(self, checker):
        """Test when both directions imply (equivalence)."""
        sva1 = "a && b |-> ##1 c"
        sva2 = "b && a |-> ##1 c"
        
        implies_forward, implies_backward = checker.get_implication_relationship(sva1, sva2)
        
        assert implies_forward is True
        assert implies_backward is True

    def test_no_relationship(self, checker):
        """Test when neither direction implies."""
        sva1 = "a |-> ##1 b"
        sva2 = "c |-> ##1 d"
        
        implies_forward, implies_backward = checker.get_implication_relationship(sva1, sva2)
        
        assert implies_forward is False
        assert implies_backward is False


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_signal_property(self, checker):
        """Test properties with single signals."""
        sva1 = "valid"
        sva2 = "valid"
        result = checker.check_equivalence(sva1, sva2)
        assert result.result == ImplicationResult.EQUIVALENT

    def test_negation_not_equivalent(self, checker):
        """Negated property should not be equivalent to original."""
        sva1 = "req |-> ##1 gnt"
        sva2 = "req |-> ##1 !gnt"
        result = checker.check_equivalence(sva1, sva2)
        assert result.result == ImplicationResult.NOT_IMPLIES

    def test_always_true_antecedent(self, checker):
        """Test with always-true antecedent (1'b1)."""
        sva1 = "1'b1 |-> ##1 done"
        sva2 = "start |-> ##1 done"
        # 1'b1 |-> done means done must always be true after 1 cycle
        # This is stronger than start |-> done
        result = checker.check_implication(sva1, sva2)
        assert result.result == ImplicationResult.IMPLIES


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_syntax_error_in_sva(self, checker):
        """Test handling of SVA with syntax errors."""
        sva_invalid = "req |-> ##[1:] gnt"  # Invalid range
        sva_valid = "req |-> ##1 gnt"
        result = checker.check_implication(sva_invalid, sva_valid)
        assert result.result == ImplicationResult.ERROR

    def test_empty_expression_handling(self, checker):
        """Test handling of empty expressions."""
        # This should either error or handle gracefully
        sva_empty = ""
        sva_valid = "req |-> gnt"
        result = checker.check_implication(sva_empty, sva_valid)
        # Should either error or not crash
        assert result.result in [ImplicationResult.ERROR, ImplicationResult.NOT_IMPLIES]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])