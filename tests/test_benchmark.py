"""Tests for SVA Benchmark Runner."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from sva_toolkit.benchmark import BenchmarkRunner, BenchmarkResult
from sva_toolkit.benchmark.runner import RelationshipType, SingleResult
from sva_toolkit.utils.llm_client import LLMClient, LLMConfig


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_equivalent_rate(self):
        """Test equivalent rate calculation."""
        result = BenchmarkResult(
            model_name="test",
            total_items=10,
            equivalent_count=5,
        )
        assert result.equivalent_rate == 0.5

    def test_equivalent_rate_zero_items(self):
        """Test equivalent rate with zero items."""
        result = BenchmarkResult(
            model_name="test",
            total_items=0,
        )
        assert result.equivalent_rate == 0

    def test_any_implication_rate(self):
        """Test any implication rate calculation."""
        result = BenchmarkResult(
            model_name="test",
            total_items=10,
            equivalent_count=2,
            generated_implies_reference_count=3,
            reference_implies_generated_count=1,
        )
        assert result.any_implication_rate == 0.6  # (2+3+1)/10

    def test_success_rate(self):
        """Test success rate calculation."""
        result = BenchmarkResult(
            model_name="test",
            total_items=10,
            error_count=2,
        )
        assert result.success_rate == 0.8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult(
            model_name="test-model",
            total_items=5,
            equivalent_count=2,
        )
        d = result.to_dict()
        assert d["model_name"] == "test-model"
        assert d["total_items"] == 5
        assert d["equivalent_count"] == 2
        assert "equivalent_rate" in d


class TestSingleResult:
    """Tests for SingleResult class."""

    def test_single_result_creation(self):
        """Test SingleResult creation."""
        result = SingleResult(
            svad="Request should be acknowledged",
            reference_sva="req |-> ack",
            generated_sva="req |-> ##1 ack",
            relationship=RelationshipType.REFERENCE_IMPLIES_GENERATED,
        )
        assert result.svad == "Request should be acknowledged"
        assert result.relationship == RelationshipType.REFERENCE_IMPLIES_GENERATED


class TestRelationshipType:
    """Tests for RelationshipType enum."""

    def test_relationship_values(self):
        """Test RelationshipType enum values."""
        assert RelationshipType.EQUIVALENT.value == "equivalent"
        assert RelationshipType.GENERATED_IMPLIES_REFERENCE.value == "generated_implies_reference"
        assert RelationshipType.REFERENCE_IMPLIES_GENERATED.value == "reference_implies_generated"
        assert RelationshipType.NO_RELATIONSHIP.value == "no_relationship"
        assert RelationshipType.ERROR.value == "error"


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock(spec=LLMClient)
        client.config = LLMConfig(
            base_url="http://test",
            model="test-model",
            api_key="test-key"
        )
        client.generate.return_value = "property test; @(posedge clk) req |-> ack; endproperty"
        return client

    @pytest.fixture
    def mock_checker(self):
        """Create a mock implication checker."""
        checker = Mock()
        checker.get_implication_relationship.return_value = (True, True)  # Equivalent
        return checker

    @pytest.fixture
    def runner(self, mock_llm_client, mock_checker):
        """Create a BenchmarkRunner with mocks."""
        return BenchmarkRunner(
            llm_clients=[mock_llm_client],
            implication_checker=mock_checker
        )

    def test_clean_sva_output(self, runner):
        """Test SVA output cleaning."""
        # With markdown code blocks
        response = "```systemverilog\nproperty test; endproperty\n```"
        cleaned = runner._clean_sva_output(response)
        assert "```" not in cleaned
        assert "property test" in cleaned

    def test_clean_sva_output_no_markdown(self, runner):
        """Test SVA output cleaning without markdown."""
        response = "property test; endproperty"
        cleaned = runner._clean_sva_output(response)
        assert cleaned == response

    def test_generate_sva(self, runner, mock_llm_client):
        """Test SVA generation."""
        svad = "Request should be acknowledged within 3 cycles"
        result = runner.generate_sva(mock_llm_client, svad)
        assert "property" in result
        mock_llm_client.generate.assert_called_once()

    def test_evaluate_relationship_equivalent(self, runner, mock_checker):
        """Test relationship evaluation for equivalent SVAs."""
        mock_checker.get_implication_relationship.return_value = (True, True)
        result = runner.evaluate_relationship("sva1", "sva2")
        assert result == RelationshipType.EQUIVALENT

    def test_evaluate_relationship_generated_implies(self, runner, mock_checker):
        """Test relationship when generated implies reference."""
        mock_checker.get_implication_relationship.return_value = (True, False)
        result = runner.evaluate_relationship("sva1", "sva2")
        assert result == RelationshipType.GENERATED_IMPLIES_REFERENCE

    def test_evaluate_relationship_reference_implies(self, runner, mock_checker):
        """Test relationship when reference implies generated."""
        mock_checker.get_implication_relationship.return_value = (False, True)
        result = runner.evaluate_relationship("sva1", "sva2")
        assert result == RelationshipType.REFERENCE_IMPLIES_GENERATED

    def test_evaluate_relationship_no_relationship(self, runner, mock_checker):
        """Test relationship when no implication exists."""
        mock_checker.get_implication_relationship.return_value = (False, False)
        result = runner.evaluate_relationship("sva1", "sva2")
        assert result == RelationshipType.NO_RELATIONSHIP

    def test_run_single(self, runner, mock_llm_client, mock_checker):
        """Test running benchmark on single item."""
        result = runner.run_single(
            mock_llm_client,
            "Request acknowledged",
            "req |-> ack"
        )
        assert isinstance(result, SingleResult)
        assert result.relationship == RelationshipType.EQUIVALENT

    def test_run_benchmark(self, runner, mock_llm_client):
        """Test running benchmark on dataset."""
        dataset = [
            {"SVAD": "Test description 1", "SVA": "req |-> ack"},
            {"SVAD": "Test description 2", "SVA": "valid |-> ready"},
        ]
        result = runner.run_benchmark(dataset, mock_llm_client, rate_limit_delay=0)
        assert isinstance(result, BenchmarkResult)
        assert result.total_items == 2
        assert result.model_name == "test-model"

    def test_run_benchmark_skips_invalid(self, runner, mock_llm_client):
        """Test that benchmark skips invalid items."""
        dataset = [
            {"SVAD": "Valid", "SVA": "req |-> ack"},
            {"SVAD": "", "SVA": "missing svad"},  # Missing SVAD
            {"SVAD": "missing sva", "SVA": ""},  # Missing SVA
        ]
        result = runner.run_benchmark(dataset, mock_llm_client, rate_limit_delay=0)
        assert result.total_items == 1  # Only 1 valid item

    def test_compare_results(self):
        """Test result comparison."""
        results = [
            BenchmarkResult(
                model_name="model-a",
                total_items=10,
                equivalent_count=5,
            ),
            BenchmarkResult(
                model_name="model-b",
                total_items=10,
                equivalent_count=7,
            ),
        ]
        comparison = BenchmarkRunner.compare_results(results)
        assert comparison["best_equivalent_model"] == "model-b"
        assert comparison["best_equivalent_rate"] == 0.7


class TestBenchmarkRunnerFromConfigs:
    """Tests for creating BenchmarkRunner from configs."""

    @patch('sva_toolkit.benchmark.runner.LLMClient')
    @patch('sva_toolkit.benchmark.runner.SVAImplicationChecker')
    def test_from_configs(self, mock_checker_cls, mock_client_cls):
        """Test creating runner from config dictionaries."""
        configs = [
            {"base_url": "http://test1", "model": "model1", "api_key": "key1"},
            {"base_url": "http://test2", "model": "model2", "api_key": "key2"},
        ]
        
        mock_client_cls.from_params.return_value = Mock()
        mock_checker_cls.return_value = Mock()
        
        runner = BenchmarkRunner.from_configs(configs)
        
        assert mock_client_cls.from_params.call_count == 2
        assert len(runner.llm_clients) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
