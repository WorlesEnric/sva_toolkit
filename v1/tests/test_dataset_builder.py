"""Tests for SVA Dataset Builder."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os

from sva_toolkit.dataset_builder import DatasetBuilder
from sva_toolkit.dataset_builder.builder import DatasetEntry
from sva_toolkit.utils.llm_client import LLMClient, LLMConfig


class TestDatasetEntry:
    """Tests for DatasetEntry class."""

    def test_entry_creation(self):
        """Test DatasetEntry creation."""
        entry = DatasetEntry(
            SVA="property p; a |-> b; endproperty",
            SVAD="When a is true, b must be true",
            CoT="## Step 1: ..."
        )
        assert entry.SVA == "property p; a |-> b; endproperty"
        assert entry.SVAD == "When a is true, b must be true"
        assert entry.CoT == "## Step 1: ..."

    def test_entry_to_dict(self):
        """Test DatasetEntry to_dict method."""
        entry = DatasetEntry(
            SVA="property p; endproperty",
            SVAD="Description",
            CoT="Chain of thought"
        )
        d = entry.to_dict()
        assert d["SVA"] == "property p; endproperty"
        assert d["SVAD"] == "Description"
        assert d["CoT"] == "Chain of thought"

    def test_entry_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        entry = DatasetEntry(SVA="property p; endproperty")
        d = entry.to_dict()
        assert "SVA" in d
        assert "SVAD" not in d  # None values excluded
        assert "CoT" not in d

    def test_entry_with_metadata(self):
        """Test entry with metadata."""
        entry = DatasetEntry(
            SVA="property p; endproperty",
            metadata={"source": "test", "difficulty": "easy"}
        )
        d = entry.to_dict()
        assert d["metadata"]["source"] == "test"


class TestDatasetBuilder:
    """Tests for DatasetBuilder class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock(spec=LLMClient)
        client.generate.return_value = "This property asserts that when request is high, acknowledgment follows."
        return client

    @pytest.fixture
    def builder(self, mock_llm_client):
        """Create a DatasetBuilder with mock LLM."""
        return DatasetBuilder(llm_client=mock_llm_client)

    @pytest.fixture
    def builder_no_llm(self):
        """Create a DatasetBuilder without LLM."""
        return DatasetBuilder()

    def test_generate_svad(self, builder, mock_llm_client):
        """Test SVAD generation."""
        sva = "property req_ack; req |-> ##[1:3] ack; endproperty"
        svad = builder.generate_svad(sva)
        assert svad is not None
        mock_llm_client.generate.assert_called_once()

    def test_generate_svad_no_llm(self, builder_no_llm):
        """Test SVAD generation without LLM raises error."""
        with pytest.raises(RuntimeError, match="LLM client not configured"):
            builder_no_llm.generate_svad("property p; endproperty")

    def test_generate_cot(self, builder):
        """Test CoT generation."""
        sva = "property req_ack; @(posedge clk) req |-> ##1 ack; endproperty"
        cot = builder.generate_cot(sva)
        assert "Step 1" in cot or "Interface" in cot
        assert "clk" in cot

    def test_process_entry_svad_and_cot(self, builder, mock_llm_client):
        """Test processing entry with both SVAD and CoT."""
        entry = DatasetEntry(SVA="property p; @(posedge clk) a |-> b; endproperty")
        processed = builder.process_entry(entry)
        assert processed.SVAD is not None
        assert processed.CoT is not None

    def test_process_entry_cot_only(self, builder):
        """Test processing entry with CoT only."""
        entry = DatasetEntry(SVA="property p; @(posedge clk) a |-> b; endproperty")
        processed = builder.process_entry(entry, generate_svad=False)
        assert processed.SVAD is None
        assert processed.CoT is not None

    def test_build_dataset(self, builder):
        """Test building dataset from input data."""
        input_data = [
            {"SVA": "property p1; a |-> b; endproperty"},
            {"SVA": "property p2; c |-> d; endproperty"},
        ]
        entries = builder.build_dataset(
            input_data,
            generate_svad=True,
            generate_cot=True,
            rate_limit_delay=0
        )
        assert len(entries) == 2
        assert all(e.SVAD is not None for e in entries)
        assert all(e.CoT is not None for e in entries)

    def test_build_dataset_cot_only(self, builder_no_llm):
        """Test building dataset with CoT only."""
        input_data = [
            {"SVA": "property p1; @(posedge clk) a |-> b; endproperty"},
        ]
        entries = builder_no_llm.build_dataset(
            input_data,
            generate_svad=False,
            generate_cot=True
        )
        assert len(entries) == 1
        assert entries[0].CoT is not None

    def test_build_dataset_skips_empty(self, builder_no_llm):
        """Test that empty SVA entries are skipped."""
        input_data = [
            {"SVA": "property p; endproperty"},
            {"SVA": ""},  # Empty
            {"other": "data"},  # Missing SVA
        ]
        entries = builder_no_llm.build_dataset(
            input_data,
            generate_svad=False,
            generate_cot=True
        )
        assert len(entries) == 1

    def test_build_from_file(self, builder, tmp_path):
        """Test building dataset from file."""
        input_file = tmp_path / "input.json"
        output_file = tmp_path / "output.json"
        
        input_data = [
            {"SVA": "property p; @(posedge clk) a |-> b; endproperty"}
        ]
        with open(input_file, 'w') as f:
            json.dump(input_data, f)
        
        entries = builder.build_from_file(
            str(input_file),
            str(output_file),
            rate_limit_delay=0
        )
        
        assert len(entries) == 1
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            output_data = json.load(f)
        assert len(output_data) == 1
        assert "SVAD" in output_data[0]

    def test_validate_dataset(self, builder):
        """Test dataset validation."""
        entries = [
            DatasetEntry(SVA="p1", SVAD="desc1", CoT="cot1"),
            DatasetEntry(SVA="p2", SVAD="desc2"),  # No CoT
            DatasetEntry(SVA="p3", CoT="cot3"),  # No SVAD
            DatasetEntry(SVA="p4", metadata={"svad_error": "Failed"}),  # Error
        ]
        report = builder.validate_dataset(entries)
        
        assert report["total_entries"] == 4
        assert report["entries_with_svad"] == 2
        assert report["entries_with_cot"] == 2
        assert report["entries_with_errors"] == 1
        assert report["svad_coverage"] == 0.5
        assert report["cot_coverage"] == 0.5

    def test_from_llm_config(self):
        """Test creating builder from LLM config."""
        with patch('sva_toolkit.dataset_builder.builder.LLMClient') as mock_cls:
            mock_cls.from_params.return_value = Mock()
            builder = DatasetBuilder.from_llm_config(
                base_url="http://test",
                model="test-model",
                api_key="test-key"
            )
            mock_cls.from_params.assert_called_once()
            assert builder.llm_client is not None

    def test_progress_callback(self, builder_no_llm):
        """Test progress callback is called."""
        input_data = [
            {"SVA": "property p1; endproperty"},
            {"SVA": "property p2; endproperty"},
            {"SVA": "property p3; endproperty"},
        ]
        
        progress_calls = []
        def callback(current, total):
            progress_calls.append((current, total))
        
        builder_no_llm.build_dataset(
            input_data,
            generate_svad=False,
            generate_cot=True,
            progress_callback=callback
        )
        
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
