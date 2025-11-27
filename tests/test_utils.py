"""Tests for utility modules."""

import pytest
import tempfile
import os
from pathlib import Path

from sva_toolkit.utils.common import (
    clean_sva_code,
    extract_sva_from_markdown,
    wrap_sva_in_module,
    extract_signals_from_expression,
    validate_sva_syntax,
    format_sva_code,
    parse_delay_spec,
    generate_signal_declaration,
    merge_datasets,
    split_dataset,
)
from sva_toolkit.utils.file_handlers import (
    load_json,
    save_json,
    ensure_directory,
    create_sby_config,
)


class TestCommonUtils:
    """Tests for common utility functions."""

    def test_clean_sva_code(self):
        """Test SVA code cleaning."""
        code = "  property  p;  a |->  b;  endproperty  "
        cleaned = clean_sva_code(code)
        assert "  " not in cleaned  # No double spaces
        assert cleaned.startswith("property")
        assert cleaned.endswith("endproperty")

    def test_extract_sva_from_markdown(self):
        """Test extraction from markdown."""
        markdown = """
Here is some SVA code:
```systemverilog
property test;
    a |-> b;
endproperty
```
"""
        extracted = extract_sva_from_markdown(markdown)
        assert extracted is not None
        assert "property test" in extracted

    def test_extract_sva_from_markdown_no_code(self):
        """Test extraction when no code block present."""
        text = "This is just regular text without code."
        extracted = extract_sva_from_markdown(text)
        assert extracted is None

    def test_wrap_sva_in_module(self):
        """Test module wrapping."""
        sva = "property p; a |-> b; endproperty"
        wrapped = wrap_sva_in_module(sva, "test_module")
        assert "module test_module" in wrapped
        assert "endmodule" in wrapped

    def test_wrap_sva_already_has_module(self):
        """Test wrapping when already has module."""
        sva = "module existing; property p; endproperty endmodule"
        wrapped = wrap_sva_in_module(sva)
        assert wrapped == sva  # Should not double-wrap

    def test_extract_signals_from_expression(self):
        """Test signal extraction."""
        expr = "valid && ready && (data_out == expected)"
        signals = extract_signals_from_expression(expr)
        assert "valid" in signals
        assert "ready" in signals
        assert "data_out" in signals
        assert "expected" in signals

    def test_extract_signals_excludes_keywords(self):
        """Test that keywords are excluded."""
        expr = "property and or not sequence"
        signals = extract_signals_from_expression(expr)
        assert "property" not in [s.lower() for s in signals]
        assert "and" not in [s.lower() for s in signals]

    def test_validate_sva_syntax_valid(self):
        """Test syntax validation with valid code."""
        code = "property p; a |-> b; endproperty"
        is_valid, error = validate_sva_syntax(code)
        assert is_valid
        assert error is None

    def test_validate_sva_syntax_unbalanced_parens(self):
        """Test syntax validation with unbalanced parens."""
        code = "property p; (a |-> b; endproperty"
        is_valid, error = validate_sva_syntax(code)
        assert not is_valid
        assert "parentheses" in error.lower()

    def test_validate_sva_syntax_mismatched_property(self):
        """Test syntax validation with mismatched property/endproperty."""
        code = "property p; a |-> b;"
        is_valid, error = validate_sva_syntax(code)
        assert not is_valid
        assert "property" in error.lower()

    def test_format_sva_code(self):
        """Test code formatting."""
        code = "property p;\na |-> b;\nendproperty"
        formatted = format_sva_code(code)
        assert "property p;" in formatted
        assert "endproperty" in formatted

    def test_parse_delay_spec_exact(self):
        """Test parsing exact delay."""
        result = parse_delay_spec("##3")
        assert result['type'] == 'exact'
        assert result['min'] == 3
        assert result['max'] == 3

    def test_parse_delay_spec_range(self):
        """Test parsing range delay."""
        result = parse_delay_spec("##[1:5]")
        assert result['type'] == 'range'
        assert result['min'] == 1
        assert result['max'] == 5

    def test_parse_delay_spec_unbounded(self):
        """Test parsing unbounded delay."""
        result = parse_delay_spec("##[1:$]")
        assert result['type'] == 'range'
        assert result['min'] == 1
        assert result['unbounded'] is True

    def test_generate_signal_declaration(self):
        """Test signal declaration generation."""
        decl = generate_signal_declaration("data", width=8)
        assert "data" in decl
        assert "[7:0]" in decl

    def test_generate_signal_declaration_single_bit(self):
        """Test single bit signal declaration."""
        decl = generate_signal_declaration("valid", width=1)
        assert "valid" in decl
        assert "[" not in decl  # No width spec for single bit

    def test_merge_datasets(self):
        """Test dataset merging."""
        ds1 = [{"SVA": "a"}, {"SVA": "b"}]
        ds2 = [{"SVA": "b"}, {"SVA": "c"}]  # 'b' is duplicate
        merged = merge_datasets([ds1, ds2])
        assert len(merged) == 3  # a, b, c (b deduplicated)

    def test_split_dataset(self):
        """Test dataset splitting."""
        data = [{"SVA": str(i)} for i in range(100)]
        train, val, test = split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


class TestFileHandlers:
    """Tests for file handling utilities."""

    def test_save_and_load_json(self):
        """Test JSON save and load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            data = {"key": "value", "list": [1, 2, 3]}
            save_json(data, filepath)
            loaded = load_json(filepath)
            assert loaded == data
        finally:
            os.unlink(filepath)

    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            result = ensure_directory(new_dir)
            assert result.exists()
            assert result.is_dir()

    def test_create_sby_config(self):
        """Test SBY config generation."""
        config = create_sby_config(
            module_file="test.sv",
            module_name="test_module",
            mode="prove",
            depth=30
        )
        assert "[options]" in config
        assert "mode prove" in config
        assert "depth 30" in config
        assert "test.sv" in config
        assert "test_module" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
