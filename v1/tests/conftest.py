"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_sva_code():
    """Sample SVA code for testing."""
    return """
property ReqAckHandshake;
    @(posedge clk) disable iff (!rst_n)
    req |-> ##[1:3] ack;
endproperty
assert property (ReqAckHandshake);
"""


@pytest.fixture
def sample_sva_simple():
    """Simple SVA property for testing."""
    return "property p; @(posedge clk) a |-> b; endproperty"


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return [
        {
            "SVA": "property p1; @(posedge clk) req |-> ##[1:3] ack; endproperty",
            "SVAD": "When request is asserted, acknowledgment must follow within 1 to 3 cycles."
        },
        {
            "SVA": "property p2; @(posedge clk) valid |-> $stable(data); endproperty",
            "SVAD": "When valid is high, data must remain stable."
        },
    ]


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for tool calls."""
    with pytest.mock.patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr=""
        )
        yield mock_run


@pytest.fixture
def mock_llm_response():
    """Mock LLM response generator."""
    def _generate(response_text):
        mock = Mock()
        mock.choices = [Mock()]
        mock.choices[0].message.content = response_text
        return mock
    return _generate


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_verible: mark test as requiring Verible installation"
    )
    config.addinivalue_line(
        "markers", "requires_sby: mark test as requiring SymbiYosys installation"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available tools."""
    import shutil
    
    skip_verible = pytest.mark.skip(reason="Verible not installed")
    skip_sby = pytest.mark.skip(reason="SymbiYosys not installed")
    
    for item in items:
        if "requires_verible" in item.keywords:
            if not shutil.which("verible-verilog-syntax"):
                item.add_marker(skip_verible)
        if "requires_sby" in item.keywords:
            if not shutil.which("sby"):
                item.add_marker(skip_sby)
