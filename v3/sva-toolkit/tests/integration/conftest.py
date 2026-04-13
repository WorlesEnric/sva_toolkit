from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner


_PROPERTY_BLOCK_RE = re.compile(r"property\b.*?endproperty", re.DOTALL)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_sva() -> str:
    return "assert property (@(posedge clk) req |-> ##1 ack);"


@pytest.fixture
def timing_diagram_path(tmp_path: Path) -> Path:
    path = tmp_path / "handshake.td"
    path.write_text(
        """
        diagram hold_until_ready {
          clock posedge clk;
          param MAX_WAIT;

          lane valid: bit;
          lane ready: bit;
          lane data: bus[8];

          anchor asserted = rise(valid);
          anchor handshake = high(valid) and high(ready);

          window ready_window = between asserted and handshake [0:MAX_WAIT];

          show high(valid) from asserted until handshake;
          show stable(data) from asserted until handshake;
        }
        """,
        encoding="utf-8",
    )
    return path


def extract_property_blocks(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _PROPERTY_BLOCK_RE.finditer(text))
