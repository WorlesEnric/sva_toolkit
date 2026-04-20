from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pytest
from click.testing import CliRunner


_PROPERTY_BLOCK_RE = re.compile(r"property\b.*?endproperty", re.DOTALL)
_SVA_CORPUS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sva_corpus"


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


@pytest.fixture
def sva_corpus_dir() -> Path:
    return _SVA_CORPUS_DIR


@pytest.fixture
def sva_corpus_file(sva_corpus_dir: Path) -> Callable[[str], Path]:
    def _resolve(name: str) -> Path:
        return sva_corpus_dir / name

    return _resolve


@pytest.fixture
def long_property_text() -> str:
    repeated_tail = " ".join("##1 ack" for _ in range(400))
    return f"assert property (@(posedge clk) disable iff (!rst_n) req |-> {repeated_tail});"


def extract_property_blocks(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _PROPERTY_BLOCK_RE.finditer(text))
