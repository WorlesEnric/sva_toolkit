from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path

from sva_toolkit.runtime.config import ToolConfig, ToolkitConfig


def test_toolkit_config_is_dataclass_with_v2_defaults() -> None:
    config = ToolkitConfig()

    assert is_dataclass(config) is True
    assert config.work_dir is None
    assert config.keep_files is False
    assert config.verbose is False
    assert config.timeout == 300


def test_toolkit_config_accepts_explicit_values() -> None:
    config = ToolkitConfig(
        work_dir=Path("/tmp/sva-toolkit"),
        keep_files=True,
        verbose=True,
        timeout=30,
    )

    assert config.work_dir == Path("/tmp/sva-toolkit")
    assert config.keep_files is True
    assert config.verbose is True
    assert config.timeout == 30


def test_tool_config_is_dataclass_with_runtime_defaults() -> None:
    config = ToolConfig(name="ebmc")

    assert is_dataclass(config) is True
    assert config.name == "ebmc"
    assert config.path is None
    assert config.available is False
    assert config.version is None
