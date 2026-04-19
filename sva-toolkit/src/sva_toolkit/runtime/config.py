from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolkitConfig:
    work_dir: Path | None = None
    keep_files: bool = False
    verbose: bool = False
    timeout: int = 300


@dataclass
class ToolConfig:
    name: str
    path: str | None = None
    available: bool = False
    version: str | None = None
