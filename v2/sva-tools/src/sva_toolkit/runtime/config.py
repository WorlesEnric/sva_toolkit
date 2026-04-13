from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ToolkitConfig(BaseModel):
    work_dir: Path | None = None
    keep_files: bool = False
    verbose: bool = False
    timeout: int = 300


class ToolConfig(BaseModel):
    name: str
    path: str | None = None
    available: bool = False
    version: str | None = None
