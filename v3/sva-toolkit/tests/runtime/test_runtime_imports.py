from __future__ import annotations

import importlib
import shutil
import sys


def test_importing_tools_module_does_not_probe_path(monkeypatch) -> None:
    calls: list[str] = []

    def fake_which(command: str) -> None:
        calls.append(command)
        return None

    monkeypatch.setattr(shutil, "which", fake_which)
    sys.modules.pop("sva_toolkit.runtime.tools", None)

    importlib.import_module("sva_toolkit.runtime.tools")

    assert calls == []


def test_runtime_package_exports_public_api() -> None:
    runtime = importlib.import_module("sva_toolkit.runtime")

    assert hasattr(runtime, "ToolkitConfig")
    assert hasattr(runtime, "ToolConfig")
    assert hasattr(runtime, "ToolRegistry")
    assert hasattr(runtime, "create_default_registry")
    assert hasattr(runtime, "RunResult")
    assert hasattr(runtime, "run_tool")
    assert hasattr(runtime, "make_work_dir")
    assert hasattr(runtime, "LLMConfig")
    assert hasattr(runtime, "LLMClient")
