from __future__ import annotations

import shutil

from sva_toolkit.runtime.config import ToolConfig


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolConfig] = {}

    def register(
        self,
        name: str,
        search_names: list[str] | tuple[str, ...] | None = None,
    ) -> ToolConfig:
        candidates = search_names or [name]
        path = None
        for candidate in candidates:
            path = shutil.which(candidate)
            if path is not None:
                break

        config = ToolConfig(
            name=name,
            path=path,
            available=path is not None,
        )
        self._tools[name] = config
        return config

    def get(self, name: str) -> ToolConfig:
        return self._tools[name]

    def require(self, name: str) -> ToolConfig:
        config = self.get(name)
        if not config.available:
            raise RuntimeError(f"Required tool '{name}' is not available")
        return config

    @property
    def available_tools(self) -> dict[str, ToolConfig]:
        return {name: config for name, config in self._tools.items() if config.available}


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("ebmc")
    registry.register("vcf")
    registry.register("verible-verilog-syntax")
    return registry
