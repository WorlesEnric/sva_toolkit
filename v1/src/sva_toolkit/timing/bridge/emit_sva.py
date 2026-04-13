"""Emit SVA from concrete or symbolic timing scenarios."""

from __future__ import annotations

import re

from sva_toolkit.timing.core.conditions import condition_to_sva
from sva_toolkit.timing.core.model import DiagramSpec
from sva_toolkit.timing.core.scenario import ExtractionStatus, ScenarioDocument
from sva_toolkit.timing.errors import TimingDslError
from sva_toolkit.timing.projection.assertion_view import build_assertion_view


def emit_parameterized_sva(diagram: DiagramSpec | ScenarioDocument, *, allow_lossy: bool = False) -> str:
    """Emit parameterized SVA properties for supported exact timing scenarios."""

    if isinstance(diagram, DiagramSpec):
        return _emit_from_legacy(diagram)

    if diagram.legacy_diagram is not None and isinstance(diagram.legacy_diagram, DiagramSpec):
        return _emit_from_legacy(diagram.legacy_diagram)

    if not diagram.properties:
        raise TimingDslError("scenario does not define any properties to emit")

    blocks = []
    anchor_map = diagram.anchor_map
    for prop in diagram.properties:
        if prop.status == ExtractionStatus.UNSUPPORTED:
            raise TimingDslError(f"property '{prop.name}' is unsupported and cannot be lowered back to SVA")
        if prop.status == ExtractionStatus.LOSSY and not allow_lossy:
            raise TimingDslError(
                f"property '{prop.name}' is lossy and cannot be lowered back to SVA without allow_lossy=True"
            )

        body = _expand_anchor_references(prop.body, anchor_map)
        used_params = tuple(param for param in diagram.params if re.search(rf"\b{param.name}\b", body))
        params = ", ".join(f"{param.kind} {param.name}" for param in used_params)
        header = f"property {prop.name}({params});" if params else f"property {prop.name};"
        if prop.status == ExtractionStatus.LOSSY:
            header = f"// lossy lowering\n{header}"

        clocking = f"@({diagram.clocking.edge} {diagram.clocking.signal})"
        if diagram.clocking.disable_iff:
            clocking += f" disable iff ({diagram.clocking.disable_iff})"

        blocks.append(
            "\n".join(
                [
                    header,
                    f"  {clocking}",
                    f"    {body};",
                    "endproperty",
                ]
            )
        )
    return "\n\n".join(blocks)


def _emit_from_legacy(diagram: DiagramSpec) -> str:
    properties = build_assertion_view(diagram)
    blocks = []
    for prop in properties:
        params = ", ".join(f"{param.kind} {param.name}" for param in prop.params)
        header = f"property {prop.name}({params});" if params else f"property {prop.name};"
        clocking = f"@({diagram.clocking.edge} {diagram.clocking.signal})"
        if diagram.clocking.disable_iff:
            clocking += f" disable iff ({diagram.clocking.disable_iff})"
        blocks.append(
            "\n".join(
                [
                    header,
                    f"  {clocking}",
                    f"    {prop.body};",
                    "endproperty",
                ]
            )
        )
    return "\n\n".join(blocks)


def _expand_anchor_references(body: str, anchor_map) -> str:
    expanded = body
    for anchor_name, anchor in sorted(anchor_map.items(), key=lambda item: len(item[0]), reverse=True):
        rendered = condition_to_sva(anchor.condition)
        if not rendered:
            continue
        expanded = re.sub(rf"\b{anchor_name}\b", _wrap(rendered), expanded)
    return expanded


def _wrap(text: str) -> str:
    if " && " in text or " || " in text or " until" in text:
        return f"({text})"
    return text
