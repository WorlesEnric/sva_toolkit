"""Emit SVA from symbolic timing scenarios."""

from __future__ import annotations

import re
from typing import Dict, Optional

from sva_toolkit.sva.emitter import emit_expr, emit_property_body, emit_sequence
from sva_toolkit.sva.lowerings.conditions import condition_to_expr
from sva_toolkit.timing.core.conditions import condition_to_sva
from sva_toolkit.timing.core.scenario import (
    AnchorRole,
    ConstraintRegion,
    ExtractionStatus,
    LaneConstraint,
    PropertyOverlay,
    ScenarioDocument,
    TimeWindow,
    WindowBoundKind,
)
from sva_toolkit.timing.core.graph import CanonicalEventGraph, CEGNode, TimeRange
from sva_toolkit.sva.ast import (
    ExprSequence,
    Literal,
    SequenceNode,
)


class DiagramCompiler:
    """Compiles a ScenarioDocument (Timing Diagram) into a Canonical Event Graph (CEG)."""

    def __init__(self, diagram: ScenarioDocument) -> None:
        self.diagram = diagram
        self.graph = diagram.ceg if isinstance(diagram.ceg, CanonicalEventGraph) else CanonicalEventGraph()

    def compile(self) -> CanonicalEventGraph:
        if self.diagram.ceg:
            return self.graph

        # Build from scratch: Map anchors to nodes
        anchor_to_node: Dict[str, CEGNode] = {}
        
        # Pass 1: Create initial nodes for each anchor
        for anchor in self.diagram.anchors:
            node = self.graph.create_node(condition=condition_to_expr(anchor.condition))
            node.is_trigger = (anchor.role == AnchorRole.TRIGGER)
            anchor_to_node[anchor.name] = node

        # Pass 2: Process windows
        for window in self.diagram.windows:
            src = anchor_to_node.get(window.start_anchor)
            dst = anchor_to_node.get(window.end_anchor)
            if src and dst:
                delay = TimeRange(
                    min_delay=window.bound.min_delay or "0",
                    max_delay=window.bound.max_delay,
                    unbounded=(window.bound.kind == WindowBoundKind.UNBOUNDED)
                )
                self.graph.add_edge(src, dst, delay=delay)

        return self.graph


class SvaSerializer:
    """Serializes a Canonical Event Graph (CEG) into SVA code."""

    def __init__(self, graph: CanonicalEventGraph) -> None:
        self.graph = self._prune_graph(graph)

    def _prune_graph(self, graph: CanonicalEventGraph) -> CanonicalEventGraph:
        """Remove all redundant structural nodes, keeping only real conditions or terminals."""
        pruned = CanonicalEventGraph()
        node_map: Dict[int, CEGNode] = {}
        
        # Pass 1: Identify nodes that MUST exist in the final SVA
        for node in graph.nodes.values():
            is_root = not graph.get_in_edges(node.id)
            is_terminal = not graph.get_out_edges(node.id)
            has_condition = (node.condition and not _is_trivial_true_expr(node.condition))
            
            if is_root or is_terminal or has_condition:
                new_node = pruned.create_node(condition=node.condition, clocking=node.clocking)
                new_node.is_trigger = node.is_trigger
                new_node.is_terminal = node.is_terminal
                new_node.assignments = node.assignments
                node_map[node.id] = new_node

        # Pass 2: Re-connect through structural nodes
        for start_id, start_node in node_map.items():
            worklist = [(e.target_id, e.delay) for e in graph.get_out_edges(start_id)]
            while worklist:
                curr_id, curr_delay = worklist.pop(0)
                if curr_id in node_map:
                    pruned.add_edge(start_node, node_map[curr_id], delay=curr_delay)
                else:
                    for e in graph.get_out_edges(curr_id):
                        worklist.append((e.target_id, _merge_delays(curr_delay, e.delay)))
        
        return pruned

    def serialize(self, mode: str = "ROOT_TRIGGER") -> str:
        roots = self.graph.get_roots()
        if not roots:
            return ""

        results = []
        for root in roots:
            if mode == "ROOT_TRIGGER":
                antecedent = self._serialize_node(root)
                out_edges = self.graph.get_out_edges(root.id)
                if not out_edges:
                    results.append(emit_sequence(antecedent))
                    continue
                
                # Use overlapped implication |-> and include full delay in consequent for correctness
                consequents = []
                for edge in out_edges:
                    target_id = edge.target_id
                    target_node = self.graph.nodes[target_id]
                    edge_delay = edge.delay
                    
                    if _is_trivial_true_expr(target_node.condition) and not target_node.is_terminal:
                        t_out = self.graph.get_out_edges(target_id)
                        if t_out:
                            target_id = t_out[0].target_id
                            edge_delay = _merge_delays(edge_delay, t_out[0].delay)

                    branch = self._serialize_path(target_id, initial_delay=edge_delay)
                    consequents.append(branch)
                
                if len(consequents) > 1:
                    results.append(f"{emit_sequence(antecedent)} |-> ({' and '.join(consequents)})")
                else:
                    results.append(f"{emit_sequence(antecedent)} |-> {consequents[0]}")
            else:
                results.append(self._serialize_path(root.id))
        
        return " and ".join(results)

    def _serialize_node(self, node: CEGNode) -> SequenceNode:
        if node.condition:
            return ExprSequence(expr=node.condition)
        return ExprSequence(expr=Literal(text="1'b1"))

    def _serialize_path(self, node_id: int, initial_delay: Optional[TimeRange] = None) -> str:
        segments = []
        curr_id = node_id
        pending_delay = initial_delay
        visited = set()

        while curr_id not in visited:
            visited.add(curr_id)
            node = self.graph.nodes[curr_id]
            cond = emit_expr(node.condition) if node.condition and not _is_trivial_true_expr(node.condition) else None
            
            if cond:
                segments.append((pending_delay, cond))
                pending_delay = None
            elif not self.graph.get_out_edges(curr_id):
                if not segments or pending_delay:
                    segments.append((pending_delay, "1'b1"))
                pending_delay = None
            
            out_edges = self.graph.get_out_edges(curr_id)
            if not out_edges:
                break
            
            edge = out_edges[0]
            if edge.delay:
                pending_delay = _merge_delays(pending_delay, edge.delay) if pending_delay else edge.delay
            curr_id = edge.target_id

        if not segments:
            return "1'b1"
        
        parts = []
        for delay, cond in segments:
            if delay:
                parts.append(f"##{_delay_to_sva(delay)}")
                parts.append(cond)
            else:
                if parts:
                    parts.append("##0")
                parts.append(cond)
        
        raw = " ".join(parts)
        cleaned = raw.replace("1'b1 ##0 ", "").replace(" ##0 1'b1", "").replace("1'b1 ##0", "")
        return cleaned if cleaned.strip() else "1'b1"


def _delay_to_sva(delay: TimeRange) -> str:
    if delay.unbounded:
        return f"[{delay.min_delay}:$]"
    if delay.is_exact():
        return delay.min_delay
    return f"[{delay.min_delay}:{delay.max_delay}]"


def _merge_delays(d1: Optional[TimeRange], d2: TimeRange) -> TimeRange:
    if not d1:
        return d2
    try:
        m1 = int(d1.min_delay)
        m2 = int(d2.min_delay)
        new_min = str(m1 + m2)
        new_max = None
        if d1.max_delay and d2.max_delay:
            new_max = str(int(d1.max_delay) + int(d2.max_delay))
        return TimeRange(min_delay=new_min, max_delay=new_max, unbounded=d1.unbounded or d2.unbounded)
    except (ValueError, TypeError):
        return d1


def _is_trivial_true_expr(node) -> bool:
    if node is None:
        return True
    if isinstance(node, Literal) and node.text in ("1", "1'b1"):
        return True
    return False


def synthesize_sva_from_diagram(diagram: ScenarioDocument) -> str:
    """Automatically synthesize SVA properties from the diagram's temporal structure."""
    compiler = DiagramCompiler(diagram)
    graph = compiler.compile()
    serializer = SvaSerializer(graph)
    
    blocks = []
    emitted_bodies = set()
    sva_body = serializer.serialize(mode="ROOT_TRIGGER")
    if sva_body:
        blocks.append(_wrap_in_property(diagram.name, sva_body, diagram))
        emitted_bodies.add(sva_body)
        
    sva_seq = serializer.serialize(mode="SEQUENTIAL_ONLY")
    if sva_seq:
        blocks.append(f"cover property (@({diagram.clocking.edge} {diagram.clocking.signal}) {sva_seq});")

    anchor_map = diagram.anchor_map
    for constraint in diagram.lane_constraints:
        overlay = _derive_constraint_property(constraint)
        if overlay is None:
            continue
        body, _ = _render_property_body(overlay, diagram, anchor_map)
        if body in emitted_bodies:
            continue
        blocks.append(_wrap_in_property(overlay.name, body, diagram))
        emitted_bodies.add(body)
        
    return "\n\n".join(blocks)


def _wrap_in_property(name: str, body: str, diagram: ScenarioDocument) -> str:
    clocking = f"@({diagram.clocking.edge} {diagram.clocking.signal})"
    if diagram.clocking.disable_iff:
        clocking += f" disable iff ({diagram.clocking.disable_iff})"
    return f"property {name};\n  {clocking}\n    {body};\nendproperty"


def emit_parameterized_sva(diagram: ScenarioDocument, *, allow_lossy: bool = False) -> str:
    """Emit SVA from preserved extracted properties or synthesized diagram structure."""
    if _should_preserve_extracted_property(diagram):
        blocks = _emit_preserved_property_blocks(diagram.properties, diagram, allow_lossy=allow_lossy)
        if blocks:
            return "\n\n".join(blocks)

    properties = _derive_properties_from_diagram(diagram)
    if properties:
        blocks = _emit_property_blocks(properties, diagram, allow_lossy=allow_lossy)
        if blocks:
            return "\n\n".join(blocks)

    # Fallback to pure synthesis if no properties are defined
    return synthesize_sva_from_diagram(diagram)


def _should_preserve_extracted_property(diagram: ScenarioDocument) -> bool:
    if len(diagram.properties) != 1:
        return False

    prop = diagram.properties[0]
    if prop.status != ExtractionStatus.EXACT:
        return False
    return prop.body_ast is not None or prop.source is not None


def _emit_preserved_property_blocks(
    properties: tuple[PropertyOverlay, ...] | tuple[PropertyOverlay],
    diagram: ScenarioDocument,
    *,
    allow_lossy: bool,
) -> list[str]:
    blocks = []
    for prop in properties:
        if prop.status == ExtractionStatus.UNSUPPORTED:
            continue
        if prop.status == ExtractionStatus.LOSSY and not allow_lossy:
            continue

        body = emit_property_body(prop.body_ast) if prop.body_ast is not None else prop.body
        used_params = tuple(param for param in diagram.params if re.search(rf"\b{param.name}\b", body))
        params_str = ", ".join(f"{param.kind} {param.name}" for param in used_params)
        header = f"property {prop.name}({params_str});" if params_str else f"property {prop.name};"
        if prop.status == ExtractionStatus.LOSSY:
            header = f"// lossy lowering\n{header}"

        blocks.append(f"{header}\n  {_property_clocking(diagram)}\n    {body};\nendproperty")
    return blocks


def _emit_property_blocks(
    properties: tuple[PropertyOverlay, ...] | tuple[PropertyOverlay],
    diagram: ScenarioDocument,
    *,
    allow_lossy: bool,
) -> list[str]:
    blocks = []
    anchor_map = diagram.anchor_map
    for prop in properties:
        if prop.status == ExtractionStatus.UNSUPPORTED:
            continue
        if prop.status == ExtractionStatus.LOSSY and not allow_lossy:
            continue

        body, used_params = _render_property_body(prop, diagram, anchor_map)
        params_str = ", ".join(f"{param.kind} {param.name}" for param in used_params)
        header = f"property {prop.name}({params_str});" if params_str else f"property {prop.name};"
        if prop.status == ExtractionStatus.LOSSY:
            header = f"// lossy lowering\n{header}"

        blocks.append(f"{header}\n  {_property_clocking(diagram)}\n    {body};\nendproperty")
    return blocks


def _property_clocking(diagram: ScenarioDocument) -> str:
    clocking = f"@({diagram.clocking.edge} {diagram.clocking.signal})"
    disable_iff = diagram.clocking.disable_iff
    if diagram.clocking.disable_iff_ast is not None:
        disable_iff = emit_expr(diagram.clocking.disable_iff_ast)
    if disable_iff:
        clocking += f" disable iff ({disable_iff})"
    return clocking


def _derive_properties_from_diagram(diagram: ScenarioDocument) -> tuple[PropertyOverlay, ...]:
    derived: list[PropertyOverlay] = []
    anchor_map = diagram.anchor_map
    trivial_map = _build_effective_trigger_map(diagram)
    for window in diagram.windows:
        overlay = _derive_window_property(window, anchor_map, trivial_map)
        if overlay is not None:
            derived.append(overlay)
    for constraint in diagram.lane_constraints:
        overlay = _derive_constraint_property(constraint)
        if overlay is not None:
            derived.append(overlay)
    return tuple(derived)


def _derive_window_property(window: TimeWindow, anchor_map=None, trivial_map=None) -> PropertyOverlay | None:
    # Skip windows whose consequent is trivially true — they generate useless `X |-> 1'b1` properties
    if anchor_map and _is_trivial_anchor_condition(window.end_anchor, anchor_map):
        return None
    consequent = _window_consequent(window)
    if consequent is None:
        return None
    # Resolve start anchor: if it is trivially 1'b1, use the effective non-trivial ancestor
    start_anchor = window.start_anchor
    if trivial_map and start_anchor in trivial_map:
        start_anchor = trivial_map[start_anchor]
    return PropertyOverlay(
        name=window.name,
        body=f"{start_anchor} |-> {consequent}",
        related_anchors=(start_anchor, window.end_anchor),
        related_windows=(window.name,),
    )


def _is_trivial_anchor_condition(anchor_name: str, anchor_map: dict) -> bool:
    """Return True if the named anchor's condition is a constant-true expression (1'b1 / 1)."""
    anchor = anchor_map.get(anchor_name)
    if anchor is None:
        return False
    cond = anchor.condition
    if cond is None:
        return True
    return cond.kind == "raw" and cond.text in ("1'b1", "1", "")


def _build_effective_trigger_map(diagram: ScenarioDocument) -> dict[str, str]:
    """Return a map from trivial (1'b1) anchor names to their nearest non-trivial ancestor.

    Traces backward through consecutive zero-delay windows whose start anchor is also
    trivial, stopping as soon as a real signal condition is found.
    """
    anchor_map = diagram.anchor_map
    # end_anchor_name -> window (last one wins if duplicates, but anchor names should be unique)
    end_to_window: dict[str, TimeWindow] = {}
    for w in diagram.windows:
        end_to_window[w.end_anchor] = w

    def is_zero_delay(w: TimeWindow) -> bool:
        b = w.bound
        return (
            b.kind == WindowBoundKind.EXACT
            and (b.min_delay or "0") == "0"
            and (b.max_delay is None or b.max_delay == "0")
        )

    result: dict[str, str] = {}
    for anchor_name in list(anchor_map.keys()):
        if not _is_trivial_anchor_condition(anchor_name, anchor_map):
            continue
        current = anchor_name
        visited: set[str] = {current}
        while True:
            w = end_to_window.get(current)
            if w is None or not is_zero_delay(w):
                break
            start = w.start_anchor
            if start in visited:
                break
            visited.add(start)
            current = start
            if not _is_trivial_anchor_condition(current, anchor_map):
                break
        if current != anchor_name:
            result[anchor_name] = current
    return result


def _window_consequent(window: TimeWindow) -> str | None:
    bound = window.bound
    if bound.kind == WindowBoundKind.OMITTED:
        return None
    if bound.kind == WindowBoundKind.EXACT and (bound.min_delay or "0") == "0":
        return window.end_anchor
    delay = _bound_to_delay_token(bound)
    if delay is None:
        return None
    return f"##{delay} {window.end_anchor}"


def _bound_to_delay_token(bound) -> str | None:
    if bound.kind == WindowBoundKind.EXACT:
        return bound.min_delay or "0"
    if bound.kind == WindowBoundKind.RANGE:
        return f"[{bound.min_delay}:{bound.max_delay}]"
    if bound.kind == WindowBoundKind.UNBOUNDED:
        return f"[{bound.min_delay}:$]"
    return None


def _derive_constraint_property(constraint: LaneConstraint) -> PropertyOverlay | None:
    expr = _constraint_expr_to_sva(constraint)
    if not expr:
        return None

    body = None
    if constraint.region == ConstraintRegion.FROM_UNTIL and constraint.start_anchor and constraint.end_anchor:
        body = f"{constraint.start_anchor} |-> {_wrap(expr)} until_with {constraint.end_anchor}"
    elif constraint.region == ConstraintRegion.BEFORE and constraint.anchor:
        body = f"{_wrap(expr)} until {constraint.anchor}"
    elif constraint.region == ConstraintRegion.AT and constraint.anchor:
        body = f"{constraint.anchor} |-> {_wrap(expr)}"

    if body is None:
        return None

    related_anchors = tuple(
        anchor_name
        for anchor_name in (constraint.anchor, constraint.start_anchor, constraint.end_anchor)
        if anchor_name is not None
    )
    # Build a human-readable property name from the constraint semantics
    signal_part = "_".join(constraint.signals) if constraint.signals else "sig"
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        region_part = f"from_{constraint.start_anchor}_until_{constraint.end_anchor}"
    elif constraint.region == ConstraintRegion.BEFORE:
        region_part = f"before_{constraint.anchor}"
    elif constraint.region == ConstraintRegion.AT:
        region_part = f"at_{constraint.anchor}"
    else:
        region_part = constraint.name
    prop_name = f"{constraint.relation}_{signal_part}_{region_part}"

    return PropertyOverlay(
        name=prop_name,
        body=body,
        related_anchors=related_anchors,
        related_constraints=(constraint.name,),
    )


def _constraint_expr_to_sva(constraint: LaneConstraint) -> str:
    if constraint.relation == "raw":
        return constraint.value or ""

    parts = []
    for signal in constraint.signals:
        if constraint.relation == "high":
            parts.append(signal)
        elif constraint.relation == "low":
            parts.append(f"!{signal}")
        elif constraint.relation == "rise":
            parts.append(f"$rose({signal})")
        elif constraint.relation == "fall":
            parts.append(f"$fell({signal})")
        elif constraint.relation == "stable":
            parts.append(f"$stable({signal})")
        elif constraint.relation == "change":
            parts.append(f"$changed({signal})")
        elif constraint.relation == "unknown":
            parts.append(f"$isunknown({signal})")
        elif constraint.relation == "dontcare":
            parts.append(signal)
        elif constraint.relation == "eq" and constraint.value is not None:
            parts.append(f"({signal} == {constraint.value})")
        elif constraint.relation == "neq" and constraint.value is not None:
            parts.append(f"({signal} != {constraint.value})")
    return " && ".join(parts)


def _render_property_body(prop, diagram: ScenarioDocument, anchor_map: dict[str, object]) -> tuple[str, tuple[object, ...]]:
    body = _expand_anchor_references(prop.body, anchor_map)
    used_params = tuple(param for param in diagram.params if re.search(rf"\b{param.name}\b", body))
    return body, used_params


def _expand_anchor_references(body: str, anchor_map: dict[str, object]) -> str:
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
