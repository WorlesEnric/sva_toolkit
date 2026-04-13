"""Temporal alignment and DAG projection for timing scenarios."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from sva_toolkit.timing.core.graph import CanonicalEventGraph, CEGNode, CEGEdge, EdgeKind, TimeRange
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ExtractionStatus,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
    LaneConstraint,
    ConstraintRegion,
)
from sva_toolkit.timing.core.conditions import parse_sva_condition, collect_signals
from sva_toolkit.sva.emitter import emit_expr


class CEGSolver:
    """Solves temporal alignment and projects CEG to ScenarioDocument elements."""

    def __init__(self, graph: CanonicalEventGraph) -> None:
        self.graph = graph
        self.node_ticks: Dict[int, int] = {}
        self.status = ExtractionStatus.EXACT
        self.notes: List[str] = []

    @classmethod
    def merge_graphs(cls, graphs: List[CanonicalEventGraph]) -> CanonicalEventGraph:
        """Unify multiple event graphs by recursively merging identical paths."""
        merged = CanonicalEventGraph()
        # (graph_idx, original_node_id) -> merged_node
        node_map: Dict[Tuple[int, int], CEGNode] = {} 

        # 1. First, identify and merge roots with identical conditions
        # (condition_str, tick) -> merged_node
        # Note: ticks are not assigned yet, so we use structural identity
        merged_nodes: Dict[Tuple[str, Optional[int]], CEGNode] = {} 

        def get_merged_node(g_idx: int, orig_node: CEGNode) -> CEGNode:
            if (g_idx, orig_node.id) in node_map:
                return node_map[(g_idx, orig_node.id)]
            
            cond_str = emit_expr(orig_node.condition) if orig_node.condition else "None"
            # In an industrial tool, we'd also check absolute_tick if solved, 
            # but here we merge based on condition and topological similarity.
            key = (cond_str, orig_node.absolute_tick)
            
            if key in merged_nodes:
                target = merged_nodes[key]
                node_map[(g_idx, orig_node.id)] = target
                # Merge assignments
                for assign in orig_node.assignments:
                    if assign not in target.assignments:
                        target.assignments.append(assign)
                target.is_trigger = target.is_trigger or orig_node.is_trigger
                target.is_terminal = target.is_terminal or orig_node.is_terminal
                return target
            
            new_node = merged.create_node(condition=orig_node.condition, clocking=orig_node.clocking)
            new_node.is_trigger = orig_node.is_trigger
            new_node.is_terminal = orig_node.is_terminal
            new_node.assignments.extend(orig_node.assignments)
            new_node.absolute_tick = orig_node.absolute_tick
            
            node_map[(g_idx, orig_node.id)] = new_node
            # Cache all nodes by their key to enable global merging
            merged_nodes[key] = new_node
            return new_node

        # We need absolute ticks for better merging
        for g in graphs:
            solver = cls(g)
            solver.solve()

        for g_idx, g in enumerate(graphs):
            for orig_node in g.nodes.values():
                get_merged_node(g_idx, orig_node)

            for edge in g.edges:
                src = node_map[(g_idx, edge.source_id)]
                dst = node_map[(g_idx, edge.target_id)]
                # Check if edge already exists in merged graph
                exists = any(
                    e.source_id == src.id and 
                    e.target_id == dst.id and 
                    e.kind == edge.kind and 
                    e.delay == edge.delay 
                    for e in merged.edges
                )
                if not exists:
                    merged.add_edge(src, dst, kind=edge.kind, delay=edge.delay, group_id=edge.group_id)

        # Add causal ordering: if graph[i]'s terminal condition shares signals
        # with graph[j]'s root condition, add a zero-delay edge to enforce
        # that j's root comes after i's terminal in the merged timeline.
        for i, g_i in enumerate(graphs):
            for t_node in g_i.get_terminals():
                t_merged = node_map[(i, t_node.id)]
                t_cond = emit_expr(t_node.condition) if t_node.condition else None
                if not t_cond:
                    continue
                t_signals = set(collect_signals(parse_sva_condition(t_cond)))
                if not t_signals:
                    continue
                for j, g_j in enumerate(graphs):
                    if j <= i:
                        continue  # Only forward edges to prevent cycles
                    for r_node in g_j.get_roots():
                        r_merged = node_map[(j, r_node.id)]
                        if t_merged.id == r_merged.id:
                            continue  # Same merged node, already ordered
                        r_cond = emit_expr(r_node.condition) if r_node.condition else None
                        if not r_cond:
                            continue
                        r_signals = set(collect_signals(parse_sva_condition(r_cond)))
                        if t_signals & r_signals:
                            edge_exists = any(
                                e.source_id == t_merged.id and e.target_id == r_merged.id
                                for e in merged.edges
                            )
                            if not edge_exists:
                                merged.add_edge(
                                    t_merged, r_merged,
                                    kind=EdgeKind.DELAY,
                                    delay=TimeRange(min_delay="0", max_delay="0"),
                                )

        return merged

    def solve(self) -> ExtractionStatus:
        """Assign absolute ticks to nodes. Returns extraction status."""
        roots = self.graph.get_roots()
        if not roots:
            return ExtractionStatus.EXACT

        # Initialize all nodes with tick 0 to avoid KeyErrors
        for nid in self.graph.nodes:
            self.node_ticks[nid] = 0

        # Topological-style traversal to find longest paths (max ticks)
        # We use a simple worklist approach that handles DAGs
        worklist = [(root.id, 0) for root in roots]
        visited_count = {nid: 0 for nid in self.graph.nodes}
        
        while worklist:
            u_id, tick = worklist.pop(0)
            self.node_ticks[u_id] = max(self.node_ticks[u_id], tick)
            self.graph.nodes[u_id].absolute_tick = self.node_ticks[u_id]
            
            for edge in self.graph.get_out_edges(u_id):
                v_id = edge.target_id
                d = 0
                if edge.kind == EdgeKind.DELAY and edge.delay:
                    try:
                        d = int(edge.delay.min_delay)
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid min_delay {edge.delay.min_delay!r} on edge "
                            f"{edge.source_id}->{edge.target_id}"
                        ) from exc
                
                new_tick = self.node_ticks[u_id] + d
                if new_tick > self.node_ticks[v_id]:
                    self.node_ticks[v_id] = new_tick
                    worklist.append((v_id, new_tick))
                
                visited_count[v_id] += 1
                if visited_count[v_id] > len(self.graph.nodes) * len(self.graph.edges):
                    # Simple cycle detection fallback
                    self.notes.append("potential cycle detected in event graph")
                    return ExtractionStatus.UNSUPPORTED

        return self.status

    def project(self, prefix: str = "sva") -> Tuple[List[Anchor], List[TimeWindow], List[LaneConstraint]]:
        """Project the solved CEG into ScenariDocument components."""
        anchors = []
        windows = []
        constraints = []

        node_to_anchor: Dict[int, str] = {}

        # 1. Create Anchors for ALL nodes
        for node in self.graph.nodes.values():
            if node.condition is not None:
                anchor_name = f"{prefix}__node_{node.id}"
                role = AnchorRole.TRIGGER if node.is_trigger else AnchorRole.RESPONSE
                condition = parse_sva_condition(emit_expr(node.condition))
                anchors.append(Anchor(name=anchor_name, condition=condition, role=role, absolute_tick=node.absolute_tick))
                node_to_anchor[node.id] = anchor_name
            else:
                # Structural point
                anchor_name = f"{prefix}__point_{node.id}"
                role = AnchorRole.TRIGGER if node.is_trigger else AnchorRole.RESPONSE
                condition = parse_sva_condition("1'b1")
                # We mark it as synthetic via role_metadata
                anchors.append(Anchor(name=anchor_name, condition=condition, role=role, role_metadata="synthetic", absolute_tick=node.absolute_tick))
                node_to_anchor[node.id] = anchor_name

        # 2. Create Windows
        for i, edge in enumerate(self.graph.edges):
            if edge.kind == EdgeKind.DELAY and edge.delay:
                src_anchor = node_to_anchor.get(edge.source_id)
                dst_anchor = node_to_anchor.get(edge.target_id)
                
                if src_anchor and dst_anchor:
                    bound = self._to_time_bound(edge.delay)
                    windows.append(TimeWindow(
                        name=f"{prefix}__win_{i}",
                        start_anchor=src_anchor,
                        end_anchor=dst_anchor,
                        bound=bound
                    ))

        # 3. Create Constraints (Assignments)
        for node in self.graph.nodes.values():
            anchor_name = node_to_anchor.get(node.id)
            if anchor_name:
                for lval, rvalue in node.assignments:
                    constraints.append(LaneConstraint(
                        name=f"{prefix}__assign_{node.id}_{lval}",
                        signals=(lval,),
                        relation="eq",
                        value=emit_expr(rvalue),
                        region=ConstraintRegion.AT,
                        anchor=anchor_name
                    ))

        return anchors, windows, constraints

    def _to_time_bound(self, delay: TimeRange) -> TimeBound:
        if delay.unbounded:
            return TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay=delay.min_delay)
        if delay.is_exact():
            return TimeBound(kind=WindowBoundKind.EXACT, min_delay=delay.min_delay, max_delay=delay.min_delay)
        return TimeBound(kind=WindowBoundKind.RANGE, min_delay=delay.min_delay, max_delay=delay.max_delay)
