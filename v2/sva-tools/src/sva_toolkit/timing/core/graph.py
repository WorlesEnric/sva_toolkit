"""Canonical Event Graph (CEG) for temporal synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

from sva_toolkit.sva.ast import ClockingEvent, ExprNode


class EdgeKind(str, Enum):
    """Types of temporal relationships between nodes."""
    DELAY = "delay"
    INTERSECT = "intersect"
    AND = "and"
    OR = "or"


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A temporal bound [min:max]."""
    min_delay: str
    max_delay: Optional[str] = None
    unbounded: bool = False

    def is_exact(self) -> bool:
        return not self.unbounded and self.min_delay == self.max_delay

    @property
    def label(self) -> str:
        if self.unbounded:
            return f"[{self.min_delay}:$]"
        if self.is_exact():
            return self.min_delay
        return f"[{self.min_delay}:{self.max_delay}]"


@dataclass(slots=True)
class CEGNode:
    """A node in the Canonical Event Graph representing an event or state."""
    id: int
    condition: Optional[ExprNode] = None
    assignments: List[Tuple[str, ExprNode]] = field(default_factory=list)
    clocking: Optional[ClockingEvent] = None
    
    # Metadata for synthesis
    is_trigger: bool = False
    is_terminal: bool = False
    absolute_tick: Optional[int] = None
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(slots=True)
class CEGEdge:
    """A temporal or logical edge between CEG nodes."""
    source_id: int
    target_id: int
    kind: EdgeKind = EdgeKind.DELAY
    delay: Optional[TimeRange] = None
    
    # For branching (AND/OR/INTERSECT)
    group_id: Optional[int] = None


class CanonicalEventGraph:
    """A DAG representing temporal and logical constraints for SVA/Diagram synthesis."""

    def __init__(self) -> None:
        self.nodes: dict[int, CEGNode] = {}
        self.edges: list[CEGEdge] = []
        self._next_node_id = 0
        self._next_group_id = 0

    def create_node(self, condition: Optional[ExprNode] = None, clocking: Optional[ClockingEvent] = None) -> CEGNode:
        node_id = self._next_node_id
        self._next_node_id += 1
        node = CEGNode(id=node_id, condition=condition, clocking=clocking)
        self.nodes[node_id] = node
        return node

    def add_edge(self, source: CEGNode, target: CEGNode, kind: EdgeKind = EdgeKind.DELAY, delay: Optional[TimeRange] = None, group_id: Optional[int] = None) -> CEGEdge:
        edge = CEGEdge(source_id=source.id, target_id=target.id, kind=kind, delay=delay, group_id=group_id)
        self.edges.append(edge)
        return edge

    def next_group_id(self) -> int:
        group_id = self._next_group_id
        self._next_group_id += 1
        return group_id

    def get_out_edges(self, node_id: int) -> list[CEGEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def get_in_edges(self, node_id: int) -> list[CEGEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def get_roots(self) -> list[CEGNode]:
        node_ids = set(self.nodes.keys())
        target_ids = {e.target_id for e in self.edges}
        root_ids = node_ids - target_ids
        return [self.nodes[nid] for nid in sorted(root_ids)]

    def get_terminals(self) -> list[CEGNode]:
        node_ids = set(self.nodes.keys())
        source_ids = {e.source_id for e in self.edges}
        terminal_ids = node_ids - source_ids
        return [self.nodes[nid] for nid in sorted(terminal_ids)]
