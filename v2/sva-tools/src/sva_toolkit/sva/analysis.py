from __future__ import annotations

from sva_toolkit.sva.ast import CallExpr, Identifier, Literal, Node, PropertySpec
from sva_toolkit.sva.visitors import NodeVisitor, iter_child_nodes


class CollectIdentifiersVisitor(NodeVisitor[set[str]]):
    def visit_Identifier(self, node: Identifier) -> set[str]:
        return {node.name}

    def generic_visit(self, node: Node | None) -> set[str]:
        found: set[str] = set()
        if node is None:
            return found
        for child in iter_child_nodes(node):
            found.update(self.visit(child))
        return found


class CollectCallsVisitor(NodeVisitor[set[str]]):
    def visit_CallExpr(self, node: CallExpr) -> set[str]:
        found = {node.name}
        for arg in node.args:
            found.update(self.visit(arg))
        return found

    def generic_visit(self, node: Node | None) -> set[str]:
        found: set[str] = set()
        if node is None:
            return found
        for child in iter_child_nodes(node):
            found.update(self.visit(child))
        return found


class CollectBoundNamesVisitor(NodeVisitor[set[str]]):
    def __init__(
        self,
        *,
        params: bool = True,
        locals: bool = True,
        clock: bool = True,
        reset: bool = True,
    ) -> None:
        self.include_params = params
        self.include_locals = locals
        self.include_clock = clock
        self.include_reset = reset

    def visit_PropertySpec(self, node: PropertySpec) -> set[str]:
        names: set[str] = set()
        if self.include_params:
            names.update(formal.name for formal in node.formals)
        if self.include_locals:
            names.update(local.name for local in node.local_vars)
        if self.include_clock and node.clocking is not None:
            names.add(node.clocking.signal.name)
        if self.include_reset and node.disable_iff is not None:
            names.update(CollectIdentifiersVisitor().visit(node.disable_iff))
        return names

    def generic_visit(self, node: Node | None) -> set[str]:
        return set()


class ContainsNodeKindVisitor(NodeVisitor[bool]):
    def __init__(self, node_kind: type[Node] | tuple[type[Node], ...]) -> None:
        self.node_kind = node_kind

    def visit(self, node: Node | None) -> bool:
        if node is None:
            return False
        if isinstance(node, self.node_kind):
            return True
        return super().visit(node)

    def generic_visit(self, node: Node | None) -> bool:
        if node is None:
            return False
        for child in iter_child_nodes(node):
            if self.visit(child):
                return True
        return False


def is_trivial_true(node: Node) -> bool:
    return isinstance(node, Literal) and node.text.strip().lower() in {"1", "1'b1"}


__all__ = [
    "CollectBoundNamesVisitor",
    "CollectCallsVisitor",
    "CollectIdentifiersVisitor",
    "ContainsNodeKindVisitor",
    "is_trivial_true",
]
