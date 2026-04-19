from __future__ import annotations

from dataclasses import fields, replace
from typing import Generic, TypeVar

from sva_toolkit.sva.ast import Node


T = TypeVar("T")


def iter_child_nodes(node: Node) -> tuple[Node, ...]:
    children: list[Node] = []
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Node):
            children.append(value)
            continue
        if isinstance(value, tuple):
            children.extend(item for item in value if isinstance(item, Node))
    return tuple(children)


class NodeVisitor(Generic[T]):
    def visit(self, node: Node | None) -> T:
        if node is None:
            return self.generic_visit(None)
        method = getattr(self, f"visit_{type(node).__name__}", self.generic_visit)
        return method(node)

    def generic_visit(self, node: Node | None) -> T:
        for child in iter_child_nodes(node) if isinstance(node, Node) else ():
            self.visit(child)
        return None  # type: ignore[return-value]


class NodeTransformer(NodeVisitor[Node]):
    def generic_visit(self, node: Node | None) -> Node:
        if node is None:
            return node
        changes: dict[str, object] = {}
        changed = False
        for field in fields(node):
            value = getattr(node, field.name)
            new_value = self._transform_value(value)
            changes[field.name] = new_value
            changed = changed or new_value is not value
        if not changed:
            return node
        return replace(node, **changes)

    def _transform_value(self, value: object) -> object:
        if isinstance(value, Node):
            return self.visit(value)
        if isinstance(value, tuple):
            items = tuple(self.visit(item) if isinstance(item, Node) else item for item in value)
            return items if items != value else value
        return value


__all__ = ["NodeTransformer", "NodeVisitor", "iter_child_nodes"]
