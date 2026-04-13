from __future__ import annotations

from dataclasses import replace

from sva_toolkit.sva.ast import Identifier, LocalVarDecl, PropertyFormal
from sva_toolkit.sva.visitors import NodeTransformer


class RenameIdentifiersTransformer(NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Identifier(self, node: Identifier):
        return replace(node, name=self.mapping.get(node.name, node.name))

    def visit_PropertyFormal(self, node: PropertyFormal):
        renamed = self.generic_visit(node)
        return replace(renamed, name=self.mapping.get(renamed.name, renamed.name))

    def visit_LocalVarDecl(self, node: LocalVarDecl):
        renamed = self.generic_visit(node)
        return replace(renamed, name=self.mapping.get(renamed.name, renamed.name))


class SubstituteIdentifiersTransformer(NodeTransformer):
    def __init__(self, mapping: dict[str, object]) -> None:
        self.mapping = mapping

    def visit_Identifier(self, node: Identifier):
        return self.mapping.get(node.name, node)


__all__ = ["RenameIdentifiersTransformer", "SubstituteIdentifiersTransformer"]
