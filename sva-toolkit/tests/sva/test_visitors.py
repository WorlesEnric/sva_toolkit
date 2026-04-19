from __future__ import annotations

from sva_toolkit.sva import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    Identifier,
    Literal,
    UnaryExpr,
    UnaryOperator,
)
from sva_toolkit.sva.analysis import (
    CollectIdentifiersVisitor,
    ContainsNodeKindVisitor,
    is_trivial_true,
)
from sva_toolkit.sva.transforms import RenameIdentifiersTransformer, SubstituteIdentifiersTransformer


def test_collect_identifiers_visitor_collects_unique_names() -> None:
    node = BinaryExpr(
        left=CallExpr(name="$rose", args=(Identifier(name="req"),)),
        op=BinaryOperator.LOGICAL_AND,
        right=BinaryExpr(
            left=Identifier(name="ack"),
            op=BinaryOperator.EQ,
            right=Identifier(name="data"),
        ),
    )

    assert CollectIdentifiersVisitor().visit(node) == {"req", "ack", "data"}


def test_rename_identifiers_transformer_renames_identifier_nodes() -> None:
    node = BinaryExpr(
        left=Identifier(name="req"),
        op=BinaryOperator.LOGICAL_AND,
        right=CallExpr(name="$rose", args=(Identifier(name="ack"),)),
    )

    rewritten = RenameIdentifiersTransformer({"req": "src_req", "ack": "src_ack"}).visit(node)

    assert isinstance(rewritten.left, Identifier)
    assert rewritten.left.name == "src_req"
    assert isinstance(rewritten.right, CallExpr)
    assert isinstance(rewritten.right.args[0], Identifier)
    assert rewritten.right.args[0].name == "src_ack"


def test_substitute_identifiers_transformer_substitutes_expressions() -> None:
    node = BinaryExpr(
        left=Identifier(name="req"),
        op=BinaryOperator.LOGICAL_OR,
        right=Identifier(name="ack"),
    )

    rewritten = SubstituteIdentifiersTransformer(
        {
            "req": UnaryExpr(op=UnaryOperator.LOGICAL_NOT, operand=Identifier(name="rst_n")),
            "ack": Literal(text="1'b1"),
        }
    ).visit(node)

    assert isinstance(rewritten.left, UnaryExpr)
    assert isinstance(rewritten.left.operand, Identifier)
    assert rewritten.left.operand.name == "rst_n"
    assert isinstance(rewritten.right, Literal)
    assert rewritten.right.text == "1'b1"


def test_contains_node_kind_visitor_detects_present_kind() -> None:
    node = BinaryExpr(
        left=Identifier(name="req"),
        op=BinaryOperator.LOGICAL_AND,
        right=CallExpr(name="$rose", args=(Identifier(name="ack"),)),
    )

    assert ContainsNodeKindVisitor(CallExpr).visit(node) is True
    assert ContainsNodeKindVisitor(Literal).visit(node) is False


def test_is_trivial_true_recognizes_true_literals() -> None:
    assert is_trivial_true(Literal(text="1'b1")) is True
    assert is_trivial_true(Literal(text="1")) is True
    assert is_trivial_true(Identifier(name="req")) is False

