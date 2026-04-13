"""Shared boolean condition and predicate objects for timing scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class Predicate:
    """A typed predicate over one timing signal."""

    op: str
    signal: Optional[str] = None
    value: Optional[str] = None
    args: Tuple[str, ...] = ()
    text: Optional[str] = None


@dataclass(frozen=True)
class Condition:
    """A normalized boolean condition tree."""

    kind: str
    predicate: Optional[Predicate] = None
    items: Tuple["Condition", ...] = field(default_factory=tuple)
    text: Optional[str] = None


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_UNARY_PREDICATE_RE = re.compile(rf"^(rise|fall|high|low|change|stable|unknown|dontcare)\(\s*({IDENT})\s*\)$")
_GROUP_STABLE_RE = re.compile(rf"^stable\(\s*\{{\s*({IDENT}(?:\s*,\s*{IDENT})*)\s*\}}\s*\)$")
_EQ_PREDICATE_RE = re.compile(rf"^(eq|neq)\(\s*({IDENT})\s*,\s*(.+)\s*\)$")
_SVA_SYSFUNC_RE = re.compile(rf"^\$(rose|fell|stable|changed|isunknown)\(\s*({IDENT})\s*\)$", re.IGNORECASE)
_SVA_PAST_RE = re.compile(rf"^\$past\(\s*({IDENT})\s*(?:,\s*([^)]+?)\s*)?\)$", re.IGNORECASE)
_SVA_EQ_RE = re.compile(rf"^({IDENT})\s*(==|!=)\s*(.+)$")


def predicate_condition(op: str, signal: str, value: Optional[str] = None) -> Condition:
    """Create a predicate-backed condition."""

    return Condition(kind="predicate", predicate=Predicate(op=op, signal=signal, value=value))


def raw_condition(text: str) -> Condition:
    """Create a raw opaque condition."""

    return Condition(kind="raw", text=text.strip())


def all_of(items: Iterable[Condition]) -> Condition:
    """Build a conjunction, flattening nested conjunctions."""

    flattened = []
    for item in items:
        if item.kind == "all":
            flattened.extend(item.items)
        else:
            flattened.append(item)
    if not flattened:
        return raw_condition("1'b1")
    if len(flattened) == 1:
        return flattened[0]
    return Condition(kind="all", items=tuple(flattened))


def any_of(items: Iterable[Condition]) -> Condition:
    """Build a disjunction, flattening nested disjunctions."""

    flattened = []
    for item in items:
        if item.kind == "any":
            flattened.extend(item.items)
        else:
            flattened.append(item)
    if not flattened:
        return raw_condition("1'b0")
    if len(flattened) == 1:
        return flattened[0]
    return Condition(kind="any", items=tuple(flattened))


def negate(item: Condition) -> Condition:
    """Build a negation with a small amount of normalization."""

    if item.kind == "not" and item.items:
        return item.items[0]
    if item.kind == "predicate" and item.predicate is not None:
        predicate = item.predicate
        if predicate.op == "high":
            return predicate_condition("low", predicate.signal or "")
        if predicate.op == "low":
            return predicate_condition("high", predicate.signal or "")
    return Condition(kind="not", items=(item,))


def collect_signals(condition: Condition) -> Tuple[str, ...]:
    """Collect unique signal names mentioned by the condition."""

    seen = []

    def visit(node: Condition) -> None:
        if node.kind == "predicate" and node.predicate is not None:
            predicate = node.predicate
            if predicate.signal and predicate.signal not in seen:
                seen.append(predicate.signal)
            for arg in predicate.args:
                if re.fullmatch(IDENT, arg) and arg not in seen:
                    seen.append(arg)
            return
        for child in node.items:
            visit(child)

    visit(condition)
    return tuple(seen)


def condition_to_dsl(condition: Condition) -> str:
    """Render a normalized condition into timing DSL syntax."""

    if condition.kind == "predicate" and condition.predicate is not None:
        return _predicate_to_dsl(condition.predicate)
    if condition.kind == "all":
        return " and ".join(_wrap_if_needed(item, "all") for item in condition.items)
    if condition.kind == "any":
        return " or ".join(_wrap_if_needed(item, "any") for item in condition.items)
    if condition.kind == "not" and condition.items:
        inner = condition.items[0]
        return f"not {_wrap_if_needed(inner, 'not')}"
    return condition.text or ""


def condition_to_sva(condition: Condition) -> str:
    """Render a normalized condition into SVA-like syntax."""

    if condition.kind == "predicate" and condition.predicate is not None:
        return _predicate_to_sva(condition.predicate)
    if condition.kind == "all":
        return " && ".join(_wrap_if_needed_sva(item, "all") for item in condition.items)
    if condition.kind == "any":
        return " || ".join(_wrap_if_needed_sva(item, "any") for item in condition.items)
    if condition.kind == "not" and condition.items:
        inner = condition.items[0]
        return f"!{_wrap_if_needed_sva(inner, 'not')}"
    return condition.text or ""


def parse_dsl_condition(text: str) -> Condition:
    """Parse a line-oriented DSL condition subset used by anchors and show clauses."""

    stripped = _strip_outer_parens(text.strip())
    if not stripped:
        return raw_condition("")

    parts = _split_top_level_word(stripped, "or")
    if len(parts) > 1:
        return any_of(parse_dsl_condition(part) for part in parts)

    parts = _split_top_level_word(stripped, "and")
    if len(parts) > 1:
        return all_of(parse_dsl_condition(part) for part in parts)

    if stripped.startswith("not "):
        return negate(parse_dsl_condition(stripped[4:].strip()))

    if group_match := _GROUP_STABLE_RE.match(stripped):
        names = [name.strip() for name in group_match.group(1).split(",") if name.strip()]
        return all_of(predicate_condition("stable", name) for name in names)

    if unary_match := _UNARY_PREDICATE_RE.match(stripped):
        return predicate_condition(unary_match.group(1), unary_match.group(2))

    if eq_match := _EQ_PREDICATE_RE.match(stripped):
        return predicate_condition(eq_match.group(1), eq_match.group(2), eq_match.group(3).strip())

    if re.fullmatch(IDENT, stripped):
        return predicate_condition("high", stripped)
    if stripped.startswith("!") and re.fullmatch(IDENT, stripped[1:].strip()):
        return predicate_condition("low", stripped[1:].strip())
    return raw_condition(stripped)


def parse_sva_condition(text: str) -> Condition:
    """Parse the practical SVA subset used by timing extraction."""

    stripped = _strip_outer_parens(text.strip())
    if not stripped:
        return raw_condition("")

    parts = _split_top_level(stripped, "||")
    if len(parts) > 1:
        return any_of(parse_sva_condition(part) for part in parts)

    parts = _split_top_level(stripped, "&&")
    if len(parts) > 1:
        return all_of(parse_sva_condition(part) for part in parts)

    if stripped.startswith("!"):
        tail = stripped[1:].strip()
        if re.fullmatch(IDENT, tail):
            return predicate_condition("low", tail)
        return negate(parse_sva_condition(tail))

    if func_match := _SVA_SYSFUNC_RE.match(stripped):
        op = func_match.group(1).lower()
        op = {"changed": "change", "isunknown": "unknown"}.get(op, op)
        return predicate_condition(op, func_match.group(2))

    if past_match := _SVA_PAST_RE.match(stripped):
        args = tuple(arg.strip() for arg in past_match.groups() if arg)
        return Condition(
            kind="predicate",
            predicate=Predicate(op="past", signal=past_match.group(1), args=args, text=stripped),
        )

    if eq_match := _SVA_EQ_RE.match(stripped):
        return predicate_condition("eq" if eq_match.group(2) == "==" else "neq", eq_match.group(1), eq_match.group(3).strip())

    if re.fullmatch(IDENT, stripped):
        return predicate_condition("high", stripped)

    return raw_condition(stripped)


def _predicate_to_dsl(predicate: Predicate) -> str:
    if predicate.op == "eq":
        return f"eq({predicate.signal}, {predicate.value})"
    if predicate.op == "neq":
        return f"neq({predicate.signal}, {predicate.value})"
    if predicate.op == "past":
        if predicate.args:
            return f"$past({predicate.signal}, {', '.join(predicate.args[1:])})" if len(predicate.args) > 1 else f"$past({predicate.signal}, {predicate.args[0]})"
        return f"$past({predicate.signal})"
    if predicate.signal is not None:
        return f"{predicate.op}({predicate.signal})"
    return predicate.text or ""


def _predicate_to_sva(predicate: Predicate) -> str:
    if predicate.op == "high":
        return predicate.signal or ""
    if predicate.op == "low":
        return f"!{predicate.signal}"
    if predicate.op == "rise":
        return f"$rose({predicate.signal})"
    if predicate.op == "fall":
        return f"$fell({predicate.signal})"
    if predicate.op == "stable":
        return f"$stable({predicate.signal})"
    if predicate.op == "change":
        return f"$changed({predicate.signal})"
    if predicate.op == "unknown":
        return f"$isunknown({predicate.signal})"
    if predicate.op == "dontcare":
        return predicate.signal or ""
    if predicate.op == "eq":
        return f"({predicate.signal} == {predicate.value})"
    if predicate.op == "neq":
        return f"({predicate.signal} != {predicate.value})"
    if predicate.op == "past":
        if len(predicate.args) >= 2:
            return f"$past({predicate.args[0]}, {predicate.args[1]})"
        return f"$past({predicate.signal})"
    return predicate.text or ""


def _wrap_if_needed(item: Condition, parent_kind: str) -> str:
    rendered = condition_to_dsl(item)
    if item.kind in {"all", "any"} and item.kind != parent_kind:
        return f"({rendered})"
    if item.kind == "not" and item.items and item.items[0].kind in {"all", "any"}:
        return f"({rendered})"
    return rendered


def _wrap_if_needed_sva(item: Condition, parent_kind: str) -> str:
    rendered = condition_to_sva(item)
    if item.kind in {"all", "any"} and item.kind != parent_kind:
        return f"({rendered})"
    if item.kind == "not" and item.items and item.items[0].kind in {"all", "any"}:
        return f"({rendered})"
    return rendered


def _strip_outer_parens(text: str) -> str:
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        wrapped = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    wrapped = False
                    break
        if not wrapped:
            break
        text = text[1:-1].strip()
    return text


def _split_top_level(text: str, token: str) -> Tuple[str, ...]:
    parts = []
    start = 0
    depth_paren = 0
    depth_brack = 0
    index = 0
    while index <= len(text) - len(token):
        char = text[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        if depth_paren == 0 and depth_brack == 0 and text[index:index + len(token)] == token:
            parts.append(text[start:index].strip())
            start = index + len(token)
            index = start
            continue
        index += 1
    if not parts:
        return (text.strip(),)
    parts.append(text[start:].strip())
    return tuple(part for part in parts if part)


def _split_top_level_word(text: str, word: str) -> Tuple[str, ...]:
    parts = []
    start = 0
    depth_paren = 0
    depth_brack = 0
    index = 0
    token = f" {word} "
    while index <= len(text) - len(token):
        char = text[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        if depth_paren == 0 and depth_brack == 0 and text[index:index + len(token)] == token:
            parts.append(text[start:index].strip())
            start = index + len(token)
            index = start
            continue
        index += 1
    if not parts:
        return (text.strip(),)
    parts.append(text[start:].strip())
    return tuple(part for part in parts if part)
