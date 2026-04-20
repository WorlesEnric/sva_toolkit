from __future__ import annotations

from dataclasses import dataclass

from sva_toolkit.sva.ast import SourceSpan
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.trivia import (
    Trivia,
    TriviaKind,
    consume_attribute,
    consume_block_comment,
    consume_line_comment,
    consume_string_literal,
    is_attribute_start,
    is_block_comment_start,
    is_line_comment_start,
)


_SUPPORTED_DIRECTIVES = {
    "define",
    "else",
    "endif",
    "ifdef",
    "ifndef",
    "include",
    "protect",
    "timescale",
    "undef",
}


@dataclass(frozen=True, slots=True)
class Directive:
    name: str
    span: SourceSpan
    text: str


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    stripped_text: str
    trivia: list[Trivia]
    directives: list[Directive]


def preprocess(source: str) -> PreprocessResult:
    stripped = list(source)
    trivia: list[Trivia] = []
    directives: list[Directive] = []
    conditional_stack: list[Directive] = []

    index = 0
    length = len(source)
    while index < length:
        if source[index] == '"':
            index = consume_string_literal(source, index)
            continue

        if is_line_comment_start(source, index):
            index = consume_line_comment(source, index)
            continue

        if is_block_comment_start(source, index):
            index = consume_block_comment(source, index)
            continue

        if is_attribute_start(source, index):
            end = consume_attribute(source, index)
            attribute = Trivia(TriviaKind.ATTRIBUTE, SourceSpan(index, end), source[index:end])
            trivia.append(attribute)
            _blank_out(stripped, source, index, end)
            index = end
            continue

        if source[index] != "`":
            index += 1
            continue

        name, name_end = _parse_directive_name(source, index)
        if name not in _SUPPORTED_DIRECTIVES:
            index += 1
            continue

        if name == "protect":
            end = _consume_protect_region(source, index)
        else:
            end = _consume_directive(source, index)

        span = SourceSpan(index, end)
        text = source[index:end]
        directive = Directive(name=name, span=span, text=text)
        directives.append(directive)
        trivia.append(Trivia(TriviaKind.DIRECTIVE, span, text))
        _blank_out(stripped, source, index, end)
        _update_conditional_stack(source, directive, conditional_stack)
        index = end

    if conditional_stack:
        opener = conditional_stack[-1]
        raise SvaSyntaxError(opener.span.start, f"unterminated `{opener.name}` block", source)

    return PreprocessResult(stripped_text="".join(stripped), trivia=trivia, directives=directives)


def _parse_directive_name(source: str, start: int) -> tuple[str, int]:
    index = start + 1
    while index < len(source) and (source[index].isalnum() or source[index] == "_"):
        index += 1
    return source[start + 1:index].lower(), index


def _consume_directive(source: str, start: int) -> int:
    _, index = _parse_directive_name(source, start)
    while index < len(source):
        if source[index] == '"':
            index = consume_string_literal(source, index)
            continue
        if source[index] == "\n":
            if index > start and source[index - 1] == "\\":
                index += 1
                continue
            return index + 1
        if source[index] == "\r":
            if index > start and source[index - 1] == "\\" and index + 1 < len(source) and source[index + 1] == "\n":
                index += 2
                continue
            if index + 1 < len(source) and source[index + 1] == "\n":
                return index + 2
            return index + 1
        index += 1
    return index


def _consume_protect_region(source: str, start: int) -> int:
    index = _consume_directive(source, start)
    while index < len(source):
        if source[index] == '"':
            index = consume_string_literal(source, index)
            continue
        if is_line_comment_start(source, index):
            index = consume_line_comment(source, index)
            continue
        if is_block_comment_start(source, index):
            index = consume_block_comment(source, index)
            continue
        if source[index] != "`":
            index += 1
            continue
        name, _ = _parse_directive_name(source, index)
        if name == "endprotect":
            return _consume_directive(source, index)
        index = _consume_directive(source, index)
    raise SvaSyntaxError(start, "unterminated `protect region", source)


def _update_conditional_stack(source: str, directive: Directive, stack: list[Directive]) -> None:
    if directive.name in {"ifdef", "ifndef"}:
        stack.append(directive)
        return
    if directive.name == "else":
        if not stack:
            raise SvaSyntaxError(directive.span.start, "`else without matching conditional directive", source)
        return
    if directive.name == "endif":
        if not stack:
            raise SvaSyntaxError(directive.span.start, "`endif without matching conditional directive", source)
        stack.pop()


def _blank_out(buffer: list[str], source: str, start: int, end: int) -> None:
    for index in range(start, end):
        if source[index] in {"\n", "\r"}:
            buffer[index] = source[index]
        else:
            buffer[index] = " "


__all__ = ["Directive", "PreprocessResult", "preprocess"]
