from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sva_toolkit.sva.ast import SourceSpan
from sva_toolkit.sva.errors import SvaSyntaxError


class TriviaKind(str, Enum):
    COMMENT_LINE = "comment_line"
    COMMENT_BLOCK = "comment_block"
    DIRECTIVE = "directive"
    ATTRIBUTE = "attribute"
    WHITESPACE = "whitespace"
    LINE_CONTINUATION = "line_continuation"


@dataclass(frozen=True, slots=True)
class Trivia:
    kind: TriviaKind
    span: SourceSpan
    text: str


def collect_trivia(source: str, start: int) -> Trivia | None:
    if start >= len(source):
        return None

    if is_line_continuation_start(source, start):
        end = consume_line_continuation(source, start)
        return Trivia(TriviaKind.LINE_CONTINUATION, SourceSpan(start, end), source[start:end])

    if source[start].isspace():
        end = consume_whitespace(source, start)
        return Trivia(TriviaKind.WHITESPACE, SourceSpan(start, end), source[start:end])

    if is_line_comment_start(source, start):
        end = consume_line_comment(source, start)
        return Trivia(TriviaKind.COMMENT_LINE, SourceSpan(start, end), source[start:end])

    if is_block_comment_start(source, start):
        end = consume_block_comment(source, start)
        return Trivia(TriviaKind.COMMENT_BLOCK, SourceSpan(start, end), source[start:end])

    if is_attribute_start(source, start):
        end = consume_attribute(source, start)
        return Trivia(TriviaKind.ATTRIBUTE, SourceSpan(start, end), source[start:end])

    return None


def is_line_comment_start(source: str, start: int) -> bool:
    return source.startswith("//", start)


def is_block_comment_start(source: str, start: int) -> bool:
    return source.startswith("/*", start)


def is_attribute_start(source: str, start: int) -> bool:
    return source.startswith("(*", start)


def is_line_continuation_start(source: str, start: int) -> bool:
    if start >= len(source) or source[start] != "\\":
        return False
    if start + 1 >= len(source):
        return False
    if source[start + 1] == "\n":
        return True
    return source[start + 1] == "\r" and start + 2 < len(source) and source[start + 2] == "\n"


def consume_whitespace(source: str, start: int) -> int:
    index = start
    while index < len(source) and source[index].isspace() and not is_line_continuation_start(source, index):
        index += 1
    return index


def consume_line_comment(source: str, start: int) -> int:
    index = start + 2
    while index < len(source) and source[index] not in {"\n", "\r"}:
        index += 1
    return index


def consume_block_comment(source: str, start: int) -> int:
    index = source.find("*/", start + 2)
    if index < 0:
        raise SvaSyntaxError(start, "unterminated block comment", source)
    return index + 2


def consume_line_continuation(source: str, start: int) -> int:
    if not is_line_continuation_start(source, start):
        raise SvaSyntaxError(start, "invalid line continuation", source)
    if source[start + 1] == "\n":
        return start + 2
    return start + 3


def consume_string_literal(source: str, start: int) -> int:
    index = start + 1
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 1
            if index >= len(source):
                break
            if source[index] in {"\n", "\r"}:
                raise SvaSyntaxError(start, "unterminated string literal", source)
            index += 1
            continue
        if char == '"':
            return index + 1
        if char in {"\n", "\r"}:
            raise SvaSyntaxError(start, "unterminated string literal", source)
        index += 1
    raise SvaSyntaxError(start, "unterminated string literal", source)


def consume_attribute(source: str, start: int) -> int:
    index = start + 2
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
        if source.startswith("*)", index):
            return index + 2
        index += 1
    raise SvaSyntaxError(start, "unterminated attribute instance", source)


__all__ = [
    "Trivia",
    "TriviaKind",
    "collect_trivia",
    "consume_attribute",
    "consume_block_comment",
    "consume_line_comment",
    "consume_line_continuation",
    "consume_string_literal",
    "consume_whitespace",
    "is_attribute_start",
    "is_block_comment_start",
    "is_line_comment_start",
    "is_line_continuation_start",
]
