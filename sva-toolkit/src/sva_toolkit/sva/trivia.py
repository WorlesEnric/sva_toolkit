"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T01.

This file owns the `Trivia` dataclass and the helpers used by
`sva_toolkit.sva.lexer` to represent and skip source-level material that is
not part of the SVA grammar proper: `//` line comments, `/* ... */` block
comments, horizontal and vertical whitespace, line continuations (`\\` at
EOL), attribute instances `(* attr = "val" *)`, and the textual spans
consumed by the preprocessor pass (see `preprocessor.py`). The module must
expose a `Trivia` frozen dataclass with fields `kind`, `text`, and
`span: SourceSpan`, a `TriviaKind` enum
(`COMMENT_LINE`, `COMMENT_BLOCK`, `WHITESPACE`, `LINE_CONTINUATION`,
`ATTRIBUTE`, `DIRECTIVE`), and a `collect_trivia(source, start)` helper
that a caller can use to walk over any non-token span. The lexer keeps
its primary return type (a `list[Token]`) but exposes a secondary
accessor `tokenize_with_trivia(source)` that returns both lists in source
order so that later tooling (describe, emitter) can preserve user
comments on round-trip. This file is the natural home for the predicates
(e.g. `is_comment_start`, `is_attribute_start`) that the lexer consults
before it commits to producing a token. It must not import the parser or
the AST; it depends only on `sva_toolkit.sva.errors.SvaSyntaxError` and
`sva_toolkit.sva.ast.SourceSpan`. Relates to DAG task T01.
"""

from __future__ import annotations

# Implementation belongs to T01. Intentionally empty.
