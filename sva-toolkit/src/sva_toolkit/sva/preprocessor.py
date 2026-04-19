"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T01.

This file is the minimal SystemVerilog preprocessor pass that runs before
`sva_toolkit.sva.lexer.tokenize`. Its responsibility is to recognize
backtick-prefixed directives (`` `define ``, `` `undef ``, `` `include ``,
`` `ifdef ``, `` `ifndef ``, `` `else ``, `` `endif ``, `` `timescale ``,
`` `protect ``), attribute instances `(* ... *)`, and encrypted-IP regions,
and to either strip them from the input or record them as `Trivia` entries
(see `trivia.py`). The module must expose a
`PreprocessResult(stripped_text, trivia: list[Trivia], directives: list[Directive])`
dataclass and a `preprocess(source)` entry point that returns it. This pass
is deliberately not a full macro-expander — captured `\`define` bodies stay
verbatim and are made available through `PreprocessResult.directives` so
that future work (out of scope for T01) can expand them. Encrypted-IP
regions (`\`protect` ... `\`endprotect`) are preserved as a single
`Trivia(kind=DIRECTIVE)` so the lexer can raise a typed, actionable error
instead of producing nonsense tokens. The module depends only on
`sva_toolkit.sva.trivia`, `sva_toolkit.sva.errors`, and Python's `re`.
Relates to DAG task T01 and feeds §2.3 of `docs/gaps.md` (lexer ignores
SystemVerilog source realities).
"""

from __future__ import annotations

# Implementation belongs to T01. Intentionally empty.
