# Adversarial SVA Corpus

This directory holds the small, hand-authored SystemVerilog assertion
files used by the T14 regression suite. Each fixture locks one lexer or
preprocessor edge from `docs/gaps.md` §2.3 / §5 so future changes cannot
quietly reintroduce the original failure modes.

- `with_line_comments.sv`: leading, inline, and trailing `//` comments.
- `with_block_comments.sv`: block comments, including a `/* nested */`
  marker that must stop at the first terminator.
- `with_backtick_directives.sv`: `` `define``, `` `ifdef``, `` `include``,
  `` `timescale``, and the closing `` `endif``.
- `with_string_literals.sv`: a real string token inside an assertion
  expression.
- `with_attributes.sv`: a `(* ... *)` attribute instance attached to an
  assertion.
- `with_encrypted_ip.sv`: an encrypted-IP marker that must fail with a
  typed parse error instead of crashing or silently skipping content.
