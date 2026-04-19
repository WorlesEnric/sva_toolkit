# Adversarial SVA Corpus

SCAFFOLD SUMMARY — replace this paragraph with the real fixture notes in task T14.

This directory holds small, hand-authored SystemVerilog assertion files
used by the T14 integration suite to stress the lexer and parser on
inputs that today (pre-remediation) fail or silently downgrade. Each
file targets one concern from `docs/gaps.md` §2.3 / §3.1 / risk
register, and the filename encodes the concern: `with_line_comments.sv`
for `//`, `with_block_comments.sv` for `/* … */`, `with_backtick_directives.sv`
for `` `define``/`` `ifdef``/`` `include``, `with_string_literals.sv`
for `"…"`, `with_attributes.sv` for `(* … *)` instances,
`with_encrypted_ip.sv` for `` `protect`` regions (expected to raise a
typed error, not crash). Relates to DAG task T14.
