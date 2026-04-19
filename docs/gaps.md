# SVA Toolkit V3 — Gap and Risk Assessment

Status: alpha (v3.0.0a1). This document enumerates implementation gaps, robustness
weaknesses, and concrete industrial-use risks observed in the current codebase
(`sva-toolkit/src/sva_toolkit`). All claims are anchored to source locations so
reviewers can verify independently.

> TL;DR — the CLI is usable as a research/experimentation toolkit. Do **not**
> deploy it as-is on regulated, tape-out-critical, or customer-facing flows. The
> lexer has no comment/string/preprocessor handling; the parser silently
> down-grades unsupported SVA to opaque strings; the formal backend injects
> hard-coded `clk`/`!rst_n` defaults; generation is non-deterministic (no seed);
> LLM failures silently fall back without surfacing to the user.

---

## 1. Scope and verification

- Source root: `sva-toolkit/src/sva_toolkit/` (packages: `cli`, `sva`, `formal`,
  `timing`, `generate`, `describe`, `data`, `runtime`).
- External binaries referenced at runtime only: `ebmc`, `vcf`,
  `verible-verilog-syntax`, `cairosvg`. Missing tools degrade silently in some
  paths and hard-fail in others (see §4.2, §4.4).
- Tests: `tests/{sva,formal,timing,generate,describe,data,integration,runtime}`.
  Adversarial, malformed-input, large-design, concurrency, and
  tool-missing paths are thinly covered.

Verified directly by file-read for this document:

- `sva/lexer.py` (237 lines) — no comment/string/backtick handling.
- `sva/parser.py:91-136` — `recover=True` → opaque fallback.
- `formal/parse.py:87-100`, `formal/model.py:47-68` — hard-coded `clk`/`!rst_n`.
- `formal/backends/ebmc.py:14-22, 131-188` — module template and validator.
- `formal/service.py:41-58` — broad `except Exception` over parse.
- `cli/main.py:17-30` — generic exception swallow to `ClickException`.
- `timing/frontend/parser.py:34-61` — regex-based DSL grammar.
- `timing/core/conditions.py:134-146` — silent `except Exception: pass`.
- `runtime/process.py:1-58` — no orphan kill, world-readable `mkdtemp`.
- `runtime/llm.py:1-52` — no retries/backoff/rate-limit handling.
- `generate/synthesizer.py:111-189` — unseeded `random.*` (41 call-sites; no
  `random.seed` anywhere in `src/`).
- `data/dataset.py:44-51, 122-146, 213-217` — silent cache failures, silent LLM
  fallback, shared multiprocess cache dir without locking.
- `describe/translator.py` (1430 lines) — handles `OpaqueProperty`,
  `OpaqueSequence`, `OpaqueExpr` by collecting them as raw text (lines
  343, 390, 404).

---

## 2. SVA syntax coverage gaps (IEEE 1800-2023)

### 2.1 Not recognized as keywords / operators in the lexer

`sva/lexer.py` only reserves the keyword set listed at lines 85-111. The
following IEEE 1800 SVA constructs are **not tokenized** and will therefore be
lexed as plain identifiers (which then fail or produce wrong ASTs downstream):

| Category | Missing construct |
|---|---|
| Temporal property operators | `nexttime`, `s_nexttime`, `always`, `s_always`, `eventually`, `s_eventually`, `strong`, `weak` |
| Directives | `restrict`, `expect` (only `assert`, `assume`, `cover` are reserved) |
| Containers | `sequence` / `endsequence`, `checker` / `endchecker`, `bind`, `clocking` / `endclocking`, `default clocking`, `let` |
| Sequence ops | `within`, `matched`, `ended` as first-class (`SequenceEndedExpr` exists only for the `.ended` accessor in `parser.py`) |
| Expr helpers | `inside`, `dist`, `->`, `<->`, `iff` at property level (tokenized only as a keyword for `disable iff`) |
| Type tokens | `bit`, `logic`, `reg`, `wire`, `input`, `output`, numeric base widths — the lexer accepts sized literals but no type declarations |

Consequence: any source file that uses `sequence … endsequence`, `let`,
`checker`, `bind`, `restrict property`, `expect`, `nexttime`, `always`,
`eventually`, or multi-clock `@(posedge a or posedge b)` will either fail at
`tokenize()` or be misparsed.

### 2.2 Silent down-grade to opaque nodes

`sva/parser.py:91-136` exposes `parse_property_text`, `parse_property_body`,
`parse_sequence`, and `parse_expr` with a `recover: bool = False` flag. When
callers pass `recover=True` and parsing fails, the original string is wrapped in
`OpaqueProperty`/`OpaqueSequence`/`OpaqueExpr` nodes. No warning is emitted, no
counter is incremented, and no log line is written.

Downstream, `describe/translator.py:343, 390, 404` happily consumes those opaque
nodes by dumping their `text` field into the natural-language output. A user
running `sva describe` therefore cannot tell the difference between "the tool
analyzed this property" and "the tool gave up and echoed it verbatim".

### 2.3 Lexer ignores SystemVerilog source realities

`sva/lexer.py` handles whitespace, numeric literals, `$` system identifiers, and
operators. It does **not** handle:

- `//` line comments and `/* … */` block comments. `/` is mapped to
  `SLASH` unconditionally (line 143, 53), so `// clk` becomes two `SLASH`
  tokens followed by an identifier.
- String literals (`"..."`). A quote character is not in
  `_SINGLE_CHAR_TOKENS` — it raises `SvaSyntaxError` (line 210).
- Backtick-prefixed preprocessor directives (`` `define``, `` `include``,
  `` `ifdef``, `` `ifndef``, `` `timescale``). Grep across `src/` finds zero
  references to preprocessor handling.
- Line continuations (`\` at end of line) and escaped identifiers
  (`\name-with-dashes`).
- Attribute instances `(* attr = "val" *)`.

Consequence: any real RTL/SVA file must be pre-stripped by the user before it
can be fed in. Copy-pasting from a signed-off testbench will usually not work.

### 2.4 Clocking is single-edge only

`sva/lexer.py` reserves `posedge`/`negedge`. Neither the lexer nor
`sva/parser.py` supports:

- `@(posedge clk or negedge rst_n)` — multi-event clocks.
- `@(edge clk)` — bi-edge.
- `@(iff_sampled)` clocking expressions.
- `default clocking` blocks; per-property clocking inheritance.

`formal/parse.py:124-127` and `formal/model.py:51` both default to
`"posedge"`/`"clk"` when clocking is absent. See §4.4 for the downstream risk.

### 2.5 Property-combination operators are a strict subset

`sva/ast.py::PropertyBinaryOperator` is restricted to `and`, `or`, `until`,
`until_with` (confirmed by `parser.py:27-33` imports). There is no support for
`s_until`, `s_until_with`, `implies` (double arrow property), `iff` (biconditional),
`until_with_exclusive`, `throughout` at the property level, or user-defined
`property p_foo(args);` instantiation with argument binding.

`ControlOperator` covers `accept_on`/`reject_on`/`sync_accept_on`/
`sync_reject_on` (`parser.py:83-88`), but nothing enforces their position
(property-level prefix only in IEEE 1800) or type-checks the abort condition.

### 2.6 System functions

The following are recognized as `DOLLAR_IDENT` by the lexer and parsed as
`CallExpr` (so they can be serialized back), but the **describe** translator
only has templates for a subset (`describe/translator.py:545-554`):

- Templated: `$rose`, `$fell`, `$stable`, `$changed`, `$onehot`, `$onehot0`,
  `$isunknown`, `$countones`.
- Parsed but **without NL template**: `$past`, `$sampled`, `$rewind`,
  `$past_gclk`, `$future_gclk`, `$assertcontrol`, `$asserton`, `$assertoff`,
  `$assertpassoff`, `$assertfailoff`, `$assertpassoncontrol`,
  `$assertfailoncontrol`, `$assertnonvacuouson`, `$assertvacuousoff`,
  assertion system tasks `$error`, `$fatal`, `$warning`, `$info`.

### 2.7 Repetition / delay edge cases

`sva/ast.py::RepeatOperator` covers consecutive `[*]`, non-consecutive `[=]`,
and goto `[->]`. `generate/synthesizer.py:157-189` emits `##[a:$]` and `[a:$]`
forms as strings. Gaps:

- No validation that `a ≤ b` for ranges; the generator can produce
  semantically degenerate `[*3:1]`.
- No recognition of `[+]` (one or more) or `[*]` without bounds in the lexer.
- No `$` (infinity) sentinel token — it is carried as text inside generated
  strings and re-lexed by consumers only approximately.

### 2.8 Unsupported declarations inside properties

Local variables (`local var T name = expr`) are lexed (`local`, `var` are
reserved) and appear in `ast.py::LocalVarDecl`, but:

- Types are stored as raw strings — no validation of `bit`, `logic`,
  `integer`, user-defined, enums, arrays.
- There is no scope tracking, so a local declared in one branch of an
  `if-else` property can collide with another.
- There is no unique-binding check for assertion match items.

Formal arguments (`PropertyFormal`) are carried by name only; no defaults,
directions (`input`/`output`/`inout`), or typed checking.

---

## 3. Robustness weaknesses

### 3.1 Silent exception swallowing (HIGH)

| File | Line | Pattern | Consequence |
|---|---|---|---|
| `formal/service.py` | 45 | `except Exception as exc:` wraps parse | Any parse failure (even `AttributeError`) becomes `SYNTAX_ERROR` with a lossy message |
| `cli/main.py` | 25 | `except Exception as exc: raise click.ClickException(str(exc))` | Tracebacks are discarded; users see only `.args[0]` |
| `timing/core/conditions.py` | 137-146 | `except Exception: pass` then fallback | Emitter failure is indistinguishable from no-op; no telemetry |
| `timing/bridge/from_sva.py` | 578, 1462, 1486, 1517 | bare `except Exception:` | Extraction quietly marks property `LOSSY`/`UNSUPPORTED` without user-visible warning |
| `data/dataset.py` | 130, 135, 141 | `except Exception` → write error key in `metadata` dict only | User sees "dataset built successfully" even when every LLM call failed |
| `data/benchmark.py` | 128, 141 | same pattern | Benchmark aggregates can be computed over a nearly-empty result set |
| `generate/stratified.py` | 512 | `except Exception` | A malformed stratum is dropped without a warning |

Mitigation (recommended):
- Replace broad handlers with the narrowest catch that covers the expected
  failure mode.
- When falling back, always emit a `logging.warning` (or `click.echo(...,
  err=True)` for CLI paths) with the operation name.

### 3.2 Regex-based parsing where a grammar is needed (MEDIUM)

- `timing/frontend/parser.py:34-61` — the DSL is defined by line-anchored regex,
  each line must match a single regex entirely. A trailing comment, a multi-line
  declaration, embedded parentheses, or a missing semicolon fails the whole
  file with no recovery.
- `formal/parse.py:13-14, 103-148` — post-`SvaSyntaxError` fallback locates
  `property`, `disable iff`, and clocking using `re.search` against the raw
  text. Nested `disable iff (cond_with_nested_parens)` works because of the
  explicit paren walker (lines 164-176), but the regex-first step misfires on
  multi-line headers and C-style comments.
- `describe/translator.py` (1430 LOC) implements expression formatting via
  string slicing and token scanning rather than AST visitation in several
  helpers — fragile under comments, string literals, and ternary nesting.
- `sva/lexer.py` **is** a real lexer, but the ecosystem around it treats it as
  "good enough"; downstream fallbacks use regex, defeating the benefit of the
  proper tokenizer.

### 3.3 Hard-coded defaults that silently apply (HIGH)

`formal/parse.py:96-97` and `formal/model.py:51-52, 63-65` inject the literal
defaults `clock_name="clk"`, `clock_edge="posedge"`, `reset_expr="!rst_n"` when
the property text does not name them. `formal/backends/ebmc.py:14-22` embeds
these into the generated checker module:

```
MODULE_TEMPLATE = """module sva_checker(
    input wire {clock_name},
    input wire {reset_name}{signal_ports}
);
    assume property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({antecedent}));
    assert property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({consequent}));
    cover property (@({clock_edge} {clock_name}) disable iff ({reset_expr}) ({antecedent}));
endmodule
"""
```

Risks:
- A design that uses `hclk`/`rst` (active-high) would be checked against
  `posedge clk` with `disable iff (!rst_n)`. EBMC will either report spurious
  `NOT_IMPLIES` (design never de-asserts the synthetic reset) or spurious
  `IMPLIES` (sampling on the wrong clock).
- The **equivalence** and **relationship** checks compose two implications
  (`service.py:60-97`) using the same defaults. Two truly-non-equivalent
  properties can be declared "equivalent" if their defaults happen to neutralize
  each other.
- `_validate_properties` (`ebmc.py:164-181`) rejects any mismatch between
  antecedent/consequent clock or reset strings, but the string comparison is
  syntactic: `!rst_n` and `rst_n == 0` are flagged as a mismatch even though
  they are semantically identical. Fast false negatives.

### 3.4 Generated Verilog is not sanitized (HIGH)

`formal/backends/ebmc.py:131-146, 185-188` splices `FormalProperty.body` and
signal names straight into the module template via `str.format`. No escaping,
no reserved-word check, no identifier validator. Inputs that would break the
generated file:

- A signal named `module`, `endmodule`, `wire`, `input`, `property`.
- A signal name containing a dot (hierarchical path like `u_dut.req`).
- A `disable iff` expression containing `{` or `}` (the `format` call raises
  `IndexError`/`KeyError`).
- Non-ASCII identifiers that EBMC rejects but that pass Python's `\w` regex.

### 3.5 Non-deterministic generation (MEDIUM)

`generate/synthesizer.py` uses `random.random`, `random.choice`, `random.randint`
at 29 call-sites (and `generate/stratified.py`, `generate/utils.py` another
12). No `random.seed(…)` is called anywhere in `src/`, and there is no CLI flag
`--seed` in `cli/main.py` for generate. Consequences:

- Two `sva generate --count N` invocations produce different outputs.
- Dataset builds (`sva data build`) that chain through generation cannot be
  reproduced.
- Regression bugs in the synthesizer cannot be bisected reliably.

### 3.6 Multiprocessing cache and LLM correctness (MEDIUM)

`data/dataset.py:213-217` creates a shared cache directory with
`tempfile.mkdtemp` or the user-supplied `cache_dir`; `_write_cached_result`
(lines 44-51) writes JSON non-atomically (no `os.replace`). With
`use_multiprocessing=True`:

- Two workers hashing the same cache key can race on write, leaving a partial
  file that is later read as truncated JSON.
- There is no cache-version sentinel: if the tool is upgraded between runs,
  stale payloads are silently reused.
- The LLM client is serialized through `_serialize_llm_config` and
  reconstituted per worker (`_materialize_llm_client`); an `api_key` is
  duplicated into every subprocess memory image.

`runtime/llm.py` has no retry, no exponential back-off, no 429/5xx handling,
and no rate-limit token bucket. Any transient error surfaces as
`metadata["svad_error"]` on that row only (`dataset.py:130-131`).

### 3.7 Subprocess handling (MEDIUM)

`runtime/process.py:18-58`:

- `subprocess.run(cmd, capture_output=True, timeout=timeout)` — good.
- No `preexec_fn=os.setsid` / `start_new_session=True`, so on `TimeoutExpired`
  the child (EBMC / VCF / Verible) is terminated but any grand-children it
  spawned are orphaned and continue consuming CPU.
- `FileNotFoundError` is re-raised as `RuntimeError(f"Failed to execute tool:
  {cmd}")` — callers (e.g. `ebmc.py:83`) then wrap it into an `ERROR` result
  that does not distinguish "tool not installed" from "tool crashed".
- `make_work_dir` (line 56) uses `tempfile.mkdtemp` with default
  permissions (`0o700` on Linux, but world-readable `TMPDIR` on some shared
  CI runners). The final `shutil.rmtree(..., ignore_errors=True)`
  (`ebmc.py:129`) swallows cleanup errors — stale proof-obligation files can
  accumulate after crashes.

### 3.8 File I/O is not atomic

Across the package, text output uses `Path.write_text(..., encoding="utf-8")`
without the write-to-temp-then-`os.replace` pattern:

- `cli/main.py:41-52` (`_write_text_output`, `_write_json_output`,
  `_write_jsonl_output`).
- `data/dataset.py:44-51` cache writes.
- `formal/backends/ebmc.py:67` checker SV file.

A Ctrl-C mid-write leaves a truncated file. For dataset builds on large
inputs this is particularly costly.

### 3.9 CLI error reporting is lossy

`cli/main.py:17-30` wraps every uncaught exception as
`click.ClickException(str(exc))`. The chained `__cause__` is preserved but
Click's default formatter only prints the message, so users lose:

- Stack traces for debugging.
- Python's actual exception class (e.g. `SvaSyntaxError` vs `TimeoutError`).
- Exit code differentiation (all non-`ClickException` errors exit `1`).

---

## 4. Per-tool gap matrix

### 4.1 `sva parse`

- Accepts a property surface or file path (`cli/main.py::_load_text_argument`).
- Real tokenizer + recursive-descent parser — the strongest module in the
  toolkit.
- But: see §2 (missing constructs), §2.3 (no comments/strings/directives).
- No multi-property input (one `property { … }` block per invocation).
- `--format json` serializes the AST (`_to_json_compatible` at
  `cli/main.py:73+`) but there is no published JSON schema, so downstream
  consumers must track private dataclass layouts.

### 4.2 `sva formal {check,equivalent,relationship}`

- External dependency: `ebmc` or `vcf` — discovered lazily. When neither is on
  `PATH`, the CLI returns a single sentence `"No formal backend is available"`
  (`service.py:54-56`) with no guidance on install paths.
- Hard-coded clock/reset defaults — see §3.3.
- No handling of multi-clock properties, `bind`s, or `checker`s.
- Counterexample extraction (`ebmc.py:194+`) is a regex slurp of backend text;
  the witness is truncated to 40 lines and is not normalized into a structured
  trace (no signal/time table).
- `service.py:60-91` performs equivalence by running two directional checks
  but does not cache the intermediate SV module, so the EBMC compile cost is
  paid twice for every pair.
- VCF backend (`formal/backends/vcformal.py`, not shown here) mirrors the EBMC
  backend's assumptions; both rely on `check_implication` seeing matched
  clock strings.

### 4.3 `sva timing {validate,render,emit-sva,extract-sva,bundle-sva}`

- DSL parser is regex-based (`frontend/parser.py:34-61`). Any syntax extension
  requires editing regexes and updating `validate.py` in lockstep.
- PNG render requires `cairosvg` as a runtime import. When missing the CLI
  surfaces the raw `ImportError` via the generic `except Exception` in
  `cli/main.py:25`.
- SVA → DSL extraction (`bridge/from_sva.py`) tracks
  `ExtractionStatus.{EXACT, LOSSY, UNSUPPORTED}`, but status is only placed
  inside the returned scenario — the CLI does not bubble LOSSY/UNSUPPORTED up
  as a non-zero exit or a visible warning. Users producing diagrams from SVA
  can easily ship lossy output without realizing it.
- `bundle-sva` groups properties by shared clock/reset; because clock/reset
  can be missing and defaulted (§3.3), unrelated properties can be grouped by
  the shared default.

### 4.4 `sva generate [--validate] [--coverage]`

- Uses unseeded `random` (§3.5).
- `--validate` depends on `verible-verilog-syntax`. When absent, the
  generator still emits modules but flags validity as unknown (no explicit
  user-visible message documenting the flag was ineffective).
- `arith_weight.json` packaged alongside the code
  (`generate/arith_weight.json`) defines operator weights; there is no schema
  or migration for when new operators are added.
- Generated properties are syntactic-only; nothing proves the assertion is
  satisfiable against a real design or even vacuously true.
- No stable naming scheme — property names are generated from counters and
  are not stable across runs, which makes diff-based review of generated
  property banks impractical.

### 4.5 `sva describe {svad,cot}`

- Pure-Python, no external deps.
- `translator.py` contains many template strings for common operators and
  falls through to raw-text emission for `Opaque*` nodes (§2.2).
- No uncertainty signal in the output: a fully understood property and a
  mostly-verbatim passthrough look identical to readers.
- CoT builder (`describe/cot.py`, 375 LOC) assembles natural-language chains
  from pre-baked fragments. There is no guard against
  contradictions when multiple fragments apply.

### 4.6 `sva data {build,benchmark}`

- Multiprocess cache races (§3.6).
- `build` is "offline" only if `llm_client is None`; when the LLM call fails
  mid-run, execution silently falls back to the heuristic translator
  (`dataset.py:122-143`) and continues. The final JSONL mixes LLM-authored
  and heuristic-authored rows distinguishable only by
  `metadata.svad_source`.
- `benchmark` requires `[llm]` extra and `OPENAI_API_KEY` + a model name;
  failure modes when the API key is invalid surface as a generic
  `click.ClickException` (§3.9).
- No guard against accidentally uploading proprietary SVA to a public
  endpoint — `LLMConfig.base_url` defaults to OpenAI unless the user sets it.

---

## 5. Industrial-use risk register

| # | Risk | Trigger scenario | Impact | Where in code |
|---|---|---|---|---|
| R1 | Wrong clock/reset defaults | Property text omits `@(...)` or `disable iff` | **False EBMC/VCF verdicts** used to sign off RTL | `formal/parse.py:96-97`, `formal/model.py:51-65` |
| R2 | Parser silently down-grades SVA | Source uses `nexttime`/`always`/`eventually`/`checker`/`let`/`restrict`/`expect` | Describe/Formal/Timing operate on the string verbatim and report success | `sva/parser.py:91-136`, `describe/translator.py:343-406` |
| R3 | Lexer cannot read real RTL | Input contains `//` comments, string literals, or `` `define `` | Lexer raises on first non-whitespace comment or backtick | `sva/lexer.py:210` |
| R4 | Module template injection | Signal/reset expression contains `{`/`}`/reserved words | `str.format` raises or EBMC compile crashes | `formal/backends/ebmc.py:14-22, 131-146` |
| R5 | Non-deterministic generation | Running `sva generate` twice | Cannot reproduce ML datasets, cannot bisect regressions | `generate/synthesizer.py`, `generate/stratified.py` |
| R6 | Orphaned child processes after timeout | EBMC/VCF spawns helpers and `sva formal` times out | CI boxes accumulate zombie solver processes | `runtime/process.py:26-44` |
| R7 | Silent LLM fallback | `sva data build` with a rate-limited API key | Dataset contains rows marked `translator_fallback` but report "built successfully" | `data/dataset.py:127-136` |
| R8 | Cache corruption under multiprocessing | `sva data build --workers N` with shared `cache_dir` | Partial JSON written, later read fails or returns wrong payload | `data/dataset.py:44-51, 213-217` |
| R9 | Lossy timing extraction, hidden | `sva timing extract-sva` on property with unsupported operator | Diagram is published as if it matched the spec | `timing/bridge/from_sva.py:84-150, 1462-1517` |
| R10 | No preprocessor support | Source file uses `` `include`` or `` `define`` | Parser must receive already-expanded text; user must run a preprocessor by hand | Entire `sva/` package |
| R11 | No vendor-extension support | Source uses VCS `$smashed_*`, Questa/Xcelium checker/let extensions, Jasper `cover` options | Constructs become identifiers or opaque nodes | `sva/lexer.py` |
| R12 | No UVM/OVL integration | Running against a UVM testbench | Nothing parses UVM macros, `uvm_component`, factory registrations | n/a |
| R13 | Counterexample is a regex slurp | EBMC outputs structured `vcd`/trace | Truncated 40-line text blob instead of signal-time table | `formal/backends/ebmc.py:194-210` |
| R14 | Encrypted IP rejected | Input contains `` `protect`` regions | Lexer fails immediately | `sva/lexer.py` |
| R15 | Coverage integrity | `sva generate --coverage` | Coverage is computed over *generated* properties; it does not reflect actual cover directive hits on a real design | `generate/coverage.py` |
| R16 | No performance ceiling | Properties ≥ 10k tokens or files ≥ 10 MB | Everything is in-memory; no streaming, no profiling budget | all parsers |
| R17 | Exit codes are coarse | Tool missing, parse error, and timeout all become exit 1 | CI cannot discriminate fatal from retryable | `cli/main.py:17-30` |
| R18 | No audit trail | Formal "implies" verdict | Verdict has no hash of inputs, no backend version, no timestamp recorded | `formal/model.py::CheckResult` |

---

## 6. Test-coverage gaps

Observed test directories vs. suggested additions (the latter are **not**
present today):

- `tests/sva/test_parser.py`, `test_emitter.py`, `test_roundtrip.py`,
  `test_visitors.py` — cover happy-path constructs.
- **Missing**: adversarial suite for each unsupported operator
  (§2.1), comment/string/preprocessor inputs (§2.3), malformed
  Unicode, extremely deep nesting, fuzz corpus.
- `tests/formal/test_service.py`, `test_parse.py` — assume backends exist and
  fake their outputs.
- **Missing**: tool-missing path, timeout path, orphan-kill verification,
  mismatched-clock reproducibility tests, counterexample-extraction
  tests against real EBMC stdout samples.
- `tests/integration/test_cli_*.py` — exercise CLIs end-to-end.
- **Missing**: determinism test for `sva generate` (should fail today), race
  test for `sva data build --workers 8`, large-file smoke test,
  regression for each risk in §5.

No coverage configuration (`coveragerc`, pytest-cov invocation in
`pyproject.toml`) is present in the repo to quantify current coverage.

---

## 7. Recommendations, ranked by ROI

1. **Surface silent fallbacks.** Every opaque down-grade (§2.2), every
   `translator_fallback` (§3.1, §3.6), every LOSSY extraction (§4.3) should
   emit a `logging.warning` and be reflected in the CLI exit code. This alone
   turns several Severity-High risks (R2, R7, R9) into Severity-Low.
2. **Replace hard-coded clock/reset defaults with a mandatory annotation.**
   In `formal/parse.py` and `formal/model.py`, require the caller to provide
   `clock_name`/`reset_expr` or raise. Expose `--clock/--reset` flags on
   `sva formal`. Eliminates R1.
3. **Harden the lexer.** Add `//` and `/* */` comment skipping, string
   literals, and at minimum ignore `` `define``, `` `include``, `` `ifdef``
   by passing input through a preprocessor or by recording them as trivia.
   Addresses R3, partially R10, R14.
4. **Seedable generation.** Thread an `rng: random.Random` instance through
   `SVASynthesizer` and expose `sva generate --seed`. Eliminates R5.
5. **Sanitize module generation.** Validate identifiers against
   SystemVerilog reserved words before splicing into the EBMC template; use
   `str.Template` or explicit `re.sub` of `{`/`}`. Addresses R4.
6. **Start a new session group for subprocesses.** Pass
   `start_new_session=True` in `runtime/process.py` and `os.killpg` on
   timeout. Addresses R6.
7. **Atomic writes.** Write to `{path}.tmp` and `os.replace`. Addresses
   corruption risks across CLI outputs and the dataset cache.
8. **Per-worker cache locking.** Guard `_write_cached_result` with
   `filelock`/`fcntl.flock`; add a cache schema version. Addresses R8.
9. **Expand syntax coverage, but document what is missing first.** Track a
   table in `docs/` (this file) enumerating every unsupported construct,
   so users can file precise issues instead of discovering gaps post-silicon.
10. **Structured counterexamples.** Parse EBMC's trace format (or at least
    pass through its path) and expose it as a typed object on `CheckResult`.
    Addresses R13 and makes R18 partially actionable.

---

## 8. Recommended usage posture

| Use case | Safe today? |
|---|---|
| Research on SVA parsing and AST transformations | Yes, with awareness of §2. |
| Teaching / coursework on SVA semantics | Yes, with hand-checked examples. |
| ML dataset construction for SVA tasks | Yes, if the user accepts non-determinism and logs `metadata.svad_source`. |
| Formal equivalence checks used for sign-off | **No.** R1, R4, R13, R18 block regulated use. |
| Generating assertion banks for production RTL | **No.** R5, R15. |
| Parsing checked-in SVA from a real project | **No.** R3, R10, R11, R14. |
| Visualizing timing diagrams in design reviews | Caveat: R9 — validate extraction status before publishing. |
| CI-integrated assertion mining | **No.** R6, R8, R17. |

The toolkit is a capable prototype that is one hardening pass away from being
dependable for internal tooling; it is several passes away from being
dependable as part of a sign-off flow.
