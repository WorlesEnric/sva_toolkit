"""EBMC witness synthesis for timing scenario extraction.

Replaces the CEG greedy longest-path tick assignment with a concrete witness
trace produced by EBMC's cover property engine.  The CEG path remains the
silent fallback when EBMC is unavailable or times out.

EBMC 5.x interface notes:
  - No ``--cover`` flag; just run with ``--trace``.
  - Each ``Transition system state N`` in the output is one clock cycle
    (EBMC models clocked properties implicitly — no half-cycle steps).
  - Signal names in the trace are prefixed: ``sva_witness.signal``.
  - Return code 0 = all properties proved/covered; rc != 0 = failure.
    We parse the trace regardless and fall back to None if no states found.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

from sva_toolkit.formal.model import FormalProperty
from sva_toolkit.runtime.process import make_work_dir, run_tool
from sva_toolkit.timing.core.conditions import Condition
from sva_toolkit.timing.core.scenario import Anchor, ScenarioDocument


# Module-level defaults — can be patched by the CLI before extraction runs.
_DEFAULT_DEPTH: int = 32
_DEFAULT_TIMEOUT: int = 60

_MODULE_NAME = "sva_witness"


# ---------------------------------------------------------------------------
# Witness trace data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WitnessTrace:
    """Concrete trace produced by EBMC witness synthesis.

    ``signal_values[sig][k]`` is the value of *sig* at the k-th clock cycle.
    All lists have the same length ``ticks``.
    """

    ticks: int
    signal_values: dict[str, list[str]]
    raw_states: int


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

# Common SVA keywords to skip from signal extraction for ports
_SVA_RESERVED = {
    "accept_on", "always", "and", "assert", "assume", "begin", "bit", "cover",
    "disable", "edge", "else", "end", "endproperty", "endsequence", "fell",
    "forall", "high", "if", "iff", "initial", "input", "intersect", "local",
    "logic", "low", "module", "negedge", "not", "or", "output", "posedge",
    "property", "reg", "reject_on", "rose", "sequence", "stable", "strong",
    "sync_accept_on", "sync_reject_on", "throughout", "until", "until_with",
    "var", "wait", "weak", "wire", "within"
}


class EbmcWitnessSynthesizer:
    """Run EBMC to obtain a witness trace from SVA properties.

    Approach:
      1. Convert each property body to a **cover sequence** (not an
         implication) so EBMC actually finds a trace where the trigger fires
         and the response is reached.
      2. Add ``assume property (rst_n)`` so reset is deasserted throughout,
         preventing trivial vacuous witnesses.
      3. Run ``ebmc --bound N --trace`` and parse
         ``Transition system state N`` blocks.
    """

    _WITNESS_MODULE_TEMPLATE = """\
module {module_name}(
    input wire {clock_name},
    input wire {reset_name}{signal_ports}
);
    // Hold reset deasserted so cover conditions require real behaviour
    assume property (@({clock_edge} {clock_name}) {reset_high});
{cover_lines}endmodule
"""

    def __init__(
        self,
        tool_path: str | None = None,
        depth: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.tool_path = tool_path or shutil.which("ebmc")
        self.depth = depth if depth is not None else _DEFAULT_DEPTH
        self.timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT

    @property
    def available(self) -> bool:
        return self.tool_path is not None

    def synthesize(
        self,
        prop: FormalProperty,
        *,
        signal_widths: dict[str, int] | None = None,
    ) -> WitnessTrace | None:
        """Synthesize a witness for a single SVA property."""
        return self._run([prop], signal_widths=signal_widths or {})

    def synthesize_joint(
        self,
        props: list[FormalProperty],
        *,
        causal_order: list[tuple[int, int]] | None = None,
        signal_widths: dict[str, int] | None = None,
    ) -> WitnessTrace | None:
        """Synthesize a joint witness covering all properties in causal order.

        The chained cover sequence is built by joining each property's derived
        cover sequence with ``##[1:$]``, giving EBMC freedom to place any
        number of cycles between phases.
        """
        if not props:
            return None
        if len(props) == 1:
            return self.synthesize(props[0], signal_widths=signal_widths)

        ordered_indices = _topological_sort(len(props), causal_order or [])
        ordered_props = [props[i] for i in ordered_indices]

        segments = [f"({_extract_cover_sequence(p.body)})" for p in ordered_props]
        chained_body = " ##[1:$] ".join(segments)

        primary = ordered_props[0]
        all_signals: frozenset[str] = frozenset().union(*(p.signals for p in ordered_props))
        chained_prop = FormalProperty(
            body=chained_body,
            clock_edge=primary.clock_edge,
            clock_name=primary.clock_name,
            reset_expr=primary.reset_expr,
            signals=all_signals,
        )

        # Chained property dominates trace length. We only need to run this one;
        # individual firing ticks are recovered by the evaluator from the trace.
        return self._run([chained_prop], signal_widths=signal_widths or {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        props: list[FormalProperty],
        signal_widths: dict[str, int],
    ) -> WitnessTrace | None:
        if not self.available or not props:
            return None

        module_text = self._build_module(props, signal_widths)
        work_dir = Path(make_work_dir(prefix="sva_witness_"))
        sv_file = work_dir / f"{_MODULE_NAME}.sv"
        sv_file.write_text(module_text, encoding="utf-8")

        try:
            result = run_tool(
                [
                    self.tool_path,
                    "--bound", str(self.depth),
                    "--top", _MODULE_NAME,
                    "--trace",
                    str(sv_file),
                ],
                cwd=work_dir,
                timeout=self.timeout,
            )
        except RuntimeError:
            return None
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        if result.timed_out:
            return None

        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        return _parse_witness_text(combined, module_prefix=_MODULE_NAME)

    def _build_module(
        self,
        props: list[FormalProperty],
        signal_widths: dict[str, int],
    ) -> str:
        primary = props[0]
        
        # Collect ALL identifiers from ALL property bodies to ensure everything is a port
        all_idents: set[str] = set()
        for p in props:
            all_idents |= {m for m in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", p.body)}
            # Also include signals explicitly listed in FormalProperty (just in case)
            all_idents |= set(p.signals)

        # Subtract reserved keywords and clock/reset
        all_sigs = all_idents - _SVA_RESERVED
        all_sigs -= {primary.clock_name, primary.reset_name}

        signal_ports = []
        for s in sorted(all_sigs):
            width = signal_widths.get(s)
            # Default to 8-bit for symbolic parameters (uppercase)
            if width is None and s[0].isupper():
                width = 8
            signal_ports.append(f",\n    input wire{_width_prefix(width)} {s}")

        # reset_high: the expression that means "reset is deasserted"
        reset_expr = primary.reset_expr.strip()
        if reset_expr.startswith(("!", "~")):
            reset_high = primary.reset_name  # !rst_n → rst_n high = deasserted
        else:
            reset_high = f"!{primary.reset_name}"

        cover_lines = []
        for p in props:
            seq = _extract_cover_sequence(p.body)
            cover_lines.append(
                f"    cover property (@({p.clock_edge} {p.clock_name})"
                f" disable iff ({p.reset_expr}) ({seq}));"
            )

        return self._WITNESS_MODULE_TEMPLATE.format(
            module_name=_MODULE_NAME,
            clock_name=primary.clock_name,
            clock_edge=primary.clock_edge,
            reset_name=primary.reset_name,
            reset_high=reset_high,
            signal_ports="".join(signal_ports),
            cover_lines="\n".join(cover_lines) + "\n",
        )


# ---------------------------------------------------------------------------
# Cover sequence extraction
# ---------------------------------------------------------------------------

def _extract_cover_sequence(body: str) -> str:
    """Convert a property body into a cover sequence for witness synthesis.

    Transformations:
      ``A |-> ##N B``          →  ``A ##N B``
      ``A |=> B``              →  ``A ##1 B``
      ``A |-> B until_with C`` →  ``A ##[0:$] C``
      ``A |-> B``              →  ``A ##0 B``
      ``A until B``            →  ``##[0:$] B``
      ``A until_with B``       →  ``##[0:$] B``
      anything else            →  body unchanged

    Note: Symbolic delays like ``##PARAM`` or ``##[0:MAX]`` are converted
    to unbounded delays (``##$`` or ``##[0:$]``) as EBMC requires constant
    bounds for its sequence engine.
    """
    body = body.strip()

    # Handle implications
    for op_str, default_delay in ((" |-> ", "##0"), (" |=> ", "##1")):
        idx = body.find(op_str)
        if idx < 0:
            continue
        ant = body[:idx].strip()
        cons = body[idx + len(op_str):]

        # until_with / until in consequent: use terminal condition
        uw = re.search(r"\buntil(_with)?\b", cons, re.IGNORECASE)
        if uw:
            terminal = cons[uw.end():].strip()
            return f"({ant}) ##[0:$] ({terminal})"

        # Explicit delay in consequent: ##N expr or ##[A:B] expr
        m = re.match(r"##(\[([^\]]+)\]|([A-Za-z0-9_]+))\s+(.*)", cons.strip(), re.DOTALL)
        if m:
            range_text = m.group(1)
            # If it's a range [A:B], convert any non-literal/non-digit to $
            if range_text.startswith("[") and range_text.endswith("]"):
                inner = m.group(2)
                if ":" in inner:
                    parts = inner.split(":")
                    new_parts = []
                    for p in parts:
                        p = p.strip()
                        if p.isdigit() or p == "$":
                            new_parts.append(p)
                        else:
                            new_parts.append("$")
                    range_text = f"[{':'.join(new_parts)}]"
                else:
                    if not inner.strip().isdigit() and inner.strip() != "$":
                        range_text = "$"
            else:
                if not range_text.isdigit() and range_text != "$":
                    range_text = "$"

            return f"({ant}) ##{range_text} ({m.group(4).strip()})"

        # Plain consequent
        return f"({ant}) {default_delay} ({cons.strip()})"

    # Handle non-implication until/until_with
    uw = re.search(r"\buntil(_with)?\b", body, re.IGNORECASE)
    if uw:
        terminal = body[uw.end():].strip()
        return f"##[0:$] ({terminal})"

    return body


# ---------------------------------------------------------------------------
# Trace parser
# ---------------------------------------------------------------------------

# EBMC 5.x format: "Transition system state N"
_STATE_RE = re.compile(r"Transition system state\s+(\d+)", re.IGNORECASE)
# Also accept plain "State N" for compatibility with other EBMC versions
_STATE_RE_PLAIN = re.compile(r"(?<!\w)State\s+(\d+)", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*([^\s,;\n]+)")


def _parse_witness_text(
    output: str,
    module_prefix: str = "sva_witness",
) -> WitnessTrace | None:
    """Parse EBMC text trace output into a WitnessTrace.

    EBMC 5.x emits one ``Transition system state N`` block per clock cycle,
    with signal names prefixed by the module name.  Each block = one cycle.
    """
    # Try primary format first, then plain fallback
    state_starts: list[tuple[int, int]] = [
        (int(m.group(1)), m.start()) for m in _STATE_RE.finditer(output)
    ]
    if not state_starts:
        state_starts = [
            (int(m.group(1)), m.start()) for m in _STATE_RE_PLAIN.finditer(output)
        ]
    if not state_starts:
        return None

    prefix = f"{module_prefix}."

    raw_states: list[dict[str, str]] = []
    for i, (_num, start) in enumerate(state_starts):
        end = state_starts[i + 1][1] if i + 1 < len(state_starts) else len(output)
        block = output[start:end]
        assignments: dict[str, str] = {}
        for m in _ASSIGN_RE.finditer(block):
            raw_name = m.group(1)
            value = m.group(2)
            # Strip module prefix and any sub-hierarchy
            if raw_name.startswith(prefix):
                name = raw_name[len(prefix):]
            else:
                name = raw_name
            # Skip EBMC meta-names that contain dots (sub-hierarchy) or are "state"
            if "." in name or name.lower() == "state":
                continue
            # Normalize EBMC unconstrained marker to standard don't-care
            assignments[name] = "x" if value == "?" else value
        raw_states.append(assignments)

    if not raw_states:
        return None

    all_sigs: set[str] = set()
    for state in raw_states:
        all_sigs |= state.keys()

    signal_values: dict[str, list[str]] = {sig: [] for sig in all_sigs}
    for state in raw_states:
        for sig in all_sigs:
            signal_values[sig].append(state.get(sig, "0"))

    return WitnessTrace(
        ticks=len(raw_states),
        signal_values=signal_values,
        raw_states=len(raw_states),
    )


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

def eval_condition(condition: Condition, trace: WitnessTrace, clk_tick: int) -> bool:
    """Return True if *condition* is satisfied at clock cycle *clk_tick*."""
    kind = condition.kind
    if kind == "predicate":
        return _eval_predicate(condition.predicate, trace, clk_tick)
    if kind == "all":
        return all(eval_condition(c, trace, clk_tick) for c in condition.items)
    if kind == "any":
        return any(eval_condition(c, trace, clk_tick) for c in condition.items)
    if kind == "not" and condition.items:
        return not eval_condition(condition.items[0], trace, clk_tick)
    return True  # "raw" or unrecognised — optimistically pass


def _eval_predicate(predicate, trace: WitnessTrace, clk_tick: int) -> bool:
    if predicate is None:
        return True
    op = predicate.op
    sig = predicate.signal
    if not sig or sig not in trace.signal_values:
        return True  # unknown signal — don't filter

    values = trace.signal_values[sig]
    if clk_tick >= len(values):
        return False

    curr = values[clk_tick]
    # Assume signals were '0' before the trace started for consistent rise/fall
    prev = values[clk_tick - 1] if clk_tick > 0 else "0"

    if op == "high":
        return _is_high(curr)
    if op == "low":
        return _is_low(curr)
    if op == "rise":
        return _is_low(prev) and _is_high(curr)
    if op == "fall":
        return _is_high(prev) and _is_low(curr)
    if op == "stable":
        return _normalize_value(curr) == _normalize_value(prev)
    if op in ("eq", "neq"):
        val = predicate.value or "0"
        eq = _normalize_value(curr) == _normalize_value(val)
        return eq if op == "eq" else not eq
    return True  # unknown op — pass


def find_anchor_tick(anchor: Anchor, trace: WitnessTrace) -> int | None:
    """Return the first clock cycle index where the anchor condition is satisfied."""
    for t in range(trace.ticks):
        if eval_condition(anchor.condition, trace, t):
            return t
    return None


# ---------------------------------------------------------------------------
# Document refiner
# ---------------------------------------------------------------------------

def refine_document_from_witness(
    document: ScenarioDocument,
    trace: WitnessTrace,
) -> ScenarioDocument:
    """Return a refined ScenarioDocument with anchor ticks and signal samples from trace."""
    clock_sig = document.clocking.signal

    new_anchors = []
    for anchor in document.anchors:
        tick = find_anchor_tick(anchor, trace)
        if tick is not None:
            new_anchors.append(replace(anchor, absolute_tick=tick))
        else:
            new_anchors.append(anchor)

    new_signals = []
    for sig in document.signals:
        if sig.name == clock_sig:
            # Rebuild clock as 0 1 0 1 ...
            samples = tuple(str(t % 2) for t in range(trace.ticks))
            new_signals.append(replace(sig, samples=samples))
        elif sig.name in trace.signal_values:
            raw = trace.signal_values[sig.name]
            new_signals.append(replace(sig, samples=tuple(raw[:trace.ticks])))
        else:
            new_signals.append(sig)

    ticks = document.ticks if document.ticks is not None else trace.ticks

    return replace(
        document,
        anchors=tuple(new_anchors),
        signals=tuple(new_signals),
        ticks=ticks,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_high(value: str) -> bool:
    v = value.strip().lower()
    if v in ("0", "1'b0", "0b0", "0x0", "false", "", "x", "z"):
        return False
    if v in ("1", "1'b1", "true"):
        return True
    try:
        return int(v, 0) != 0
    except ValueError:
        return True  # unknown non-zero string — treat as high


def _is_low(value: str) -> bool:
    return not _is_high(value)


def _normalize_value(v: str) -> str:
    v = v.strip().lower()
    try:
        return str(int(v, 0))
    except ValueError:
        return v


def _width_prefix(width: int | None) -> str:
    if width and width > 1:
        return f" [{width - 1}:0]"
    return ""


def _topological_sort(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Kahn's topological sort; appends remaining nodes for cycles/disconnected."""
    in_deg = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for src, dst in edges:
        if 0 <= src < n and 0 <= dst < n:
            adj[src].append(dst)
            in_deg[dst] += 1
    queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    order.extend(sorted(set(range(n)) - set(order)))
    return order
