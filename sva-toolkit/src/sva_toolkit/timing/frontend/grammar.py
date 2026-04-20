"""Grammar-based timing DSL parser with source-aware syntax diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Sequence

from sva_toolkit.timing.core.conditions import Condition, Predicate, parse_dsl_condition
from sva_toolkit.timing.core.scenario import (
    Anchor,
    AnchorRole,
    ClockingSpec,
    ConstraintRegion,
    Cut,
    CutMeaning,
    CutPlacement,
    ExtractionStatus,
    LaneConstraint,
    ParameterDecl,
    PropertyOverlay,
    ScenarioDocument,
    SignalDecl,
    SignalKind,
    TimeBound,
    TimeWindow,
    WindowBoundKind,
    normalize_signal_width,
)
from sva_toolkit.timing.errors import TimingDslError, TimingSyntaxError
from sva_toolkit.timing.frontend.validate import validate_diagram


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_IDENT_RE = re.compile(rf"^{IDENT}$")
_PROPERTY_NOTE_RE = re.compile(rf"^\s*({IDENT})\s*:\s*(.+)$")
_NOT_BEFORE_RE = re.compile(rf"^not\s+({IDENT})\s+before\s+({IDENT})$")
_RESPONSE_RE = re.compile(
    rf"^({IDENT})\s*->\s*after\s*\[\s*({IDENT}|[0-9]+)\s*:\s*({IDENT}|[0-9]+)\s*\]\s*({IDENT})$"
)
_HOLD_UNTIL_RE = re.compile(rf"^(.+)\s+from\s+({IDENT})\s+until\s+({IDENT})$")
_SIGNAL_ASSIGN_RE = re.compile(rf"^({IDENT})\s*=\s*(.+)$")


@dataclass(frozen=True)
class TdToken:
    """Single token emitted by the timing DSL lexer."""

    kind: str
    text: str
    start: int
    end: int
    line: int
    column: int


class TdLexer:
    """Turn timing DSL source text into a compact token stream."""

    _PUNCTUATION = {
        "{": "LBRACE",
        "}": "RBRACE",
        "[": "LBRACKET",
        "]": "RBRACKET",
        "(": "LPAREN",
        ")": "RPAREN",
        ":": "COLON",
        ";": "SEMI",
        "=": "EQUAL",
        ",": "COMMA",
    }

    def __init__(self, source: str) -> None:
        self.source = source
        self.index = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> list[TdToken]:
        tokens: list[TdToken] = []
        line_has_code = False

        while not self._at_end():
            char = self.source[self.index]

            if char in " \t\r\f\v":
                self._advance()
                continue
            if char == "\n":
                self._advance()
                line_has_code = False
                continue
            if char == "/" and self._peek(1) == "/":
                comment_start = self._mark()
                self._advance(2)
                comment_text = self._read_until_newline()
                if not line_has_code:
                    stripped = comment_text.strip()
                    if _PROPERTY_NOTE_RE.match(stripped):
                        tokens.append(self._token("NOTE", stripped, comment_start))
                continue
            if self._starts_hash_comment():
                self._advance()
                self._read_until_newline()
                continue
            if char in self._PUNCTUATION:
                mark = self._mark()
                self._advance()
                tokens.append(self._token(self._PUNCTUATION[char], char, mark))
                line_has_code = True
                continue
            if char == '"':
                tokens.append(self._read_string())
                line_has_code = True
                continue

            tokens.append(self._read_atom())
            line_has_code = True

        eof_mark = self._mark()
        tokens.append(self._token("EOF", "", eof_mark))
        return tokens

    def _read_atom(self) -> TdToken:
        mark = self._mark()
        start = self.index
        while not self._at_end():
            char = self.source[self.index]
            if char.isspace() or char in self._PUNCTUATION or char == '"':
                break
            if self._starts_hash_comment():
                break
            if char == "/" and self._peek(1) == "/":
                break
            self._advance()
        return self._token("ATOM", self.source[start:self.index], mark)

    def _read_string(self) -> TdToken:
        mark = self._mark()
        quote = self.source[self.index]
        start = self.index
        self._advance()
        escaped = False
        while not self._at_end():
            char = self.source[self.index]
            if escaped:
                escaped = False
                self._advance()
                continue
            if char == "\\":
                escaped = True
                self._advance()
                continue
            if char == quote:
                self._advance()
                return self._token("STRING", self.source[start:self.index], mark)
            if char == "\n":
                raise TimingSyntaxError(start, "unterminated string literal", self.source)
            self._advance()
        raise TimingSyntaxError(start, "unterminated string literal", self.source)

    def _read_until_newline(self) -> str:
        start = self.index
        while not self._at_end() and self.source[self.index] != "\n":
            self._advance()
        return self.source[start:self.index]

    def _token(self, kind: str, text: str, mark: tuple[int, int, int]) -> TdToken:
        start, line, column = mark
        return TdToken(kind=kind, text=text, start=start, end=self.index, line=line, column=column)

    def _mark(self) -> tuple[int, int, int]:
        return (self.index, self.line, self.column)

    def _advance(self, count: int = 1) -> None:
        for _ in range(count):
            if self._at_end():
                return
            char = self.source[self.index]
            self.index += 1
            if char == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1

    def _peek(self, offset: int) -> str:
        target = self.index + offset
        if target >= len(self.source):
            return ""
        return self.source[target]

    def _starts_hash_comment(self) -> bool:
        if self.source[self.index] != "#":
            return False
        if self._peek(1) == "#":
            return False
        if self.index == 0:
            return True
        return self.source[self.index - 1].isspace()

    def _at_end(self) -> bool:
        return self.index >= len(self.source)


class TdParser:
    """Recursive-descent parser for the timing DSL."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = TdLexer(source).tokenize()
        self.index = 0

    def parse(self) -> ScenarioDocument:
        while self._match_kind("NOTE"):
            pass
        if self._peek().kind == "EOF":
            raise TimingDslError("diagram text is empty")

        self._expect_atom("diagram", "diagram must start with 'diagram <name> {'")
        name = self._expect_identifier("expected diagram name").text
        self._expect("{", "diagram must start with 'diagram <name> {'")

        ticks = None
        clocking = None
        params: list[ParameterDecl] = []
        signals: list[SignalDecl] = []
        anchors: list[Anchor] = []
        windows: list[TimeWindow] = []
        cuts: list[Cut] = []
        lane_constraints: list[LaneConstraint] = []
        properties: list[PropertyOverlay] = []

        while True:
            token = self._peek()
            if token.kind == "EOF":
                raise TimingDslError("diagram is missing closing '}'")
            if self._match("}"):
                break
            if note := self._match_kind("NOTE"):
                self._attach_property_note(properties, note.text)
                continue

            keyword = token.text
            if keyword == "clock":
                clocking = self._parse_clock()
                continue
            if keyword == "disable":
                if clocking is None:
                    raise TimingDslError("disable iff must appear after clock")
                clocking = replace(clocking, disable_iff=self._parse_disable())
                continue
            if keyword == "ticks":
                ticks = self._parse_ticks()
                continue
            if keyword == "params":
                params.extend(self._parse_params_block())
                continue
            if keyword == "param":
                params.append(self._parse_param_decl(require_keyword=True))
                continue
            if keyword == "lane":
                signals.append(self._parse_lane())
                continue
            if keyword in {"anchor", "event"}:
                anchors.append(self._parse_anchor())
                continue
            if keyword == "window":
                windows.append(self._parse_window())
                continue
            if keyword == "cut":
                cuts.append(self._parse_cut())
                continue
            if keyword == "show":
                lane_constraints.extend(self._parse_show(len(lane_constraints)))
                continue
            if keyword == "property":
                properties.append(self._parse_property())
                continue
            if keyword == "rule":
                rule_windows, rule_constraints, property_overlay = self._parse_rule()
                windows.extend(rule_windows)
                lane_constraints.extend(rule_constraints)
                properties.append(property_overlay)
                continue
            raise self._syntax_error(token, f"unrecognized statement: {token.text}")

        if clocking is None:
            raise TimingDslError("diagram is missing clock declaration")

        scenario = ScenarioDocument(
            name=name,
            clocking=clocking,
            params=tuple(params),
            signals=tuple(signals),
            anchors=tuple(anchors),
            windows=tuple(windows),
            cuts=tuple(cuts),
            lane_constraints=tuple(lane_constraints),
            properties=tuple(_link_property_references(properties, anchors, windows, lane_constraints)),
            ticks=ticks,
        )
        validate_diagram(scenario)
        return scenario

    def _parse_clock(self) -> ClockingSpec:
        self._expect_atom("clock")
        edge = self._expect_one_of({"posedge", "negedge"}, "expected 'posedge' or 'negedge'").text
        signal = self._expect_identifier("expected clock signal name").text
        self._expect(";")
        return ClockingSpec(edge=edge, signal=signal)

    def _parse_disable(self) -> str:
        self._expect_atom("disable")
        self._expect_atom("iff", "disable iff must start with 'disable iff'")
        return self._read_segment_until_semicolon()

    def _parse_ticks(self) -> int:
        self._expect_atom("ticks")
        count_token = self._expect_atom_matching(str.isdigit, "expected tick count")
        self._expect(";")
        return int(count_token.text)

    def _parse_params_block(self) -> list[ParameterDecl]:
        self._expect_atom("params")
        if not self._match("{"):
            raise TimingDslError("params block must use 'params {'")

        params: list[ParameterDecl] = []
        while True:
            if self._peek().kind == "EOF":
                raise TimingDslError("unterminated params block")
            if self._match("}"):
                return params
            params.append(self._parse_param_decl(require_keyword=False))

    def _parse_param_decl(self, *, require_keyword: bool) -> ParameterDecl:
        if require_keyword:
            self._expect_atom("param")
        elif self._peek().text == "param":
            self._advance()

        name = self._expect_identifier("expected parameter name").text
        kind = "int"
        if self._match(":"):
            kind = self._expect_identifier("expected parameter type").text
        self._expect(";")
        return ParameterDecl(name=name, kind=kind)

    def _parse_lane(self) -> SignalDecl:
        self._expect_atom("lane")
        signal_name = self._expect_identifier("expected lane name").text
        self._expect(":")
        signal_kind_text = self._expect_one_of({"bit", "bus"}, "expected 'bit' or 'bus'").text
        signal_kind = SignalKind(signal_kind_text)

        width = None
        if self._match("["):
            width = self._expect_atom_matching(_is_ident_or_int, "expected lane width").text
            self._expect("]")

        if self._match("="):
            sample_text = self._read_segment_until_semicolon()
            samples = tuple(sample_text.split())
        else:
            self._expect(";")
            samples = ()

        return SignalDecl(
            name=signal_name,
            kind=signal_kind,
            width=normalize_signal_width(signal_kind, width),
            samples=samples,
        )

    def _parse_anchor(self) -> Anchor:
        keyword = self._expect_one_of({"anchor", "event"}, "expected 'anchor' or 'event'").text
        anchor_name = self._expect_identifier("expected anchor name").text
        self._expect("=")
        expr_text = self._read_segment_until_semicolon().removesuffix(" same_cycle").strip()
        condition = parse_dsl_condition(expr_text)
        return Anchor(
            name=anchor_name,
            condition=condition,
            role=AnchorRole.SYNTHETIC if keyword == "event" else AnchorRole.STATE,
        )

    def _parse_window(self) -> TimeWindow:
        self._expect_atom("window")
        window_name = self._expect_identifier("expected window name").text
        self._expect("=")
        self._expect_atom("between", "window must use 'between <anchor> and <anchor> <bound>'")
        start_anchor = self._expect_identifier("expected window start anchor").text
        self._expect_atom("and", "window must use 'between <anchor> and <anchor> <bound>'")
        end_anchor = self._expect_identifier("expected window end anchor").text
        bound_text = self._read_segment_until_semicolon()
        return TimeWindow(
            name=window_name,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            bound=_parse_window_bound(bound_text),
        )

    def _parse_cut(self) -> Cut:
        self._expect_atom("cut")
        cut_name = self._expect_identifier("expected cut name").text
        self._expect("=")

        if self._match_atom("before"):
            anchor_name = self._expect_identifier("expected anchor name").text
            meaning = _parse_cut_meaning(self._expect_one_of({"omitted", "compressed", "lookback"}).text, placement="before")
            label = self._parse_optional_label()
            self._expect(";")
            return Cut(
                name=cut_name,
                placement=CutPlacement.BEFORE_ANCHOR,
                meaning=meaning,
                anchor=anchor_name,
                label=label,
            )

        if self._match_atom("after"):
            anchor_name = self._expect_identifier("expected anchor name").text
            meaning = _parse_cut_meaning(self._expect_one_of({"omitted", "compressed", "lookback"}).text, placement="after")
            label = self._parse_optional_label()
            self._expect(";")
            return Cut(
                name=cut_name,
                placement=CutPlacement.AFTER_ANCHOR,
                meaning=meaning,
                anchor=anchor_name,
                label=label,
            )

        if self._match_atom("between"):
            left_window = self._expect_identifier("expected window name").text
            self._expect_atom("and", "cut window placement must use 'between <window> and <window>'")
            right_window = self._expect_identifier("expected window name").text
            meaning = _parse_cut_meaning(
                self._expect_one_of({"omitted", "compressed", "lookback"}).text,
                placement="between",
            )
            label = self._parse_optional_label()
            self._expect(";")
            return Cut(
                name=cut_name,
                placement=CutPlacement.BETWEEN_WINDOWS,
                meaning=meaning,
                left_window=left_window,
                right_window=right_window,
                label=label,
            )

        raise self._syntax_error(self._peek(), "cut must use before/after/between placement")

    def _parse_optional_label(self) -> str | None:
        if not self._match_atom("label"):
            return None
        start = self._peek()
        if start.kind == "STRING":
            self._advance()
            return _strip_optional_quotes(start.text)
        label_text = self._read_segment_until_semicolon(consume_semicolon=False)
        return _strip_optional_quotes(label_text)

    def _parse_show(self, sequence: int) -> list[LaneConstraint]:
        self._expect_atom("show")
        body = self._read_segment_until_semicolon()
        return _parse_show_clause(body, sequence)

    def _parse_property(self) -> PropertyOverlay:
        self._expect_atom("property")
        property_name = self._expect_identifier("expected property name").text
        status = ExtractionStatus.EXACT
        if self._match("["):
            status_token = self._expect_one_of({"exact", "lossy", "unsupported"}, "expected property status")
            status = _parse_status(status_token.text)
            self._expect("]")
        self._expect(":")
        body = self._read_segment_until_semicolon()
        return PropertyOverlay(name=property_name, body=body, status=status)

    def _parse_rule(self) -> tuple[list[TimeWindow], list[LaneConstraint], PropertyOverlay]:
        self._expect_atom("rule")
        rule_name = self._expect_identifier("expected rule name").text
        self._expect(":")
        rule_body = self._read_segment_until_semicolon()
        return _parse_legacy_rule(rule_name, rule_body)

    def _read_segment_until_semicolon(self, *, consume_semicolon: bool = True) -> str:
        if self._peek().text == ";":
            if consume_semicolon:
                self._advance()
            return ""

        start_token = self._peek()
        stack: list[TdToken] = []
        index = self.index

        while index < len(self.tokens):
            token = self.tokens[index]
            if token.kind == "EOF" or (token.text == "}" and not stack):
                if stack:
                    opener = stack[-1]
                    expected = _closing_delimiter(opener.text)
                    raise TimingSyntaxError(opener.start, f"expected '{expected}' to close '{opener.text}'", self.source)
                raise self._syntax_error(token, "statement body must end with ';'")
            if not stack and token.text == ";":
                segment = _normalize_fragment(self.source[start_token.start : token.start])
                self.index = index + 1 if consume_semicolon else index
                return segment
            if token.text in {"(", "[", "{"}:
                stack.append(token)
            elif token.text in {")", "]", "}"}:
                if not stack:
                    raise TimingSyntaxError(token.start, f"unexpected '{token.text}'", self.source)
                opener = stack.pop()
                if _closing_delimiter(opener.text) != token.text:
                    expected = _closing_delimiter(opener.text)
                    raise TimingSyntaxError(token.start, f"expected '{expected}' to close '{opener.text}'", self.source)
            index += 1

        raise self._syntax_error(self.tokens[-1], "statement body must end with ';'")

    def _attach_property_note(self, properties: list[PropertyOverlay], note_text: str) -> None:
        match = _PROPERTY_NOTE_RE.match(note_text)
        if not match:
            return
        property_name, note = match.groups()
        for property_index in range(len(properties) - 1, -1, -1):
            if properties[property_index].name != property_name:
                continue
            prop = properties[property_index]
            properties[property_index] = replace(prop, notes=prop.notes + (note.strip(),))
            return

    def _expect(self, text: str, message: str | None = None) -> TdToken:
        token = self._peek()
        if token.text != text:
            raise self._syntax_error(token, message or f"expected '{text}'")
        self.index += 1
        return token

    def _expect_atom(self, text: str, message: str | None = None) -> TdToken:
        token = self._peek()
        if token.kind != "ATOM" or token.text != text:
            raise self._syntax_error(token, message or f"expected '{text}'")
        self.index += 1
        return token

    def _expect_identifier(self, message: str) -> TdToken:
        return self._expect_atom_matching(_is_identifier, message)

    def _expect_one_of(self, choices: set[str], message: str | None = None) -> TdToken:
        token = self._peek()
        if token.kind != "ATOM" or token.text not in choices:
            raise self._syntax_error(token, message or f"expected one of {sorted(choices)!r}")
        self.index += 1
        return token

    def _expect_atom_matching(self, predicate, message: str) -> TdToken:
        token = self._peek()
        if token.kind != "ATOM" or not predicate(token.text):
            raise self._syntax_error(token, message)
        self.index += 1
        return token

    def _match(self, text: str) -> bool:
        if self._peek().text != text:
            return False
        self.index += 1
        return True

    def _match_atom(self, text: str) -> bool:
        token = self._peek()
        if token.kind != "ATOM" or token.text != text:
            return False
        self.index += 1
        return True

    def _match_kind(self, kind: str) -> TdToken | None:
        token = self._peek()
        if token.kind != kind:
            return None
        self.index += 1
        return token

    def _peek(self) -> TdToken:
        return self.tokens[self.index]

    def _advance(self) -> TdToken:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _syntax_error(self, token: TdToken, message: str) -> TimingSyntaxError:
        position = token.start if token.kind != "EOF" else len(self.source)
        return TimingSyntaxError(position, message, self.source)


def parse_diagram(source: str) -> ScenarioDocument:
    """Parse timing DSL source into the canonical scenario model."""

    return TdParser(source).parse()


def _normalize_fragment(text: str) -> str:
    parts: list[str] = []
    index = 0
    in_string: str | None = None
    escaped = False
    pending_space = False

    while index < len(text):
        char = text[index]

        if in_string is not None:
            parts.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            index += 1
            continue

        if char == '"':
            if pending_space and parts and parts[-1] != " ":
                parts.append(" ")
            pending_space = False
            in_string = char
            parts.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            index += 2
            while index < len(text) and text[index] != "\n":
                index += 1
            pending_space = True
            continue
        if _starts_hash_comment(text, index):
            index += 1
            while index < len(text) and text[index] != "\n":
                index += 1
            pending_space = True
            continue
        if char.isspace():
            pending_space = True
            index += 1
            continue

        if pending_space and parts and parts[-1] != " ":
            parts.append(" ")
        pending_space = False
        parts.append(char)
        index += 1

    return "".join(parts).strip()


def _closing_delimiter(opener: str) -> str:
    return {"(": ")", "[": "]", "{": "}"}[opener]


def _starts_hash_comment(text: str, index: int) -> bool:
    if text[index] != "#":
        return False
    if index + 1 < len(text) and text[index + 1] == "#":
        return False
    if index == 0:
        return True
    return text[index - 1].isspace()


def _is_identifier(text: str) -> bool:
    return bool(_IDENT_RE.fullmatch(text))


def _is_ident_or_int(text: str) -> bool:
    return text.isdigit() or _is_identifier(text)


def _parse_window_bound(text: str) -> TimeBound:
    if re.fullmatch(rf"{IDENT}|[0-9]+", text):
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=text, max_delay=text)
    match = re.match(rf"^\[\s*({IDENT}|[0-9]+)\s*:\s*({IDENT}|[0-9]+|\$)\s*\]$", text)
    if not match:
        raise TimingDslError(f"invalid window bound: {text}")
    min_delay, max_delay = match.groups()
    if max_delay == "$":
        return TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay=min_delay, max_delay=max_delay)
    if min_delay == max_delay:
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=min_delay, max_delay=max_delay)
    return TimeBound(kind=WindowBoundKind.RANGE, min_delay=min_delay, max_delay=max_delay)


def _parse_cut_meaning(text: str, *, placement: str) -> CutMeaning:
    if text == "omitted":
        return CutMeaning.OMITTED_HISTORY if placement == "before" else CutMeaning.OMITTED_FUTURE
    if text == "compressed":
        return CutMeaning.SYMBOLIC_GAP
    return CutMeaning.LOOKBACK


def _parse_show_clause(body: str, sequence: int) -> list[LaneConstraint]:
    region_patterns = [
        (ConstraintRegion.FROM_UNTIL, re.compile(rf"^(.+?)\s+from\s+({IDENT})\s+until\s+({IDENT})$")),
        (ConstraintRegion.BEFORE, re.compile(rf"^(.+?)\s+before\s+({IDENT})$")),
        (ConstraintRegion.AFTER, re.compile(rf"^(.+?)\s+after\s+({IDENT})$")),
        (ConstraintRegion.IN, re.compile(rf"^(.+?)\s+in\s+({IDENT})$")),
        (ConstraintRegion.AT, re.compile(rf"^(.+?)\s+at\s+({IDENT})$")),
    ]
    for region, pattern in region_patterns:
        match = pattern.match(body)
        if not match:
            continue
        expr_text = match.group(1).strip()
        if region == ConstraintRegion.FROM_UNTIL:
            target = (match.group(2), match.group(3))
        else:
            target = match.group(2)
        return _build_constraints_from_show_expr(expr_text, region, target, sequence)
    raise TimingDslError(f"invalid show clause: {body}")


def _build_constraints_from_show_expr(
    expr_text: str, region: ConstraintRegion, target, sequence: int
) -> list[LaneConstraint]:
    constraints: list[LaneConstraint] = []

    assign_match = _SIGNAL_ASSIGN_RE.match(expr_text)
    if assign_match:
        signal_name, value = assign_match.groups()
        signals = (signal_name,)
        relation = "eq"
        value = value.strip()
    else:
        condition = parse_dsl_condition(expr_text)
        predicates = _flatten_condition_predicates(condition)
        if not predicates:
            raise TimingDslError(f"show clause must use typed predicates: {expr_text}")
        for offset, predicate in enumerate(predicates):
            constraints.append(_constraint_from_predicate(predicate, region, target, f"show_{sequence}_{offset}"))
        return constraints

    kwargs = _region_kwargs(region, target)
    return [
        LaneConstraint(
            name=f"show_{sequence}_0",
            signals=signals,
            relation=relation,
            value=value,
            region=region,
            **kwargs,
        )
    ]


def _constraint_from_predicate(
    predicate: Predicate, region: ConstraintRegion, target, name: str
) -> LaneConstraint:
    if predicate.signal is None:
        raise TimingDslError("predicate-backed constraints require a concrete signal")
    return LaneConstraint(
        name=name,
        signals=(predicate.signal,),
        relation=predicate.op,
        value=predicate.value,
        region=region,
        **_region_kwargs(region, target),
    )


def _region_kwargs(region: ConstraintRegion, target):
    if region == ConstraintRegion.FROM_UNTIL:
        return {"start_anchor": target[0], "end_anchor": target[1]}
    if region in {ConstraintRegion.BEFORE, ConstraintRegion.AFTER, ConstraintRegion.AT}:
        return {"anchor": target}
    if region == ConstraintRegion.IN:
        return {"window": target}
    return {}


def _parse_legacy_rule(
    rule_name: str,
    body_text: str,
) -> tuple[list[TimeWindow], list[LaneConstraint], PropertyOverlay]:
    if match := _NOT_BEFORE_RE.match(body_text):
        property_overlay = PropertyOverlay(
            name=rule_name,
            body=f"!{match.group(1)} until {match.group(2)}",
            related_anchors=(match.group(1), match.group(2)),
        )
        return ([], [], property_overlay)

    if match := _RESPONSE_RE.match(body_text):
        trigger_event, min_delay, max_delay, response_event = match.groups()
        window_name = f"{rule_name}__window"
        property_overlay = PropertyOverlay(
            name=rule_name,
            body=f"{trigger_event} |-> ##[{min_delay}:{max_delay}] {response_event}",
            related_anchors=(trigger_event, response_event),
            related_windows=(window_name,),
        )
        return (
            [
                TimeWindow(
                    name=window_name,
                    start_anchor=trigger_event,
                    end_anchor=response_event,
                    bound=TimeBound(
                        kind=WindowBoundKind.EXACT if min_delay == max_delay else WindowBoundKind.RANGE,
                        min_delay=min_delay,
                        max_delay=max_delay,
                    ),
                )
            ],
            [],
            property_overlay,
        )

    if match := _HOLD_UNTIL_RE.match(body_text):
        predicate_expr = _parse_event_expr(match.group(1))
        start_event = match.group(2)
        end_event = match.group(3)
        window_name = f"{rule_name}__window"
        constraints = [
            LaneConstraint(
                name=f"{rule_name}__constraint_{index}",
                signals=(predicate.signal,),
                relation=predicate.op,
                value=predicate.value,
                region=ConstraintRegion.FROM_UNTIL,
                start_anchor=start_event,
                end_anchor=end_event,
            )
            for index, predicate in enumerate(predicate_expr)
            if predicate.signal is not None
        ]
        property_overlay = PropertyOverlay(
            name=rule_name,
            body=f"{start_event} |-> {_event_expr_to_sva(predicate_expr)} until_with {end_event}",
            related_anchors=(start_event, end_event),
            related_windows=(window_name,),
            related_constraints=tuple(constraint.name for constraint in constraints),
        )
        return (
            [
                TimeWindow(
                    name=window_name,
                    start_anchor=start_event,
                    end_anchor=end_event,
                    bound=TimeBound(kind=WindowBoundKind.OMITTED),
                )
            ],
            constraints,
            property_overlay,
        )

    raise TimingDslError(f"unsupported rule body: {body_text}")


def _parse_event_expr(expr_text: str) -> tuple[Predicate, ...]:
    expr_text = expr_text.removesuffix(" same_cycle").strip()
    condition = parse_dsl_condition(expr_text)
    return _condition_to_event_expr(condition)


def _condition_to_event_expr(condition: Condition) -> tuple[Predicate, ...]:
    predicates = tuple(_flatten_condition_predicates(condition))
    if not predicates:
        raise TimingDslError(f"event/anchor expression is not diagram-compatible: {condition}")
    return predicates


def _flatten_condition_predicates(condition: Condition) -> list[Predicate]:
    if condition.kind == "predicate" and condition.predicate is not None:
        predicate = condition.predicate
        if predicate.signal is None:
            return []
        if predicate.op == "past":
            return []
        return [predicate]
    if condition.kind == "all":
        predicates: list[Predicate] = []
        for item in condition.items:
            predicates.extend(_flatten_condition_predicates(item))
        return predicates
    return []


def _event_expr_to_sva(expr: Sequence[Predicate]) -> str:
    parts = []
    for predicate in expr:
        if predicate.op == "rise":
            parts.append(f"$rose({predicate.signal})")
        elif predicate.op == "fall":
            parts.append(f"$fell({predicate.signal})")
        elif predicate.op == "high":
            parts.append(predicate.signal or "")
        elif predicate.op == "low":
            parts.append(f"!{predicate.signal}")
        elif predicate.op == "stable":
            parts.append(f"$stable({predicate.signal})")
        elif predicate.op == "change":
            parts.append(f"$changed({predicate.signal})")
        elif predicate.op == "eq":
            parts.append(f"({predicate.signal} == {predicate.value})")
        elif predicate.op == "neq":
            parts.append(f"({predicate.signal} != {predicate.value})")
        else:
            raise TimingDslError(f"unsupported event predicate: {predicate.op}")
    return " && ".join(parts)


def _link_property_references(
    properties: Sequence[PropertyOverlay],
    anchors: Sequence[Anchor],
    windows: Sequence[TimeWindow],
    lane_constraints: Sequence[LaneConstraint],
) -> tuple[PropertyOverlay, ...]:
    anchor_names = {anchor.name for anchor in anchors}
    window_names = {window.name for window in windows}
    constraint_names = {constraint.name for constraint in lane_constraints}
    linked = []
    for property_overlay in properties:
        referenced_anchors = list(property_overlay.related_anchors)
        referenced_windows = list(property_overlay.related_windows)
        referenced_constraints = list(property_overlay.related_constraints)
        for anchor_name in sorted(anchor_names):
            if re.search(rf"\b{anchor_name}\b", property_overlay.body) and anchor_name not in referenced_anchors:
                referenced_anchors.append(anchor_name)
        for window_name in sorted(window_names):
            if re.search(rf"\b{window_name}\b", property_overlay.body) and window_name not in referenced_windows:
                referenced_windows.append(window_name)
        for constraint_name in sorted(constraint_names):
            if (
                re.search(rf"\b{constraint_name}\b", property_overlay.body)
                and constraint_name not in referenced_constraints
            ):
                referenced_constraints.append(constraint_name)
        linked.append(
            replace(
                property_overlay,
                related_anchors=tuple(referenced_anchors),
                related_windows=tuple(referenced_windows),
                related_constraints=tuple(referenced_constraints),
            )
        )
    return tuple(linked)


def _parse_status(text: str | None) -> ExtractionStatus:
    if text is None:
        return ExtractionStatus.EXACT
    return ExtractionStatus(text)


def _strip_optional_quotes(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped[1:-1]
    return stripped
