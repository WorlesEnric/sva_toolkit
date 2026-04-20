from __future__ import annotations

from dataclasses import replace

from sva_toolkit.sva.ast import (
    Always,
    BinaryExpr,
    BinaryOperator,
    Bind,
    CallExpr,
    CheckerDecl,
    ClockEdge,
    ClockingDecl,
    ClockingEvent,
    ClockingSequence,
    ControlOperator,
    ControlProperty,
    CycleRange,
    DelaySequence,
    Dist,
    DistItem,
    Ended,
    Eventually,
    Expect,
    ExprNode,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationOperator,
    ImplicationProperty,
    Inside,
    LetDecl,
    Literal,
    LocalVarDecl,
    Matched,
    MultiEventClocking,
    Nexttime,
    Node,
    OpaqueExpr,
    OpaqueProperty,
    OpaqueSequence,
    PropertyBinary,
    PropertyBinaryOperator,
    PropertyFormal,
    PropertyNode,
    PropertySpec,
    PropertyUnaryOperator,
    RepeatOperator,
    RepeatSequence,
    Restrict,
    SequenceBinary,
    SequenceBinaryOperator,
    SequenceDecl,
    SequenceEndedExpr,
    SequenceMatch,
    SequenceMatchItem,
    SequenceNode,
    SourceSpan,
    StatementKind,
    Strong,
    TernaryExpr,
    UnaryExpr,
    UnaryOperator,
    UnaryProperty,
    Weak,
    Within,
)
from sva_toolkit.sva.diagnostics import ParserDiagnostics
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import Token, TokenKind, tokenize


_UNARY_OPERATORS = {
    TokenKind.BANG: UnaryOperator.LOGICAL_NOT,
    TokenKind.TILDE: UnaryOperator.BITWISE_NOT,
    TokenKind.PLUS: UnaryOperator.UNARY_PLUS,
    TokenKind.MINUS: UnaryOperator.UNARY_MINUS,
}

_BINARY_OPERATORS = {
    TokenKind.MINUS_GT: BinaryOperator.IMPLIES,
    TokenKind.LT_MINUS_GT: BinaryOperator.IFF,
    TokenKind.PIPE_PIPE: BinaryOperator.LOGICAL_OR,
    TokenKind.AMP_AMP: BinaryOperator.LOGICAL_AND,
    TokenKind.EQ_EQ: BinaryOperator.EQ,
    TokenKind.BANG_EQ: BinaryOperator.NE,
    TokenKind.EQ_EQ_EQ: BinaryOperator.CASE_EQ,
    TokenKind.BANG_EQ_EQ: BinaryOperator.CASE_NE,
    TokenKind.LT: BinaryOperator.LT,
    TokenKind.LT_EQ: BinaryOperator.LE,
    TokenKind.GT: BinaryOperator.GT,
    TokenKind.GT_EQ: BinaryOperator.GE,
    TokenKind.PLUS: BinaryOperator.ADD,
    TokenKind.MINUS: BinaryOperator.SUB,
    TokenKind.STAR: BinaryOperator.MUL,
    TokenKind.SLASH: BinaryOperator.DIV,
    TokenKind.PERCENT: BinaryOperator.MOD,
    TokenKind.AMP: BinaryOperator.BITWISE_AND,
    TokenKind.PIPE: BinaryOperator.BITWISE_OR,
    TokenKind.CARET: BinaryOperator.BITWISE_XOR,
    TokenKind.CARET_TILDE: BinaryOperator.BITWISE_XNOR,
    TokenKind.TILDE_CARET: BinaryOperator.BITWISE_XNOR_ALT,
}

_CONTROL_OPERATORS = {
    TokenKind.ACCEPT_ON: ControlOperator.ACCEPT_ON,
    TokenKind.REJECT_ON: ControlOperator.REJECT_ON,
    TokenKind.SYNC_ACCEPT_ON: ControlOperator.SYNC_ACCEPT_ON,
    TokenKind.SYNC_REJECT_ON: ControlOperator.SYNC_REJECT_ON,
}

_TYPE_TOKENS = {
    TokenKind.BIT,
    TokenKind.LOGIC,
    TokenKind.REG,
    TokenKind.WIRE,
}


def parse_property_text(text: str, *, recover: bool = False) -> PropertySpec:
    try:
        parser = _Parser(text)
        spec = parser.parse_property_text()
        parser.expect(TokenKind.EOF)
        return spec
    except SvaSyntaxError:
        if not recover:
            raise
        span = SourceSpan(0, len(text))
        ParserDiagnostics.emit_warning("opaque_property", text, span)
        return PropertySpec(body=OpaqueProperty(text=text.strip(), span=span), span=span)


def parse_property_body(text: str, *, recover: bool = False) -> PropertyNode:
    try:
        parser = _Parser(text)
        node = parser.parse_property_expr()
        parser.expect(TokenKind.EOF)
        return node
    except SvaSyntaxError:
        if not recover:
            raise
        span = SourceSpan(0, len(text))
        ParserDiagnostics.emit_warning("opaque_property", text, span)
        return OpaqueProperty(text=text.strip(), span=span)


def parse_sequence(text: str, *, recover: bool = False) -> SequenceNode:
    try:
        parser = _Parser(text)
        node = parser.parse_sequence_expr()
        parser.expect(TokenKind.EOF)
        return node
    except SvaSyntaxError:
        if not recover:
            raise
        span = SourceSpan(0, len(text))
        ParserDiagnostics.emit_warning("opaque_sequence", text, span)
        return OpaqueSequence(text=text.strip(), span=span)


def parse_expr(text: str, *, recover: bool = False) -> ExprNode:
    try:
        parser = _Parser(text)
        node = parser.parse_expr()
        parser.expect(TokenKind.EOF)
        return node
    except SvaSyntaxError:
        if not recover:
            raise
        span = SourceSpan(0, len(text))
        ParserDiagnostics.emit_warning("opaque_expr", text, span)
        return OpaqueExpr(text=text.strip(), span=span)


def parse_declaration_text(text: str) -> Node:
    parser = _Parser(text)
    node = parser.parse_declaration()
    parser.expect(TokenKind.EOF)
    return node


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.index = 0
        self._clocking_context = False
        self._known_type_names: set[str] = set()

    def parse_property_text(self) -> PropertySpec:
        if self.at(TokenKind.PROPERTY):
            return self.parse_named_property()
        if self.at(TokenKind.ASSERT, TokenKind.ASSUME, TokenKind.COVER, TokenKind.RESTRICT) and self.peek(1).kind is TokenKind.PROPERTY:
            return self.parse_property_statement(require_property_keyword=True)
        if self.at(TokenKind.EXPECT):
            return self.parse_property_statement(require_property_keyword=False)
        return self.parse_property_surface()

    def parse_declaration(self) -> Node:
        if self.at(TokenKind.PROPERTY):
            return self.parse_named_property()
        if self.at(TokenKind.SEQUENCE):
            return self.parse_sequence_decl()
        if self.at(TokenKind.CHECKER):
            return self.parse_checker_decl()
        if self.at(TokenKind.LET):
            return self.parse_let_decl()
        if self.at(TokenKind.BIND):
            return self.parse_bind_decl()
        if self.at(TokenKind.CLOCKING) or self.at_ident_text("default") and self.peek(1).kind is TokenKind.CLOCKING:
            return self.parse_clocking_decl()
        if self.at(TokenKind.ASSERT, TokenKind.ASSUME, TokenKind.COVER, TokenKind.RESTRICT) and self.peek(1).kind is TokenKind.PROPERTY:
            return self.parse_property_statement(require_property_keyword=True)
        if self.at(TokenKind.EXPECT):
            return self.parse_property_statement(require_property_keyword=False)
        self.error(self.current(), "expected SVA declaration")

    def parse_named_property(self) -> PropertySpec:
        start = self.expect(TokenKind.PROPERTY)
        name = self.expect(TokenKind.IDENT)
        formals = self.parse_formals() if self.at(TokenKind.LPAREN) else ()
        self.expect(TokenKind.SEMI)
        local_vars: list[LocalVarDecl] = []
        while self.at(TokenKind.LOCAL):
            local_vars.append(self.parse_local_var_decl())
        surface = self.parse_property_surface()
        self.expect(TokenKind.ENDPROPERTY)
        return replace(
            surface,
            name=name.text,
            formals=tuple(formals),
            local_vars=tuple(local_vars),
            span=self.span(start, self.previous()),
        )

    def parse_sequence_decl(self) -> SequenceDecl:
        start = self.expect(TokenKind.SEQUENCE)
        name = self.expect(TokenKind.IDENT)
        formals = self.parse_formals() if self.at(TokenKind.LPAREN) else ()
        self.expect(TokenKind.SEMI)
        local_vars: list[LocalVarDecl] = []
        while self.at(TokenKind.LOCAL):
            local_vars.append(self.parse_local_var_decl())
        body = self.parse_sequence_expr()
        self.match(TokenKind.SEMI)
        self.expect(TokenKind.ENDSEQUENCE)
        return SequenceDecl(
            name=name.text,
            formals=tuple(formals),
            local_vars=tuple(local_vars),
            body=body,
            span=self.span(start, self.previous()),
        )

    def parse_checker_decl(self) -> CheckerDecl:
        start = self.expect(TokenKind.CHECKER)
        name = self.expect(TokenKind.IDENT)
        formals = self.parse_formals() if self.at(TokenKind.LPAREN) else ()
        self.expect(TokenKind.SEMI)
        items: list[Node] = []
        while not self.at(TokenKind.ENDCHECKER):
            items.append(self.parse_declaration())
        self.expect(TokenKind.ENDCHECKER)
        return CheckerDecl(name=name.text, formals=tuple(formals), items=tuple(items), span=self.span(start, self.previous()))

    def parse_clocking_decl(self) -> ClockingDecl:
        start = self.current()
        default = False
        if self.at_ident_text("default"):
            self.advance()
            default = True
        self.expect(TokenKind.CLOCKING)
        name = self.expect(TokenKind.IDENT).text if self.at(TokenKind.IDENT) else None
        event = self.parse_clocking_event() if self.at(TokenKind.AT) else None
        self.expect(TokenKind.SEMI)
        self.expect(TokenKind.ENDCLOCKING)
        return ClockingDecl(name=name, default=default, event=event, span=self.span(start, self.previous()))

    def parse_let_decl(self) -> LetDecl:
        start = self.expect(TokenKind.LET)
        name = self.expect(TokenKind.IDENT)
        formals = self.parse_formals() if self.at(TokenKind.LPAREN) else ()
        self.expect(TokenKind.ASSIGN)
        body_tokens = self.collect_clause_tokens({TokenKind.SEMI})
        if not body_tokens:
            self.error(self.current(), "expected let body")
        self.expect(TokenKind.SEMI)
        return LetDecl(
            name=name.text,
            formals=tuple(formals),
            body=self.parse_embedded_body(body_tokens),
            span=self.span(start, self.previous()),
        )

    def parse_bind_decl(self) -> Bind:
        start = self.expect(TokenKind.BIND)
        target_token = self.expect(TokenKind.IDENT)
        target = Identifier(name=target_token.text, span=target_token.span)
        head = self.expect(TokenKind.IDENT)
        instance_name: str | None = None
        checker_name = head.text
        if self.at(TokenKind.IDENT) and self.peek(1).kind is TokenKind.LPAREN:
            instance_name = head.text
            checker_name = self.expect(TokenKind.IDENT).text
        args = self.parse_call_args()
        self.expect(TokenKind.SEMI)
        return Bind(
            target=target,
            instance_name=instance_name,
            checker_name=checker_name,
            args=args,
            span=self.span(start, self.previous()),
        )

    def parse_property_statement(self, *, require_property_keyword: bool) -> PropertySpec:
        start = self.advance()
        statement_kind = StatementKind(start.text.lower())
        if require_property_keyword:
            self.expect(TokenKind.PROPERTY)
        else:
            self.match(TokenKind.PROPERTY)
        self.expect(TokenKind.LPAREN)
        clocking, disable_iff, body = self.parse_property_surface_body()
        self.expect(TokenKind.RPAREN)
        self.match(TokenKind.SEMI)
        return PropertySpec(
            body=body,
            statement_kind=statement_kind,
            clocking=clocking,
            disable_iff=disable_iff,
            span=self.span(start, self.previous()),
        )

    def parse_property_surface(self) -> PropertySpec:
        start = self.current()
        clocking, disable_iff, body = self.parse_property_surface_body()
        self.match(TokenKind.SEMI)
        return PropertySpec(
            body=body,
            clocking=clocking,
            disable_iff=disable_iff,
            span=self.span(start, body),
        )

    def parse_property_surface_body(
        self,
    ) -> tuple[ClockingEvent | MultiEventClocking | None, ExprNode | None, PropertyNode]:
        clocking = self.parse_clocking_event() if self.at(TokenKind.AT) else None
        disable_iff = self.parse_disable_iff_clause() if self.at(TokenKind.DISABLE) else None
        previous_clocking = self._clocking_context
        self._clocking_context = previous_clocking or clocking is not None
        try:
            body = self.parse_property_expr()
        finally:
            self._clocking_context = previous_clocking
        return clocking, disable_iff, body

    def parse_clocking_event(self) -> ClockingEvent | MultiEventClocking:
        start = self.expect(TokenKind.AT)
        self.expect(TokenKind.LPAREN)
        events = [self.parse_clock_event_item()]
        while self.match(TokenKind.OR):
            events.append(self.parse_clock_event_item())
        self.expect(TokenKind.RPAREN)
        if len(events) == 1:
            return replace(events[0], span=self.span(start, self.previous()))
        return MultiEventClocking(events=tuple(events), span=self.span(start, self.previous()))

    def parse_clock_event_item(self) -> ClockingEvent:
        edge_token = self.expect(TokenKind.POSEDGE, TokenKind.NEGEDGE, TokenKind.EDGE)
        signal_token = self.expect(TokenKind.IDENT)
        return ClockingEvent(
            edge=ClockEdge(edge_token.text.lower()),
            signal=Identifier(name=signal_token.text, span=signal_token.span),
            span=self.span(edge_token, signal_token),
        )

    def parse_disable_iff_clause(self) -> ExprNode:
        self.expect(TokenKind.DISABLE)
        self.expect(TokenKind.IFF)
        self.expect(TokenKind.LPAREN)
        expr = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        return expr

    def parse_property_expr(self) -> PropertyNode:
        if self.at(TokenKind.IF):
            return self.parse_if_else_property()
        return self.parse_implication_property()

    def parse_if_else_property(self) -> PropertyNode:
        start = self.expect(TokenKind.IF)
        self.expect(TokenKind.LPAREN)
        condition = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        when_true = self.parse_property_expr()
        when_false = self.parse_property_expr() if self.match(TokenKind.ELSE) else None
        end = when_false or when_true
        return IfElseProperty(condition=condition, when_true=when_true, when_false=when_false, span=self.span(start, end))

    def parse_control_property(self) -> PropertyNode:
        start = self.advance()
        op = _CONTROL_OPERATORS[start.kind]
        self.expect(TokenKind.LPAREN)
        condition = self.parse_expr()
        self.expect(TokenKind.RPAREN)
        operand = self.parse_property_unary()
        return ControlProperty(op=op, condition=condition, operand=operand, span=self.span(start, operand))

    def parse_implication_property(self) -> PropertyNode:
        mark = self.index
        try:
            antecedent = self.parse_sequence_expr()
            if op_token := self.match(TokenKind.BAR_ARROW, TokenKind.BAR_FAT_ARROW):
                consequent = self.parse_property_expr()
                return ImplicationProperty(
                    antecedent=antecedent,
                    op=ImplicationOperator(op_token.text),
                    consequent=consequent,
                    span=self.span(antecedent, consequent),
                )
        except SvaSyntaxError:
            pass
        self.index = mark
        return self.parse_property_iff()

    def parse_property_iff(self) -> PropertyNode:
        node = self.parse_property_implies()
        while self.match(TokenKind.IFF):
            right = self.parse_property_implies()
            node = PropertyBinary(left=node, op=PropertyBinaryOperator.IFF, right=right, span=self.span(node, right))
        return node

    def parse_property_implies(self) -> PropertyNode:
        node = self.parse_property_or()
        while self.match(TokenKind.IMPLIES):
            right = self.parse_property_or()
            node = PropertyBinary(left=node, op=PropertyBinaryOperator.IMPLIES, right=right, span=self.span(node, right))
        return node

    def parse_property_or(self) -> PropertyNode:
        node = self.parse_property_until()
        while self.match(TokenKind.OR):
            right = self.parse_property_until()
            node = PropertyBinary(left=node, op=PropertyBinaryOperator.OR, right=right, span=self.span(node, right))
        return node

    def parse_property_until(self) -> PropertyNode:
        node = self.parse_property_and()
        while token := self.match(TokenKind.UNTIL, TokenKind.UNTIL_WITH, TokenKind.S_UNTIL, TokenKind.S_UNTIL_WITH):
            right = self.parse_property_and()
            node = PropertyBinary(
                left=node,
                op=PropertyBinaryOperator(token.text),
                right=right,
                span=self.span(node, right),
            )
        return node

    def parse_property_and(self) -> PropertyNode:
        node = self.parse_property_unary()
        while self.match(TokenKind.AND):
            right = self.parse_property_unary()
            node = PropertyBinary(left=node, op=PropertyBinaryOperator.AND, right=right, span=self.span(node, right))
        return node

    def parse_property_unary(self) -> PropertyNode:
        if token := self.match(TokenKind.NOT):
            operand = self.parse_property_unary()
            return UnaryProperty(op=PropertyUnaryOperator.NOT, operand=operand, span=self.span(token, operand))
        if self.at(*_CONTROL_OPERATORS):
            return self.parse_control_property()
        if self.at(TokenKind.RESTRICT):
            return self.parse_restrict_property()
        if self.at(TokenKind.EXPECT):
            return self.parse_expect_property()
        if self.at(TokenKind.STRONG, TokenKind.WEAK):
            return self.parse_strength_property()
        if self.at(
            TokenKind.NEXTTIME,
            TokenKind.S_NEXTTIME,
            TokenKind.ALWAYS,
            TokenKind.S_ALWAYS,
            TokenKind.EVENTUALLY,
            TokenKind.S_EVENTUALLY,
        ):
            return self.parse_temporal_property()
        if self.match(TokenKind.LPAREN):
            node = self.parse_property_expr()
            self.expect(TokenKind.RPAREN)
            return node
        return self.parse_sequence_term()

    def parse_restrict_property(self) -> PropertyNode:
        start = self.expect(TokenKind.RESTRICT)
        if self.match(TokenKind.PROPERTY):
            operand = self.parse_parenthesized_property_expr()
        else:
            operand = self.parse_property_unary()
        return Restrict(operand=operand, span=self.span(start, operand))

    def parse_expect_property(self) -> PropertyNode:
        start = self.expect(TokenKind.EXPECT)
        if self.match(TokenKind.PROPERTY) or self.at(TokenKind.LPAREN):
            operand = self.parse_parenthesized_property_expr()
        else:
            operand = self.parse_property_unary()
        return Expect(operand=operand, span=self.span(start, operand))

    def parse_strength_property(self) -> PropertyNode:
        start = self.advance()
        operand = self.parse_parenthesized_property_expr() if self.at(TokenKind.LPAREN) else self.parse_property_unary()
        node_cls = Strong if start.kind is TokenKind.STRONG else Weak
        return node_cls(operand=operand, span=self.span(start, operand))

    def parse_temporal_property(self) -> PropertyNode:
        start = self.advance()
        if not self._clocking_context:
            self.error(start, f"{start.text} requires a clocking event")
        cycle_range = None
        if self.match(TokenKind.LBRACKET):
            cycle_range = self.parse_inner_cycle_range()
            self.expect(TokenKind.RBRACKET)
        operand = self.parse_property_unary()
        strong = start.kind in {TokenKind.S_NEXTTIME, TokenKind.S_ALWAYS, TokenKind.S_EVENTUALLY}
        if start.kind in {TokenKind.NEXTTIME, TokenKind.S_NEXTTIME}:
            return Nexttime(operand=operand, cycle_delay=cycle_range, strong=strong, span=self.span(start, operand))
        if start.kind in {TokenKind.ALWAYS, TokenKind.S_ALWAYS}:
            return Always(operand=operand, cycle_range=cycle_range, strong=strong, span=self.span(start, operand))
        return Eventually(operand=operand, cycle_range=cycle_range, strong=strong, span=self.span(start, operand))

    def parse_parenthesized_property_expr(self) -> PropertyNode:
        self.expect(TokenKind.LPAREN)
        node = self.parse_property_expr()
        self.expect(TokenKind.RPAREN)
        return node

    def parse_sequence_expr(self) -> SequenceNode:
        return self.parse_sequence_within()

    def parse_sequence_property_term(self) -> SequenceNode:
        return self.parse_sequence_property_within()

    def parse_sequence_property_within(self) -> SequenceNode:
        node = self.parse_sequence_intersect()
        while self.match(TokenKind.WITHIN):
            right = self.parse_sequence_intersect()
            node = Within(left=node, right=right, span=self.span(node, right))
        return node

    def parse_sequence_within(self) -> SequenceNode:
        node = self.parse_sequence_or()
        while self.match(TokenKind.WITHIN):
            right = self.parse_sequence_or()
            node = Within(left=node, right=right, span=self.span(node, right))
        return node

    def parse_sequence_or(self) -> SequenceNode:
        node = self.parse_sequence_and()
        while self.match(TokenKind.OR):
            right = self.parse_sequence_and()
            node = SequenceBinary(left=node, op=SequenceBinaryOperator.OR, right=right, span=self.span(node, right))
        return node

    def parse_sequence_and(self) -> SequenceNode:
        node = self.parse_sequence_intersect()
        while self.match(TokenKind.AND):
            right = self.parse_sequence_intersect()
            node = SequenceBinary(left=node, op=SequenceBinaryOperator.AND, right=right, span=self.span(node, right))
        return node

    def parse_sequence_term(self) -> SequenceNode:
        return self.parse_sequence_property_term()

    def parse_sequence_intersect(self) -> SequenceNode:
        node = self.parse_sequence_throughout()
        while self.match(TokenKind.INTERSECT):
            right = self.parse_sequence_throughout()
            node = SequenceBinary(left=node, op=SequenceBinaryOperator.INTERSECT, right=right, span=self.span(node, right))
        return node

    def parse_sequence_throughout(self) -> SequenceNode:
        node = self.parse_sequence_delay()
        while self.match(TokenKind.THROUGHOUT):
            right = self.parse_sequence_delay()
            node = SequenceBinary(left=node, op=SequenceBinaryOperator.THROUGHOUT, right=right, span=self.span(node, right))
        return node

    def parse_sequence_delay(self) -> SequenceNode:
        if start := self.match(TokenKind.HASH_HASH):
            delay = self.parse_delay_cycle_range()
            right = self.parse_clocked_sequence_operand()
            left = ExprSequence(
                expr=Literal(text="1'b1", span=SourceSpan(start.span.start, start.span.start)),
                span=SourceSpan(start.span.start, start.span.start),
            )
            node = DelaySequence(left=left, delay=delay, right=right, span=self.span(start, right))
        elif self.at(TokenKind.AT):
            clocking = self.parse_clocking_event()
            inner = self.parse_sequence_postfix()
            node = ClockingSequence(clocking=clocking, body=inner, span=self.span(clocking, inner))
        else:
            node = self.parse_sequence_postfix()

        while self.match(TokenKind.HASH_HASH):
            delay = self.parse_delay_cycle_range()
            right = self.parse_clocked_sequence_operand()
            node = DelaySequence(left=node, delay=delay, right=right, span=self.span(node, right))
        return node

    def parse_clocked_sequence_operand(self) -> SequenceNode:
        if self.at(TokenKind.AT):
            clocking = self.parse_clocking_event()
            inner = self.parse_sequence_postfix()
            return ClockingSequence(clocking=clocking, body=inner, span=self.span(clocking, inner))
        return self.parse_sequence_postfix()

    def parse_sequence_postfix(self) -> SequenceNode:
        node = self.parse_sequence_primary()
        while True:
            if token := self.match(TokenKind.LBRACKET_PLUS_RBRACKET):
                node = RepeatSequence(body=node, op=RepeatOperator.ONE_OR_MORE, count=self.one_or_more_range(token), span=self.span(node, token))
                continue
            if token := self.match(TokenKind.LBRACKET_STAR_RBRACKET):
                node = RepeatSequence(body=node, op=RepeatOperator.CONSECUTIVE, count=self.zero_or_more_range(token), span=self.span(node, token))
                continue
            if not self.match(TokenKind.LBRACKET):
                break
            if self.match(TokenKind.PLUS):
                end = self.expect(TokenKind.RBRACKET)
                node = RepeatSequence(body=node, op=RepeatOperator.ONE_OR_MORE, count=self.one_or_more_range(end), span=self.span(node, end))
                continue
            if self.match(TokenKind.STAR):
                if token := self.match(TokenKind.RBRACKET):
                    node = RepeatSequence(body=node, op=RepeatOperator.CONSECUTIVE, count=self.zero_or_more_range(token), span=self.span(node, token))
                    continue
                op = RepeatOperator.CONSECUTIVE
            elif self.match(TokenKind.ASSIGN):
                op = RepeatOperator.NON_CONSECUTIVE
            elif self.match(TokenKind.MINUS):
                self.expect(TokenKind.GT)
                op = RepeatOperator.GOTO
            else:
                self.error(self.previous(), "expected repetition operator")
            count = self.parse_inner_cycle_range()
            end = self.expect(TokenKind.RBRACKET)
            node = RepeatSequence(body=node, op=op, count=count, span=self.span(node, end))
        return node

    def parse_sequence_primary(self) -> SequenceNode:
        if start := self.match(TokenKind.FIRST_MATCH):
            self.expect(TokenKind.LPAREN)
            body = self.parse_sequence_expr()
            self.expect(TokenKind.RPAREN)
            return FirstMatchSequence(body=body, span=self.span(start, self.previous()))
        if start := self.match(TokenKind.MATCHED):
            self.expect(TokenKind.LPAREN)
            body = self.parse_sequence_expr()
            self.expect(TokenKind.RPAREN)
            return Matched(sequence=body, span=self.span(start, self.previous()))
        if self.at_ident_text("ended"):
            start = self.advance()
            self.expect(TokenKind.LPAREN)
            body = self.parse_sequence_expr()
            self.expect(TokenKind.RPAREN)
            return Ended(sequence=body, span=self.span(start, self.previous()))
        if self.match(TokenKind.LPAREN):
            node = self.parse_sequence_expr()
            if self.match(TokenKind.COMMA):
                items = []
                while True:
                    items.append(self.parse_sequence_match_item())
                    if not self.match(TokenKind.COMMA):
                        break
                self.expect(TokenKind.RPAREN)
                return SequenceMatch(body=node, items=tuple(items), span=self.span(node, self.previous()))
            self.expect(TokenKind.RPAREN)
            return node

        expr = self.parse_expr()
        if self.match(TokenKind.DOT):
            suffix = self.expect(TokenKind.IDENT)
            if suffix.text != "ended":
                self.error(suffix, "expected 'ended'")
            ended = SequenceEndedExpr(sequence=ExprSequence(expr=expr, span=expr.span), span=self.span(expr, suffix))
            return ExprSequence(expr=ended, span=self.span(expr, ended))
        return ExprSequence(expr=expr, span=expr.span)

    def parse_sequence_match_item(self) -> SequenceMatchItem:
        start = self.expect(TokenKind.IDENT)
        lvalue = Identifier(name=start.text, span=start.span)
        self.expect(TokenKind.ASSIGN)
        rvalue = self.parse_expr()
        return SequenceMatchItem(lvalue=lvalue, rvalue=rvalue, span=self.span(start, rvalue))

    def parse_delay_cycle_range(self) -> CycleRange:
        if self.match(TokenKind.LBRACKET):
            inner = self.parse_inner_cycle_range()
            self.expect(TokenKind.RBRACKET)
            return inner
        minimum = self.parse_expr()
        return CycleRange(minimum=minimum, span=minimum.span)

    def parse_inner_cycle_range(self) -> CycleRange:
        minimum = self.parse_expr()
        if not self.match(TokenKind.COLON):
            return CycleRange(minimum=minimum, span=minimum.span)
        if end_token := self.match(TokenKind.DOLLAR):
            return CycleRange(minimum=minimum, unbounded=True, span=self.span(minimum, end_token))
        maximum = self.parse_expr()
        return CycleRange(minimum=minimum, maximum=maximum, span=self.span(minimum, maximum))

    def parse_expr(self) -> ExprNode:
        return self.parse_ternary_expr()

    def parse_ternary_expr(self) -> ExprNode:
        condition = self.parse_logical_iff_expr()
        if not self.match(TokenKind.QUESTION):
            return condition
        when_true = self.parse_expr()
        self.expect(TokenKind.COLON)
        when_false = self.parse_expr()
        return TernaryExpr(condition=condition, when_true=when_true, when_false=when_false, span=self.span(condition, when_false))

    def parse_logical_iff_expr(self) -> ExprNode:
        node = self.parse_logical_implies_expr()
        while token := self.match(TokenKind.LT_MINUS_GT):
            right = self.parse_logical_implies_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

    def parse_logical_implies_expr(self) -> ExprNode:
        node = self.parse_logical_or_expr()
        while token := self.match(TokenKind.MINUS_GT):
            right = self.parse_logical_or_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

    def parse_logical_or_expr(self) -> ExprNode:
        node = self.parse_logical_and_expr()
        while self.match(TokenKind.PIPE_PIPE):
            right = self.parse_logical_and_expr()
            node = BinaryExpr(left=node, op=BinaryOperator.LOGICAL_OR, right=right, span=self.span(node, right))
        return node

    def parse_logical_and_expr(self) -> ExprNode:
        node = self.parse_bitwise_or_expr()
        while self.match(TokenKind.AMP_AMP):
            right = self.parse_bitwise_or_expr()
            node = BinaryExpr(left=node, op=BinaryOperator.LOGICAL_AND, right=right, span=self.span(node, right))
        return node

    def parse_bitwise_or_expr(self) -> ExprNode:
        node = self.parse_bitwise_xor_expr()
        while self.match(TokenKind.PIPE):
            right = self.parse_bitwise_xor_expr()
            node = BinaryExpr(left=node, op=BinaryOperator.BITWISE_OR, right=right, span=self.span(node, right))
        return node

    def parse_bitwise_xor_expr(self) -> ExprNode:
        node = self.parse_bitwise_and_expr()
        while token := self.match(TokenKind.CARET, TokenKind.CARET_TILDE, TokenKind.TILDE_CARET):
            right = self.parse_bitwise_and_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

    def parse_bitwise_and_expr(self) -> ExprNode:
        node = self.parse_equality_expr()
        while self.match(TokenKind.AMP):
            right = self.parse_equality_expr()
            node = BinaryExpr(left=node, op=BinaryOperator.BITWISE_AND, right=right, span=self.span(node, right))
        return node

    def parse_equality_expr(self) -> ExprNode:
        node = self.parse_relational_expr()
        while token := self.match(TokenKind.EQ_EQ, TokenKind.BANG_EQ, TokenKind.EQ_EQ_EQ, TokenKind.BANG_EQ_EQ):
            right = self.parse_relational_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

    def parse_relational_expr(self) -> ExprNode:
        node = self.parse_additive_expr()
        while True:
            if token := self.match(TokenKind.LT, TokenKind.LT_EQ, TokenKind.GT, TokenKind.GT_EQ):
                right = self.parse_additive_expr()
                node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
                continue
            if self.match(TokenKind.INSIDE):
                items, end = self.parse_inside_items()
                node = Inside(expr=node, items=items, span=self.span(node, end))
                continue
            if self.match(TokenKind.DIST):
                items, end = self.parse_dist_items()
                node = Dist(expr=node, items=items, span=self.span(node, end))
                continue
            return node

    def parse_inside_items(self) -> tuple[tuple[ExprNode, ...], Token]:
        self.expect(TokenKind.LBRACE)
        items: list[ExprNode] = []
        if not self.at(TokenKind.RBRACE):
            while True:
                items.append(self.parse_expr())
                if not self.match(TokenKind.COMMA):
                    break
        end = self.expect(TokenKind.RBRACE)
        return tuple(items), end

    def parse_dist_items(self) -> tuple[tuple[DistItem, ...], Token]:
        self.expect(TokenKind.LBRACE)
        items: list[DistItem] = []
        if not self.at(TokenKind.RBRACE):
            while True:
                items.append(self.parse_dist_item())
                if not self.match(TokenKind.COMMA):
                    break
        end = self.expect(TokenKind.RBRACE)
        return tuple(items), end

    def parse_dist_item(self) -> DistItem:
        value = self.parse_expr()
        weight = None
        per_item = False
        if self.match(TokenKind.COLON):
            if self.match(TokenKind.ASSIGN):
                weight = self.parse_expr()
            elif self.match(TokenKind.SLASH):
                weight = self.parse_expr()
                per_item = True
            else:
                self.error(self.current(), "expected := or :/ in dist item")
        return DistItem(value=value, weight=weight, per_item=per_item, span=self.span(value, weight or value))

    def parse_additive_expr(self) -> ExprNode:
        node = self.parse_multiplicative_expr()
        while token := self.match(TokenKind.PLUS, TokenKind.MINUS):
            right = self.parse_multiplicative_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

    def parse_multiplicative_expr(self) -> ExprNode:
        node = self.parse_unary_expr()
        while token := self.match(TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT):
            right = self.parse_unary_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

    def parse_unary_expr(self) -> ExprNode:
        if token := self.match(TokenKind.BANG, TokenKind.TILDE, TokenKind.PLUS, TokenKind.MINUS):
            operand = self.parse_unary_expr()
            return UnaryExpr(op=_UNARY_OPERATORS[token.kind], operand=operand, span=self.span(token, operand))
        return self.parse_primary_expr()

    def parse_primary_expr(self) -> ExprNode:
        if self.match(TokenKind.LPAREN):
            expr = self.parse_expr()
            self.expect(TokenKind.RPAREN)
            return expr
        if self.at(TokenKind.IDENT):
            token = self.advance()
            if self.at(TokenKind.LPAREN):
                return self.finish_call(token)
            return Identifier(name=token.text, span=token.span)
        if self.at(TokenKind.DOLLAR_IDENT):
            return self.finish_call(self.advance())
        if self.at(TokenKind.LITERAL, TokenKind.STRING):
            token = self.advance()
            return Literal(text=token.text, span=token.span)
        self.error(self.current(), "expected expression")

    def finish_call(self, name_token: Token) -> CallExpr:
        return CallExpr(name=name_token.text, args=self.parse_call_args(), span=self.span(name_token, self.previous()))

    def parse_call_args(self) -> tuple[ExprNode, ...]:
        self.expect(TokenKind.LPAREN)
        args: list[ExprNode] = []
        if not self.at(TokenKind.RPAREN):
            while True:
                args.append(self.parse_expr())
                if not self.match(TokenKind.COMMA):
                    break
        self.expect(TokenKind.RPAREN)
        return tuple(args)

    def parse_formals(self) -> tuple[PropertyFormal, ...]:
        self.expect(TokenKind.LPAREN)
        formals: list[PropertyFormal] = []
        if not self.at(TokenKind.RPAREN):
            while True:
                formals.append(self.parse_formal())
                if not self.match(TokenKind.COMMA):
                    break
        self.expect(TokenKind.RPAREN)
        return tuple(formals)

    def parse_formal(self) -> PropertyFormal:
        tokens = self.collect_clause_tokens({TokenKind.COMMA, TokenKind.RPAREN})
        if not tokens:
            self.error(self.current(), "expected property formal")
        default_index = next((index for index, token in enumerate(tokens) if token.kind is TokenKind.ASSIGN), None)
        left_tokens = tokens if default_index is None else tokens[:default_index]
        default = None
        if default_index is not None:
            default_text = self.slice_text(tokens[default_index + 1], tokens[-1])
            default = parse_expr(default_text, recover=True)
        direction = None
        if left_tokens and left_tokens[0].text in {"input", "output", "inout", "ref"}:
            direction = left_tokens[0].text
            left_tokens = left_tokens[1:]
        if not left_tokens:
            self.error(tokens[0], "malformed property formal")
        name_token = left_tokens[-1]
        if name_token.kind is not TokenKind.IDENT:
            self.error(name_token, "expected formal name")
        type_text = self.validate_type_tokens(left_tokens[:-1], allow_default=True)
        return PropertyFormal(name=name_token.text, type_text=type_text, direction=direction, default=default, span=self.span(tokens[0], tokens[-1]))

    def parse_local_var_decl(self) -> LocalVarDecl:
        start = self.expect(TokenKind.LOCAL)
        clause_tokens = [start]
        while not self.at(TokenKind.SEMI, TokenKind.EOF):
            clause_tokens.append(self.advance())
        self.expect(TokenKind.SEMI)
        body_tokens = clause_tokens[1:]
        qualifiers = ["local"]
        while body_tokens and body_tokens[0].text in {"var", "const", "static"}:
            qualifiers.append(body_tokens[0].text)
            body_tokens = body_tokens[1:]
        if not body_tokens:
            self.error(start, "malformed local variable declaration")
        assign_index = next((index for index, token in enumerate(body_tokens) if token.kind is TokenKind.ASSIGN), None)
        left_tokens = body_tokens if assign_index is None else body_tokens[:assign_index]
        initializer = None
        if assign_index is not None:
            init_text = self.slice_text(body_tokens[assign_index + 1], body_tokens[-1])
            initializer = parse_expr(init_text, recover=True)
        if len(left_tokens) < 2:
            self.error(start, "local variable declaration requires an explicit type")
        name_token = left_tokens[-1]
        if name_token.kind is not TokenKind.IDENT:
            self.error(name_token, "expected local variable name")
        type_text = self.validate_type_tokens(left_tokens[:-1], allow_default=False)
        return LocalVarDecl(
            name=name_token.text,
            type_text=type_text,
            qualifiers=tuple(qualifiers),
            initializer=initializer,
            span=self.span(start, self.previous()),
        )

    def validate_type_tokens(self, tokens: list[Token], *, allow_default: bool) -> str:
        if not tokens:
            if allow_default:
                return "int"
            self.error(self.current(), "expected type name")
        first = tokens[0]
        if first.kind not in _TYPE_TOKENS and first.kind is not TokenKind.IDENT:
            self.error(first, "expected built-in or user-defined type name")
        if first.kind is TokenKind.IDENT:
            self._known_type_names.add(first.text)
        return " ".join(token.text for token in tokens)

    def parse_embedded_body(self, tokens: list[Token]) -> ExprNode | SequenceNode | PropertyNode:
        text = self.slice_text(tokens[0], tokens[-1])
        for parser in (parse_expr, parse_sequence, parse_property_body):
            try:
                return parser(text)
            except SvaSyntaxError:
                continue
        self.error(tokens[0], "unsupported declaration body")

    def collect_clause_tokens(self, stop_kinds: set[TokenKind]) -> list[Token]:
        depth_paren = 0
        depth_brace = 0
        depth_bracket = 0
        tokens: list[Token] = []
        while True:
            token = self.current()
            if token.kind is TokenKind.EOF:
                break
            if depth_paren == 0 and depth_brace == 0 and depth_bracket == 0 and token.kind in stop_kinds:
                break
            if token.kind is TokenKind.LPAREN:
                depth_paren += 1
            elif token.kind is TokenKind.RPAREN:
                depth_paren -= 1
            elif token.kind is TokenKind.LBRACE:
                depth_brace += 1
            elif token.kind is TokenKind.RBRACE:
                depth_brace -= 1
            elif token.kind is TokenKind.LBRACKET:
                depth_bracket += 1
            elif token.kind is TokenKind.RBRACKET:
                depth_bracket -= 1
            tokens.append(self.advance())
        return tokens

    def zero_or_more_range(self, token: Token) -> CycleRange:
        minimum = Literal(text="0", span=token.span)
        return CycleRange(minimum=minimum, unbounded=True, span=token.span)

    def one_or_more_range(self, token: Token) -> CycleRange:
        minimum = Literal(text="1", span=token.span)
        return CycleRange(minimum=minimum, unbounded=True, span=token.span)

    def current(self) -> Token:
        return self.tokens[self.index]

    def previous(self) -> Token:
        return self.tokens[max(0, self.index - 1)]

    def peek(self, offset: int) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def at(self, *kinds: TokenKind) -> bool:
        return self.current().kind in kinds

    def at_ident_text(self, *names: str) -> bool:
        return self.current().kind is TokenKind.IDENT and self.current().text.lower() in names

    def advance(self) -> Token:
        token = self.current()
        if self.index < len(self.tokens) - 1:
            self.index += 1
        return token

    def match(self, *kinds: TokenKind) -> Token | None:
        if self.at(*kinds):
            return self.advance()
        return None

    def expect(self, *kinds: TokenKind) -> Token:
        token = self.current()
        if token.kind not in kinds:
            expected = ", ".join(kind.value for kind in kinds)
            self.error(token, f"expected {expected}")
        return self.advance()

    def error(self, token: Token, message: str) -> None:
        raise SvaSyntaxError(token.span.start, message, self.text)

    def span(self, start: Token | Node, end: Token | Node) -> SourceSpan:
        return SourceSpan(self.start_of(start), self.end_of(end))

    @staticmethod
    def start_of(item: Token | Node) -> int:
        if isinstance(item, Token):
            return item.span.start
        if item.span is not None:
            return item.span.start
        return 0

    @staticmethod
    def end_of(item: Token | Node) -> int:
        if isinstance(item, Token):
            return item.span.end
        if item.span is not None:
            return item.span.end
        return 0

    def slice_text(self, start: Token, end: Token) -> str:
        return self.text[start.span.start:end.span.end]


__all__ = [
    "parse_declaration_text",
    "parse_expr",
    "parse_property_body",
    "parse_property_text",
    "parse_sequence",
]
