from __future__ import annotations

from dataclasses import replace

from sva_toolkit.sva.ast import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    ClockEdge,
    ClockingEvent,
    ClockingSequence,
    ControlOperator,
    ControlProperty,
    CycleRange,
    DelaySequence,
    ExprNode,
    ExprSequence,
    FirstMatchSequence,
    Identifier,
    IfElseProperty,
    ImplicationOperator,
    ImplicationProperty,
    Literal,
    LocalVarDecl,
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
    SequenceBinary,
    SequenceBinaryOperator,
    SequenceEndedExpr,
    SequenceMatch,
    SequenceMatchItem,
    SequenceNode,
    SourceSpan,
    StatementKind,
    TernaryExpr,
    UnaryExpr,
    UnaryOperator,
    UnaryProperty,
)
from sva_toolkit.sva.errors import SvaSyntaxError
from sva_toolkit.sva.lexer import Token, TokenKind, tokenize


_UNARY_OPERATORS = {
    TokenKind.BANG: UnaryOperator.LOGICAL_NOT,
    TokenKind.TILDE: UnaryOperator.BITWISE_NOT,
    TokenKind.PLUS: UnaryOperator.UNARY_PLUS,
    TokenKind.MINUS: UnaryOperator.UNARY_MINUS,
}

_BINARY_OPERATORS = {
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


def parse_property_text(text: str, *, recover: bool = False) -> PropertySpec:
    try:
        parser = _Parser(text)
        spec = parser.parse_property_text()
        parser.expect(TokenKind.EOF)
        return spec
    except SvaSyntaxError:
        if not recover:
            raise
        return PropertySpec(body=OpaqueProperty(text=text.strip()), span=SourceSpan(0, len(text)))


def parse_property_body(text: str, *, recover: bool = False) -> PropertyNode:
    try:
        parser = _Parser(text)
        node = parser.parse_property_expr()
        parser.expect(TokenKind.EOF)
        return node
    except SvaSyntaxError:
        if not recover:
            raise
        return OpaqueProperty(text=text.strip(), span=SourceSpan(0, len(text)))


def parse_sequence(text: str, *, recover: bool = False) -> SequenceNode:
    try:
        parser = _Parser(text)
        node = parser.parse_sequence_expr()
        parser.expect(TokenKind.EOF)
        return node
    except SvaSyntaxError:
        if not recover:
            raise
        return OpaqueSequence(text=text.strip(), span=SourceSpan(0, len(text)))


def parse_expr(text: str, *, recover: bool = False) -> ExprNode:
    try:
        parser = _Parser(text)
        node = parser.parse_expr()
        parser.expect(TokenKind.EOF)
        return node
    except SvaSyntaxError:
        if not recover:
            raise
        return OpaqueExpr(text=text.strip(), span=SourceSpan(0, len(text)))


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.index = 0

    def parse_property_text(self) -> PropertySpec:
        if self.at(TokenKind.PROPERTY):
            return self.parse_named_property()
        if self.at(TokenKind.ASSERT, TokenKind.ASSUME, TokenKind.COVER) and self.peek(1).kind is TokenKind.PROPERTY:
            return self.parse_assertion_statement()
        return self.parse_property_surface()

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

    def parse_assertion_statement(self) -> PropertySpec:
        start = self.advance()
        statement_kind = StatementKind(start.text.lower())
        self.expect(TokenKind.PROPERTY)
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

    def parse_property_surface_body(self) -> tuple[ClockingEvent | None, ExprNode | None, PropertyNode]:
        clocking = self.parse_clocking_event() if self.at(TokenKind.AT) else None
        disable_iff = self.parse_disable_iff_clause() if self.at(TokenKind.DISABLE) else None
        body = self.parse_property_expr()
        return clocking, disable_iff, body

    def parse_clocking_event(self) -> ClockingEvent:
        start = self.expect(TokenKind.AT)
        self.expect(TokenKind.LPAREN)
        edge_token = self.expect(TokenKind.POSEDGE, TokenKind.NEGEDGE)
        signal_token = self.expect(TokenKind.IDENT)
        self.expect(TokenKind.RPAREN)
        return ClockingEvent(
            edge=ClockEdge(edge_token.text.lower()),
            signal=Identifier(name=signal_token.text, span=signal_token.span),
            span=self.span(start, self.previous()),
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
        if self.at(*_CONTROL_OPERATORS):
            return self.parse_control_property()
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
        operand = self.parse_property_expr()
        return ControlProperty(op=op, condition=condition, operand=operand, span=self.span(start, operand))

    def parse_implication_property(self) -> PropertyNode:
        mark = self.index
        try:
            antecedent = self.parse_sequence_expr()
            if op_token := self.match(TokenKind.BAR_ARROW, TokenKind.BAR_FAT_ARROW):
                consequent = self.parse_property_expr()
                op = ImplicationOperator(op_token.text)
                return ImplicationProperty(
                    antecedent=antecedent,
                    op=op,
                    consequent=consequent,
                    span=self.span(antecedent, consequent),
                )
        except SvaSyntaxError:
            pass
        self.index = mark
        return self.parse_property_or()

    def parse_property_or(self) -> PropertyNode:
        node = self.parse_property_until()
        while self.match(TokenKind.OR):
            right = self.parse_property_until()
            node = PropertyBinary(
                left=node,
                op=PropertyBinaryOperator.OR,
                right=right,
                span=self.span(node, right),
            )
        return node

    def parse_property_until(self) -> PropertyNode:
        node = self.parse_property_and()
        while token := self.match(TokenKind.UNTIL, TokenKind.UNTIL_WITH):
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
            node = PropertyBinary(
                left=node,
                op=PropertyBinaryOperator.AND,
                right=right,
                span=self.span(node, right),
            )
        return node

    def parse_property_unary(self) -> PropertyNode:
        if token := self.match(TokenKind.NOT):
            operand = self.parse_property_unary()
            return UnaryProperty(op=PropertyUnaryOperator.NOT, operand=operand, span=self.span(token, operand))
        if self.at(*_CONTROL_OPERATORS):
            return self.parse_control_property()
        return self.parse_sequence_term()

    def parse_sequence_expr(self) -> SequenceNode:
        return self.parse_sequence_or()

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
        return self.parse_sequence_intersect()

    def parse_sequence_intersect(self) -> SequenceNode:
        node = self.parse_sequence_throughout()
        while self.match(TokenKind.INTERSECT):
            right = self.parse_sequence_throughout()
            node = SequenceBinary(
                left=node,
                op=SequenceBinaryOperator.INTERSECT,
                right=right,
                span=self.span(node, right),
            )
        return node

    def parse_sequence_throughout(self) -> SequenceNode:
        node = self.parse_sequence_delay()
        while self.match(TokenKind.THROUGHOUT):
            right = self.parse_sequence_delay()
            node = SequenceBinary(
                left=node,
                op=SequenceBinaryOperator.THROUGHOUT,
                right=right,
                span=self.span(node, right),
            )
        return node

    def parse_sequence_delay(self) -> SequenceNode:
        if start := self.match(TokenKind.HASH_HASH):
            delay = self.parse_delay_cycle_range()
            if self.at(TokenKind.AT):
                clocking = self.parse_clocking_event()
                inner = self.parse_sequence_postfix()
                right = ClockingSequence(clocking=clocking, body=inner, span=self.span(clocking, inner))
            else:
                right = self.parse_sequence_postfix()
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
            if self.at(TokenKind.AT):
                clocking = self.parse_clocking_event()
                inner = self.parse_sequence_postfix()
                right = ClockingSequence(clocking=clocking, body=inner, span=self.span(clocking, inner))
            else:
                right = self.parse_sequence_postfix()
            node = DelaySequence(left=node, delay=delay, right=right, span=self.span(node, right))
        return node

    def parse_sequence_postfix(self) -> SequenceNode:
        node = self.parse_sequence_primary()
        while self.at(TokenKind.LBRACKET):
            mark = self.index
            self.advance()
            if self.match(TokenKind.STAR):
                op = RepeatOperator.CONSECUTIVE
            elif self.match(TokenKind.ASSIGN):
                op = RepeatOperator.NON_CONSECUTIVE
            elif self.match(TokenKind.MINUS):
                self.expect(TokenKind.GT)
                op = RepeatOperator.GOTO
            else:
                self.index = mark
                break
            count = self.parse_inner_cycle_range()
            self.expect(TokenKind.RBRACKET)
            node = RepeatSequence(body=node, op=op, count=count, span=self.span(node, self.previous()))
        return node

    def parse_sequence_primary(self) -> SequenceNode:
        if start := self.match(TokenKind.FIRST_MATCH):
            self.expect(TokenKind.LPAREN)
            body = self.parse_sequence_expr()
            self.expect(TokenKind.RPAREN)
            return FirstMatchSequence(body=body, span=self.span(start, self.previous()))
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
            ended_token = self.expect(TokenKind.IDENT)
            if ended_token.text != "ended":
                self.error(ended_token, "expected 'ended'")
            ended = SequenceEndedExpr(sequence=ExprSequence(expr=expr, span=expr.span), span=self.span(expr, ended_token))
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
            minimum = self.parse_expr()
            self.expect(TokenKind.COLON)
            if self.at(TokenKind.LITERAL) and self.current().text == "$":
                end_token = self.advance()
                self.expect(TokenKind.RBRACKET)
                return CycleRange(minimum=minimum, unbounded=True, span=self.span(minimum, end_token))
            maximum = self.parse_expr()
            self.expect(TokenKind.RBRACKET)
            return CycleRange(minimum=minimum, maximum=maximum, span=self.span(minimum, maximum))
        minimum = self.parse_expr()
        return CycleRange(minimum=minimum, span=minimum.span)

    def parse_inner_cycle_range(self) -> CycleRange:
        minimum = self.parse_expr()
        if not self.match(TokenKind.COLON):
            return CycleRange(minimum=minimum, span=minimum.span)
        if self.at(TokenKind.LITERAL) and self.current().text == "$":
            end_token = self.advance()
            return CycleRange(minimum=minimum, unbounded=True, span=self.span(minimum, end_token))
        maximum = self.parse_expr()
        return CycleRange(minimum=minimum, maximum=maximum, span=self.span(minimum, maximum))

    def parse_expr(self) -> ExprNode:
        return self.parse_ternary_expr()

    def parse_ternary_expr(self) -> ExprNode:
        condition = self.parse_logical_or_expr()
        if not self.match(TokenKind.QUESTION):
            return condition
        when_true = self.parse_expr()
        self.expect(TokenKind.COLON)
        when_false = self.parse_expr()
        return TernaryExpr(condition=condition, when_true=when_true, when_false=when_false, span=self.span(condition, when_false))

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
        while token := self.match(TokenKind.LT, TokenKind.LT_EQ, TokenKind.GT, TokenKind.GT_EQ):
            right = self.parse_additive_expr()
            node = BinaryExpr(left=node, op=_BINARY_OPERATORS[token.kind], right=right, span=self.span(node, right))
        return node

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
        if self.at(TokenKind.LITERAL):
            token = self.advance()
            return Literal(text=token.text, span=token.span)
        self.error(self.current(), "expected expression")

    def finish_call(self, name_token: Token) -> CallExpr:
        self.expect(TokenKind.LPAREN)
        args: list[ExprNode] = []
        if not self.at(TokenKind.RPAREN):
            while True:
                args.append(self.parse_expr())
                if not self.match(TokenKind.COMMA):
                    break
        self.expect(TokenKind.RPAREN)
        return CallExpr(name=name_token.text, args=tuple(args), span=self.span(name_token, self.previous()))

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
        type_text = " ".join(token.text for token in left_tokens[:-1]) or "int"
        return PropertyFormal(name=name_token.text, type_text=type_text, direction=direction, default=default, span=self.span(tokens[0], tokens[-1]))

    def parse_local_var_decl(self) -> LocalVarDecl:
        start = self.expect(TokenKind.LOCAL)
        clause_tokens = [start]
        while not self.at(TokenKind.SEMI, TokenKind.EOF):
            clause_tokens.append(self.advance())
        self.expect(TokenKind.SEMI)
        body_tokens = clause_tokens[1:]
        qualifiers: list[str] = ["local"]
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
        name_token = left_tokens[-1]
        type_text = " ".join(token.text for token in left_tokens[:-1])
        return LocalVarDecl(
            name=name_token.text,
            type_text=type_text,
            qualifiers=tuple(qualifiers),
            initializer=initializer,
            span=self.span(start, self.previous()),
        )

    def collect_clause_tokens(self, stop_kinds: set[TokenKind]) -> list[Token]:
        depth_paren = 0
        depth_bracket = 0
        tokens: list[Token] = []
        while True:
            token = self.current()
            if token.kind is TokenKind.EOF:
                break
            if depth_paren == 0 and depth_bracket == 0 and token.kind in stop_kinds:
                break
            if token.kind is TokenKind.LPAREN:
                depth_paren += 1
            elif token.kind is TokenKind.RPAREN:
                depth_paren -= 1
            elif token.kind is TokenKind.LBRACKET:
                depth_bracket += 1
            elif token.kind is TokenKind.RBRACKET:
                depth_bracket -= 1
            tokens.append(self.advance())
        return tokens

    def current(self) -> Token:
        return self.tokens[self.index]

    def previous(self) -> Token:
        return self.tokens[max(0, self.index - 1)]

    def peek(self, offset: int) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def at(self, *kinds: TokenKind) -> bool:
        return self.current().kind in kinds

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

    def span(self, start: Token | ExprNode | SequenceNode | PropertyNode, end: Token | ExprNode | SequenceNode | PropertyNode) -> SourceSpan:
        return SourceSpan(self.start_of(start), self.end_of(end))

    @staticmethod
    def start_of(item: Token | ExprNode | SequenceNode | PropertyNode) -> int:
        if isinstance(item, Token):
            return item.span.start
        if item.span is not None:
            return item.span.start
        return 0

    @staticmethod
    def end_of(item: Token | ExprNode | SequenceNode | PropertyNode) -> int:
        if isinstance(item, Token):
            return item.span.end
        if item.span is not None:
            return item.span.end
        return 0

    def slice_text(self, start: Token, end: Token) -> str:
        return self.text[start.span.start:end.span.end]


__all__ = ["parse_expr", "parse_property_body", "parse_property_text", "parse_sequence"]
