"""SVA to timing scenario extraction and bundling."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, List, Sequence

from sva_toolkit.sva.analysis import CollectBoundNamesVisitor, ContainsNodeKindVisitor
from sva_toolkit.sva.emitter import emit_expr, emit_property_body, emit_sequence
from sva_toolkit.sva.parser import parse_property_body
from sva_toolkit.sva.transforms import RenameIdentifiersTransformer
from sva_toolkit.timing.bridge.solver import CEGSolver
from sva_toolkit.timing.bridge.ebmc_witness import (
    EbmcWitnessSynthesizer,
    refine_document_from_witness,
)
from sva_toolkit.sva.visitors import NodeVisitor
from sva_toolkit.timing.core.graph import CanonicalEventGraph, EdgeKind, TimeRange, CEGNode
from sva_toolkit.sva.ast import (
    BinaryExpr,
    BinaryOperator,
    CallExpr,
    ControlProperty,
    FirstMatchSequence,
    IfElseProperty,
    Identifier,
    ImplicationProperty,
    PropertySpec,
    PropertyNode,
    SequenceNode,
    ExprSequence,
    DelaySequence,
    RepeatOperator,
    RepeatSequence,
    SequenceBinary,
    SequenceBinaryOperator,
    ImplicationOperator,
    PropertyBinary,
    PropertyBinaryOperator,
    ClockingSequence,
    SequenceMatch,
    SequenceMatchItem,
)
from sva_toolkit.timing.core.conditions import (
    Condition,
    Predicate,
    collect_signals,
    condition_to_sva,
    parse_sva_condition,
)
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
    merge_bundle_metadata,
    rebind_names,
)


class SvaCompiler(NodeVisitor[None]):
    """Compiles SVA AST into a Canonical Event Graph (CEG)."""

    def __init__(self, graph: CanonicalEventGraph) -> None:
        self.graph = graph
        self.current_nodes: List[CEGNode] = []
        self.status = ExtractionStatus.EXACT
        self.notes: List[str] = []
        # Side-channel: hold regions from until_with/until properties
        # Each entry: (start_nodes, left_ast, end_nodes, op)
        self.hold_regions: List[tuple] = []

    def compile(self, node: PropertyNode) -> List[CEGNode]:
        """Compile a property node and return its terminal nodes."""
        self.current_nodes = self.graph.get_roots()
        
        if not self.current_nodes:
            # Create a base root
            root = self.graph.create_node(condition=None)
            root.is_trigger = True
            self.current_nodes = [root]

        # Optimization: If it's a simple implication, we already have our root.
        if isinstance(node, ImplicationProperty):
            self.visit(node.antecedent)
            # Mark current nodes (after antecedent) as triggers
            for n in self.current_nodes:
                n.is_trigger = True
            
            # Implication delay
            delay_min = "0" if node.op == ImplicationOperator.OVERLAPPED else "1"
            delay_range = TimeRange(min_delay=delay_min, max_delay=delay_min)
            
            next_nodes = []
            for current in self.current_nodes:
                target = self.graph.create_node()
                self.graph.add_edge(current, target, delay=delay_range)
                next_nodes.append(target)
            self.current_nodes = next_nodes
            self.visit(node.consequent)
            return self.current_nodes
        
        self.visit(node)
        return self.current_nodes

    def visit_ExprSequence(self, node: ExprSequence) -> None:
        new_nodes = []
        for current in self.current_nodes:
            if current.condition is None and not self.graph.get_in_edges(current.id):
                # Optimization: put condition on the root if it's empty and has no inputs
                current.condition = node.expr
                new_nodes.append(current)
            else:
                target = self.graph.create_node(condition=node.expr)
                self.graph.add_edge(current, target, delay=TimeRange(min_delay="0", max_delay="0"))
                new_nodes.append(target)
        self.current_nodes = new_nodes

    def visit_DelaySequence(self, node: DelaySequence) -> None:
        # Compile left side
        self.visit(node.left)
        
        # Determine delay range
        min_val = emit_expr(node.delay.minimum)
        max_val = emit_expr(node.delay.maximum) if node.delay.maximum else (None if node.delay.unbounded else min_val)
        delay_range = TimeRange(min_delay=min_val, max_delay=max_val, unbounded=node.delay.unbounded)
        
        # Optimization: If right side is a simple expression, avoid intermediate node
        if isinstance(node.right, ExprSequence):
            next_nodes = []
            for current in self.current_nodes:
                target = self.graph.create_node(condition=node.right.expr)
                self.graph.add_edge(current, target, delay=delay_range)
                next_nodes.append(target)
            self.current_nodes = next_nodes
        else:
            mid_nodes = []
            for current in self.current_nodes:
                target = self.graph.create_node()
                self.graph.add_edge(current, target, delay=delay_range)
                mid_nodes.append(target)
            self.current_nodes = mid_nodes
            self.visit(node.right)

    def visit_ImplicationProperty(self, node: ImplicationProperty) -> None:
        # Antecedent
        self.visit(node.antecedent)
        
        # Mark current nodes as triggers
        for n in self.current_nodes:
            n.is_trigger = True
            
        # Implication delay
        delay_min = "0" if node.op == ImplicationOperator.OVERLAPPED else "1"
        delay_range = TimeRange(min_delay=delay_min, max_delay=delay_min)
        
        next_nodes = []
        for current in self.current_nodes:
            target = self.graph.create_node()
            self.graph.add_edge(current, target, delay=delay_range)
            next_nodes.append(target)
        
        self.current_nodes = next_nodes
        
        # Consequent
        self.visit(node.consequent)

    def visit_SequenceBinary(self, node: SequenceBinary) -> None:
        start_nodes = self.current_nodes
        if node.op == SequenceBinaryOperator.INTERSECT:
            self.notes.append("intersect modeled as parallel concurrent sequences in DAG")
            
            # Left branch
            self.current_nodes = start_nodes
            self.visit(node.left)
            left_terminals = self.current_nodes
            
            # Right branch
            self.current_nodes = start_nodes
            self.visit(node.right)
            right_terminals = self.current_nodes
            
            # Merge terminals (simplified for now: intersection implies cumulative path length matching)
            self.current_nodes = list(set(left_terminals) | set(right_terminals))
        elif node.op in (SequenceBinaryOperator.AND, SequenceBinaryOperator.OR):
            kind = EdgeKind.AND if node.op == SequenceBinaryOperator.AND else EdgeKind.OR
            
            self.current_nodes = start_nodes
            self.visit(node.left)
            left_terminals = self.current_nodes
            
            self.current_nodes = start_nodes
            self.visit(node.right)
            right_terminals = self.current_nodes
            
            self.current_nodes = list(set(left_terminals) | set(right_terminals))
        else:
            self.generic_visit(node)

    def visit_ClockingSequence(self, node: ClockingSequence) -> None:
        for n in self.current_nodes:
            n.clocking = node.clocking
        self.visit(node.body)

    def visit_SequenceMatch(self, node: SequenceMatch) -> None:
        self.visit(node.body)
        for n in self.current_nodes:
            for item in node.items:
                n.assignments.append((item.lvalue.name, item.rvalue))

    def visit_RepeatSequence(self, node: RepeatSequence) -> None:
        # Map repetition to a temporal duration in the CEG
        # For [*N], it's an exact delay. For [*M:N], it's a range.
        self.visit(node.body)
        
        min_count = emit_expr(node.count.minimum)
        max_count = emit_expr(node.count.maximum) if node.count.maximum else (None if node.count.unbounded else min_count)
        
        delay_range = TimeRange(
            min_delay=min_count,
            max_delay=max_count,
            unbounded=node.count.unbounded
        )
        
        # Add an edge representing the repetition duration
        next_nodes = []
        for current in self.current_nodes:
            target = self.graph.create_node()
            self.graph.add_edge(current, target, delay=delay_range)
            next_nodes.append(target)
        self.current_nodes = next_nodes

    def visit_PropertyBinary(self, node: PropertyBinary) -> None:
        if node.op in (PropertyBinaryOperator.UNTIL_WITH, PropertyBinaryOperator.UNTIL):
            # Record start nodes (these are the current position in the CEG)
            start_nodes = list(self.current_nodes)

            # Visit right side (termination condition) to create CEG node(s).
            # Force fresh intermediate nodes so ExprSequence doesn't reuse
            # the start nodes (which would make start == end).
            fresh_nodes = []
            for current in self.current_nodes:
                if current.condition is None and not self.graph.get_in_edges(current.id):
                    # Empty root node — ExprSequence would put the condition
                    # on it, making start==end. Create an intermediate node.
                    mid = self.graph.create_node()
                    self.graph.add_edge(current, mid, delay=TimeRange(min_delay="0", max_delay="0"))
                    fresh_nodes.append(mid)
                else:
                    fresh_nodes.append(current)
            self.current_nodes = fresh_nodes
            self.visit(node.right)
            end_nodes = list(self.current_nodes)

            # Record hold region for later conversion to lane_constraints
            self.hold_regions.append((start_nodes, node.left, end_nodes, node.op))

            if node.op == PropertyBinaryOperator.UNTIL:
                self.status = _merge_status(self.status, ExtractionStatus.LOSSY)
                self.notes.append("until excludes the terminating cycle and is shown as a hold-until skeleton")

            # Terminals of until_with are the right-side (termination) nodes
            self.current_nodes = end_nodes
        elif node.op in (PropertyBinaryOperator.AND, PropertyBinaryOperator.OR):
            start = list(self.current_nodes)
            self.visit(node.left)
            left_t = self.current_nodes
            self.current_nodes = start
            self.visit(node.right)
            right_t = self.current_nodes
            self.current_nodes = list(set(left_t) | set(right_t))
            self.status = _merge_status(self.status, ExtractionStatus.LOSSY)
            self.notes.append(f"property {node.op.value} is diagrammed as decomposed sub-properties")
        else:
            self.generic_visit(node)


_UNSUPPORTED_KEYWORDS = (
    "accept_on",
    "reject_on",
    "sync_accept_on",
    "sync_reject_on",
    "local",
    "var",
)


class _OrderedIdentifierVisitor(NodeVisitor[None]):
    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_Identifier(self, node: Identifier) -> None:
        if node.name not in self.names:
            self.names.append(node.name)
        return None


class _AstSignalMetadataVisitor(_OrderedIdentifierVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.bus_candidates: set[str] = set()

    def visit_BinaryExpr(self, node: BinaryExpr) -> None:
        if isinstance(node.left, Identifier) and node.op in {BinaryOperator.EQ, BinaryOperator.NE}:
            if emit_expr(node.right) not in {"0", "1"}:
                self.bus_candidates.add(node.left.name)
        self.generic_visit(node)
        return None

    def visit_CallExpr(self, node: CallExpr) -> None:
        if node.name in {"$stable", "$changed", "$past"}:
            for arg in node.args:
                if isinstance(arg, Identifier):
                    self.bus_candidates.add(arg.name)
        self.generic_visit(node)
        return None


class _AstExtractionStatusVisitor(NodeVisitor[None]):
    def __init__(self) -> None:
        self.status = ExtractionStatus.EXACT
        self.notes: list[str] = []

    def visit_IfElseProperty(self, node) -> None:
        self._mark_lossy("if/else property body preserved as lossy overlay")
        self.generic_visit(node)
        return None

    def visit_PropertyBinary(self, node: PropertyBinary) -> None:
        if node.op == PropertyBinaryOperator.UNTIL:
            self._mark_lossy("until excludes the terminating cycle and is shown as a hold-until skeleton")
        elif node.op in {PropertyBinaryOperator.AND, PropertyBinaryOperator.OR}:
            self._mark_lossy(f"property {node.op.value} is diagrammed as decomposed sub-properties")
        self.generic_visit(node)
        return None

    def visit_SequenceBinary(self, node: SequenceBinary) -> None:
        if node.op == SequenceBinaryOperator.INTERSECT:
            self._mark_lossy("intersect can admit multiple interval layouts; diagram is lossy")
        self.generic_visit(node)
        return None

    def visit_RepeatSequence(self, node: RepeatSequence) -> None:
        count_text = emit_expr(node.count.minimum)
        if node.count.unbounded:
            count_text = f"[{count_text}:$]"
        elif node.count.maximum is not None and emit_expr(node.count.maximum) != count_text:
            count_text = f"[{count_text}:{emit_expr(node.count.maximum)}]"

        if node.op == RepeatOperator.NON_CONSECUTIVE:
            self._mark_lossy(f"non-consecutive repetition {count_text} shown as a compressed hold region")
        elif node.op == RepeatOperator.GOTO:
            self._mark_lossy(f"goto repetition {count_text} shown as a compressed reachability region")
        self.generic_visit(node)
        return None

    def _mark_lossy(self, note: str) -> None:
        self.status = _merge_status(self.status, ExtractionStatus.LOSSY)
        if note not in self.notes:
            self.notes.append(note)


def _collect_parameters_from_ast(spec: PropertySpec) -> tuple[str, ...]:
    collector = _OrderedIdentifierVisitor()
    collector.visit(spec.body)
    params = []
    for name in collector.names:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", name) and name not in params:
            params.append(name)
    return tuple(params)


def _collect_projected_signal_names(
    anchors: Sequence[Anchor],
    constraints: Sequence[LaneConstraint],
) -> tuple[str, ...]:
    names: list[str] = []
    for anchor in anchors:
        for signal_name in collect_signals(anchor.condition):
            if signal_name and signal_name not in names:
                names.append(signal_name)
    for constraint in constraints:
        for signal_name in constraint.signals:
            if signal_name and signal_name not in names:
                names.append(signal_name)
    return tuple(names)


def _augment_signal_decls(
    signals: Sequence[SignalDecl],
    projected_signal_names: Sequence[str],
    parameter_names: Sequence[str],
    spec: PropertySpec | None,
    clock_signal: str,
) -> list[SignalDecl]:
    bus_candidates = set()
    if spec is not None:
        visitor = _AstSignalMetadataVisitor()
        visitor.visit(spec.body)
        bus_candidates.update(visitor.bus_candidates)

    ordered = {signal.name: signal for signal in signals}
    for signal_name in projected_signal_names:
        if not signal_name or signal_name in {clock_signal, *parameter_names}:
            continue
        if signal_name in ordered:
            current = ordered[signal_name]
            if current.kind == SignalKind.BIT and signal_name in bus_candidates:
                ordered[signal_name] = SignalDecl(name=current.name, kind=SignalKind.BUS, width=current.width, samples=current.samples)
            continue
        kind = SignalKind.BUS if signal_name in bus_candidates else SignalKind.BIT
        ordered[signal_name] = SignalDecl(name=signal_name, kind=kind)
    return list(ordered.values())


def _build_signal_decls_from_ast(structure, clock_signal: str, params: Sequence[str]) -> List[SignalDecl]:
    spec = getattr(structure, "ast", None)
    if spec is None:
        return _build_signal_decls(structure, getattr(structure, "body", ""), clock_signal)

    visitor = _AstSignalMetadataVisitor()
    visitor.visit(spec.body)
    excluded = CollectBoundNamesVisitor().visit(spec)
    excluded.update(params)
    excluded.add(clock_signal)

    signals = []
    seen = set()
    for signal_name in sorted(name for name in visitor.names if name and name not in excluded):
        if signal_name in seen:
            continue
        seen.add(signal_name)
        kind = SignalKind.BUS if signal_name in visitor.bus_candidates else SignalKind.BIT
        signals.append(SignalDecl(name=signal_name, kind=kind))
    return signals


def _extract_implication_from_ast(body) -> tuple[str, str, str] | None:
    if not isinstance(body, ImplicationProperty):
        return None
    return body.op.value, emit_sequence(body.antecedent), emit_property_body(body.consequent)


def extract_sva_scenario(sva_text: str, *, name: str | None = None) -> ScenarioDocument:
    """Extract one SVA property into a scenario document."""

    from sva_toolkit.formal.parse import parse_property

    structure = parse_property(sva_text)
    structure_ast = getattr(structure, "ast", None)
    property_body = getattr(structure, "body", "").strip().strip(";")
    property_name = name or getattr(structure, "name", None) or getattr(structure, "property_name", None) or "sva_property"
    normalized_body = _normalize_property_body(structure, property_body)

    clock_signal = getattr(structure, "clock_name", None) or "clk"
    clock_edge = getattr(structure, "clock_edge", None) or "posedge"
    reset_expr = getattr(structure, "reset_expr", None)
    if not getattr(structure, "has_explicit_reset", True):
        reset_expr = None
    clocking = ClockingSpec(
        edge=clock_edge,
        signal=clock_signal,
        disable_iff=reset_expr,
        disable_iff_ast=structure_ast.disable_iff if structure_ast is not None else None,
    )
    raw_parameter_names = (
        _collect_parameters_from_ast(structure_ast) if structure_ast is not None else _collect_parameters(normalized_body)
    )

    notes: List[str] = []
    status = ExtractionStatus.EXACT
    if structure_ast is not None:
        if getattr(structure_ast, "local_vars", ()) or ContainsNodeKindVisitor(ControlProperty).visit(structure_ast.body):
            status = ExtractionStatus.UNSUPPORTED
            notes.append("unsupported control wrapper or local sampled variable")
        elif ContainsNodeKindVisitor(FirstMatchSequence).visit(structure_ast.body):
            status = ExtractionStatus.UNSUPPORTED
            notes.append("first_match with branching is not diagrammed")
        else:
            classification = _AstExtractionStatusVisitor()
            classification.visit(structure_ast.body)
            status = _merge_status(status, classification.status)
            notes.extend(classification.notes)
    else:
        lower_body = normalized_body.lower()
        if any(keyword in lower_body for keyword in _UNSUPPORTED_KEYWORDS):
            status = ExtractionStatus.UNSUPPORTED
            notes.append("unsupported control wrapper or local sampled variable")
        elif "first_match" in lower_body:
            status = ExtractionStatus.UNSUPPORTED
            notes.append("first_match with branching is not diagrammed")

    anchors: List[Anchor] = []
    windows: List[TimeWindow] = []
    cuts: List[Cut] = []
    constraints: List[LaneConstraint] = []

    if status != ExtractionStatus.UNSUPPORTED:
        graph = CanonicalEventGraph()
        compiler = SvaCompiler(graph)
        try:
            compiler.compile(structure_ast.body if structure_ast is not None else parse_property_body(normalized_body))
            status = _merge_status(status, compiler.status)
            notes.extend(compiler.notes)

            solver = CEGSolver(graph)
            solver.solve()
            status = _merge_status(status, solver.status)
            notes.extend(solver.notes)

            dag_anchors, dag_windows, dag_constraints = solver.project(property_name)
            anchors.extend(dag_anchors)
            windows.extend(dag_windows)
            constraints.extend(dag_constraints)

            # Convert hold_regions (from until_with/until) to lane_constraints
            if compiler.hold_regions:
                node_to_anchor_name: dict[int, str] = {}
                for a in dag_anchors:
                    for tag in ("__node_", "__point_"):
                        if tag in a.name:
                            try:
                                node_id = int(a.name.rsplit(tag, 1)[1])
                                node_to_anchor_name[node_id] = a.name
                            except (ValueError, IndexError):
                                pass

                for start_nodes, left_ast, end_nodes, op in compiler.hold_regions:
                    start_anchor = None
                    end_anchor = None
                    for sn in start_nodes:
                        if sn.id in node_to_anchor_name:
                            start_anchor = node_to_anchor_name[sn.id]
                            break
                    for en in end_nodes:
                        if en.id in node_to_anchor_name:
                            end_anchor = node_to_anchor_name[en.id]
                            break
                    if not start_anchor or not end_anchor:
                        continue

                    # Add OMITTED hold window
                    windows.append(TimeWindow(
                        name=f"{property_name}__hold_{len(windows)}",
                        start_anchor=start_anchor,
                        end_anchor=end_anchor,
                        bound=TimeBound(kind=WindowBoundKind.OMITTED),
                    ))

                    # Convert left AST to condition text, then to lane_constraints
                    if isinstance(left_ast, ExprSequence):
                        left_text = emit_expr(left_ast.expr)
                    elif hasattr(left_ast, 'expr'):
                        left_text = emit_expr(left_ast.expr)
                    else:
                        left_text = emit_property_body(left_ast)
                    hold_constraints = _constraints_from_condition(
                        parse_sva_condition(left_text),
                        f"{property_name}__hold_{len(constraints)}",
                        ConstraintRegion.FROM_UNTIL,
                        start_anchor=start_anchor,
                        end_anchor=end_anchor,
                    )
                    constraints.extend(hold_constraints)

                    if op == PropertyBinaryOperator.UNTIL:
                        status = _merge_status(status, ExtractionStatus.LOSSY)
                        notes.append("until excludes the terminating cycle and is shown as a hold-until skeleton")
        except Exception as e:
            status = ExtractionStatus.UNSUPPORTED
            notes.append(f"DAG compilation failed: {str(e)}")

    if not anchors and status == ExtractionStatus.UNSUPPORTED:
        anchors.append(_make_anchor("unsupported", parse_sva_condition("1'b1"), AnchorRole.SYNTHETIC))
    elif status != ExtractionStatus.UNSUPPORTED:
        _add_focus_cuts(property_name, anchors, windows, cuts)

    projected_signal_names = _collect_projected_signal_names(anchors, constraints)
    parameter_names = tuple(name for name in raw_parameter_names if name not in projected_signal_names)
    params = tuple(ParameterDecl(name=token) for token in parameter_names)
    if structure_ast is not None:
        signals = tuple(
            _augment_signal_decls(
                _build_signal_decls_from_ast(structure, clock_signal, parameter_names),
                projected_signal_names,
                parameter_names,
                structure_ast,
                clock_signal,
            )
        )
    else:
        signals = tuple(
            _augment_signal_decls(
                _build_signal_decls(structure, normalized_body, clock_signal),
                projected_signal_names,
                parameter_names,
                None,
                clock_signal,
            )
        )

    property_overlay = PropertyOverlay(
        name=property_name,
        body=normalized_body,
        body_ast=structure_ast.body if structure_ast is not None else None,
        status=status,
        source=sva_text.strip(),
        related_anchors=tuple(anchor.name for anchor in anchors),
        related_windows=tuple(window.name for window in windows),
        related_constraints=tuple(constraint.name for constraint in constraints),
        notes=tuple(notes),
    )

    document = ScenarioDocument(
        name=property_name,
        clocking=clocking,
        params=params,
        signals=signals,
        anchors=tuple(anchors),
        windows=tuple(windows),
        cuts=tuple(cuts),
        lane_constraints=tuple(constraints),
        properties=(property_overlay,),
        extraction_status=status,
        notes=tuple(notes),
        ceg=graph if status != ExtractionStatus.UNSUPPORTED else None,
    )

    if status != ExtractionStatus.UNSUPPORTED:
        document = _try_refine_with_witness(structure, document)

    return document


def extract_sva_scenarios(sva_texts: Iterable[str]) -> tuple[ScenarioDocument, ...]:
    """Extract multiple SVA properties into independent scenario documents."""

    return tuple(extract_sva_scenario(sva_text) for sva_text in sva_texts)


def bundle_sva_scenarios(
    documents: Sequence[ScenarioDocument],
    *,
    overlap_threshold: float = 0.35,
    max_signals: int = 10,
    max_properties: int = 5,
    max_chains: int = 3,
) -> tuple[ScenarioDocument, ...]:
    """Group compatible scenario documents into deterministic bundles."""

    if not documents:
        return ()

    key_groups: dict[tuple[str, str, str], list[ScenarioDocument]] = {}
    for document in documents:
        key = (
            document.clocking.edge,
            document.clocking.signal,
            document.clocking.disable_iff or "",
        )
        key_groups.setdefault(key, []).append(document)

    bundled: List[ScenarioDocument] = []
    for group_documents in key_groups.values():
        ordered = sorted(group_documents, key=lambda document: document.name)
        while ordered:
            seed = ordered.pop(0)
            cluster = [seed]
            seed_signals = {signal.name for signal in seed.signals}
            remaining = []
            for candidate in ordered:
                candidate_signals = {signal.name for signal in candidate.signals}
                overlap = _signal_overlap(seed_signals, candidate_signals)
                if overlap >= overlap_threshold and _compatible_documents(cluster, candidate):
                    cluster.append(candidate)
                    seed_signals |= candidate_signals
                else:
                    remaining.append(candidate)
            ordered = remaining
            if len(cluster) == 1:
                bundled.append(cluster[0])
                continue
            if len(seed_signals) > max_signals or sum(len(doc.properties) for doc in cluster) > max_properties:
                bundled.extend(cluster)
                continue
            if _independent_chain_count(cluster) > max_chains:
                bundled.extend(cluster)
                continue
            bundled.append(_merge_cluster(tuple(cluster)))
    return tuple(bundled)


def _extract_sequence(
    property_name: str,
    expression: str,
    start_anchor_name: str,
    anchors: List[Anchor],
    windows: List[TimeWindow],
    cuts: List[Cut],
    constraints: List[LaneConstraint],
) -> tuple[ExtractionStatus, list[str]]:
    expression = _strip_outer_parens(expression.strip())
    notes: List[str] = []
    status = ExtractionStatus.EXACT

    if not expression:
        return ExtractionStatus.UNSUPPORTED, ["empty consequent"]

    if expression.lower().startswith("if "):
        return ExtractionStatus.LOSSY, ["if/else property body preserved as lossy overlay"]

    for op in (" and ", " or "):
        parts = _split_top_level_word(expression, op.strip())
        if len(parts) > 1:
            child_statuses = []
            for index, part in enumerate(parts):
                child_status, child_notes = _extract_sequence(
                    property_name,
                    part,
                    start_anchor_name,
                    anchors,
                    windows,
                    cuts,
                    constraints,
                )
                child_statuses.append(child_status)
                notes.extend(child_notes)
                if op.strip() == "or":
                    notes.append(f"branch {index + 1} retained as one alternative")
            return _merge_status(ExtractionStatus.LOSSY, *child_statuses), notes

    until_split = _split_until(expression)
    if until_split is not None:
        left_text, op, right_text = until_split
        end_anchor = _append_condition_anchor(
            anchors,
            f"{property_name}__until_end",
            parse_sva_condition(right_text),
            AnchorRole.RESPONSE,
        )
        windows.append(
            TimeWindow(
                name=f"{property_name}__hold_window",
                start_anchor=start_anchor_name,
                end_anchor=end_anchor.name,
                bound=TimeBound(kind=WindowBoundKind.OMITTED),
            )
        )
        constraints.extend(
            _constraints_from_condition(
                parse_sva_condition(left_text),
                f"{property_name}__hold",
                ConstraintRegion.FROM_UNTIL,
                start_anchor=start_anchor_name,
                end_anchor=end_anchor.name,
            )
        )
        if op == "until":
            status = ExtractionStatus.LOSSY
            notes.append("until excludes the terminating cycle and is shown as a hold-until skeleton")
        return status, notes

    throughout_split = _split_top_level_word(expression, "throughout")
    if len(throughout_split) > 1:
        left_text = throughout_split[0]
        right_text = throughout_split[1]
        child_status, child_notes = _extract_sequence(
            property_name,
            right_text,
            start_anchor_name,
            anchors,
            windows,
            cuts,
            constraints,
        )
        notes.extend(child_notes)
        target_window = windows[-1].name if windows else None
        if target_window is None:
            return ExtractionStatus.UNSUPPORTED, ["throughout target did not produce a window"]
        constraints.extend(
            _constraints_from_condition(
                parse_sva_condition(left_text),
                f"{property_name}__throughout",
                ConstraintRegion.IN,
                window=target_window,
            )
        )
        return child_status, notes

    intersect_split = _split_top_level_word(expression, "intersect")
    if len(intersect_split) > 1:
        child_statuses = []
        notes.append("intersect can admit multiple interval layouts; diagram is lossy")
        for part in intersect_split:
            child_status, child_notes = _extract_sequence(
                property_name,
                part,
                start_anchor_name,
                anchors,
                windows,
                cuts,
                constraints,
            )
            child_statuses.append(child_status)
            notes.extend(child_notes)
        return _merge_status(ExtractionStatus.LOSSY, *child_statuses), notes

    delay_match = _match_leading_delay(expression)
    if delay_match is not None:
        delay_token, rest = delay_match
        end_anchor = _append_condition_anchor(
            anchors,
            f"{property_name}__response",
            _extract_anchor_condition(rest),
            AnchorRole.RESPONSE,
        )
        bound = _delay_to_bound(delay_token)
        window_name = f"{property_name}__delay_window"
        windows.append(TimeWindow(name=window_name, start_anchor=start_anchor_name, end_anchor=end_anchor.name, bound=bound))
        if bound.kind == WindowBoundKind.UNBOUNDED:
            cuts.append(
                Cut(
                    name=f"{property_name}__future_cut",
                    placement=CutPlacement.AFTER_ANCHOR,
                    meaning=CutMeaning.OMITTED_FUTURE,
                    anchor=end_anchor.name,
                    label=bound.label,
                )
            )
        repeat_status, repeat_notes = _extract_repeat_overlay(rest, property_name, start_anchor_name, end_anchor.name, constraints)
        notes.extend(repeat_notes)
        return _merge_status(status, repeat_status), notes

    repeat_status, repeat_notes = _extract_repeat_overlay(expression, property_name, start_anchor_name, None, constraints)
    if repeat_notes or repeat_status != ExtractionStatus.EXACT:
        notes.extend(repeat_notes)
        return repeat_status, notes

    response_anchor = _append_condition_anchor(
        anchors,
        f"{property_name}__response",
        _extract_anchor_condition(expression),
        AnchorRole.RESPONSE,
    )
    windows.append(
        TimeWindow(
            name=f"{property_name}__instant_window",
            start_anchor=start_anchor_name,
            end_anchor=response_anchor.name,
            bound=TimeBound(kind=WindowBoundKind.EXACT, min_delay="0", max_delay="0"),
        )
    )
    return status, notes


def _extract_repeat_overlay(
    expression: str,
    property_name: str,
    start_anchor_name: str,
    end_anchor_name: str | None,
    constraints: List[LaneConstraint],
) -> tuple[ExtractionStatus, list[str]]:
    repeat_match = _match_repetition(expression)
    if repeat_match is None:
        return ExtractionStatus.EXACT, []

    base_text, op, count_text = repeat_match
    relation_constraints = _constraints_from_condition(
        parse_sva_condition(base_text),
        f"{property_name}__repeat",
        ConstraintRegion.FROM_UNTIL,
        start_anchor=start_anchor_name,
        end_anchor=end_anchor_name or start_anchor_name,
    )
    constraints.extend(relation_constraints)

    if op == "[*":
        return (ExtractionStatus.EXACT, [])
    if op == "[=":
        return (ExtractionStatus.LOSSY, [f"non-consecutive repetition {count_text} shown as a compressed hold region"])
    return (ExtractionStatus.LOSSY, [f"goto repetition {count_text} shown as a compressed reachability region"])


def _constraints_from_condition(
    condition: Condition, prefix: str, region: ConstraintRegion, **region_kwargs
) -> List[LaneConstraint]:
    constraints = []
    predicates = _condition_predicates(condition)
    if not predicates:
        text = condition_to_sva(condition)
        if text:
            constraints.append(
                LaneConstraint(
                    name=f"{prefix}_raw",
                    signals=tuple(sorted(collect_signals(condition))),
                    relation="raw",
                    value=text,
                    region=region,
                    **region_kwargs,
                )
            )
        return constraints
    for index, predicate in enumerate(predicates):
        constraints.append(
            LaneConstraint(
                name=f"{prefix}_{index}",
                signals=(predicate.signal,),
                relation=predicate.op,
                value=predicate.value,
                region=region,
                **region_kwargs,
            )
        )
    return constraints


def _condition_predicates(condition: Condition) -> list[Predicate]:
    if condition.kind == "predicate" and condition.predicate is not None and condition.predicate.signal:
        return [condition.predicate]
    if condition.kind == "all":
        predicates = []
        for item in condition.items:
            predicates.extend(_condition_predicates(item))
        return predicates
    return []


def _extract_anchor_condition(expression: str) -> Condition:
    expression = _strip_outer_parens(expression.strip())
    repeat_match = _match_repetition(expression)
    if repeat_match is not None:
        expression = repeat_match[0]
    return parse_sva_condition(expression)


def _append_condition_anchor(anchors: List[Anchor], base_name: str, condition: Condition, role: AnchorRole) -> Anchor:
    existing_names = {anchor.name for anchor in anchors}
    candidate = base_name
    index = 2
    while candidate in existing_names:
        candidate = f"{base_name}_{index}"
        index += 1
    anchor = _make_anchor(candidate, condition, role)
    anchors.append(anchor)
    if _condition_uses_past(condition):
        anchors.append(_make_anchor(f"{candidate}__lookback", condition, AnchorRole.LOOKBACK))
    return anchor


def _condition_uses_past(condition: Condition) -> bool:
    if condition.kind == "predicate" and condition.predicate is not None:
        return condition.predicate.op == "past"
    return any(_condition_uses_past(item) for item in condition.items)


def _make_anchor(name: str, condition: Condition, role: AnchorRole) -> Anchor:
    return Anchor(name=name, condition=condition, role=role)


def _add_focus_cuts(property_name: str, anchors: List[Anchor], windows: List[TimeWindow], cuts: List[Cut]) -> None:
    if not windows:
        return

    anchor_map = {anchor.name: anchor for anchor in anchors}
    focus_start_name = windows[0].start_anchor
    focus_start = anchor_map.get(focus_start_name)
    if (
        focus_start is not None
        and not (
            focus_start.role == AnchorRole.SYNTHETIC
            and condition_to_sva(focus_start.condition).strip() == "1'b1"
        )
        and not any(cut.placement == CutPlacement.BEFORE_ANCHOR and cut.anchor == focus_start_name for cut in cuts)
    ):
        cuts.insert(
            0,
            Cut(
                name=f"{property_name}__history_cut",
                placement=CutPlacement.BEFORE_ANCHOR,
                meaning=CutMeaning.OMITTED_HISTORY,
                anchor=focus_start_name,
            ),
        )

    start_anchor_names = {window.start_anchor for window in windows}
    terminal_anchor_names = []
    for window in windows:
        if window.end_anchor not in start_anchor_names and window.end_anchor not in terminal_anchor_names:
            terminal_anchor_names.append(window.end_anchor)
    if not terminal_anchor_names:
        terminal_anchor_names.append(windows[-1].end_anchor)

    for anchor_name in enumerate(terminal_anchor_names):
        anchor_name = anchor_name[1]
        if any(cut.placement == CutPlacement.AFTER_ANCHOR and cut.anchor == anchor_name for cut in cuts):
            continue
        cut_name = f"{property_name}__future_cut" if len(terminal_anchor_names) == 1 else f"{anchor_name}__future_cut"
        cuts.append(
            Cut(
                name=cut_name,
                placement=CutPlacement.AFTER_ANCHOR,
                meaning=CutMeaning.OMITTED_FUTURE,
                anchor=anchor_name,
            )
        )


def _delay_to_bound(delay_token: str) -> TimeBound:
    token = delay_token.replace(" ", "")
    if re.fullmatch(r"##\d+", token):
        value = token[2:]
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=value, max_delay=value)
    match = re.fullmatch(r"##\[(.+):(.+)\]", token)
    if not match:
        return TimeBound(kind=WindowBoundKind.OMITTED)
    min_delay, max_delay = match.groups()
    if max_delay == "$":
        return TimeBound(kind=WindowBoundKind.UNBOUNDED, min_delay=min_delay, max_delay=max_delay)
    if min_delay == max_delay:
        return TimeBound(kind=WindowBoundKind.EXACT, min_delay=min_delay, max_delay=max_delay)
    return TimeBound(kind=WindowBoundKind.RANGE, min_delay=min_delay, max_delay=max_delay)


def _match_leading_delay(expression: str) -> tuple[str, str] | None:
    if not expression.startswith("##"):
        return None
    match = re.match(r"^(##(?:\[\s*[^]]+\s*\]|\d+))\s+(.+)$", expression)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def _match_repetition(expression: str) -> tuple[str, str, str] | None:
    expression = _strip_outer_parens(expression)
    if not expression.endswith("]"):
        return None
    depth = 0
    for index in range(len(expression) - 1, -1, -1):
        char = expression[index]
        if char == "]":
            depth += 1
        elif char == "[":
            depth -= 1
            if depth == 0:
                content = expression[index + 1:-1].strip()
                if content.startswith("*"):
                    return expression[:index].strip(), "[*", content[1:].strip()
                if content.startswith("="):
                    return expression[:index].strip(), "[=", content[1:].strip()
                if content.startswith("->"):
                    return expression[:index].strip(), "[->", content[2:].strip()
                return None
    return None


def _split_until(expression: str) -> tuple[str, str, str] | None:
    for op in ("until_with", "until"):
        parts = _split_top_level_word(expression, op)
        if len(parts) > 1:
            return parts[0], op, parts[1]
    return None


def _split_top_level_word(expression: str, word: str) -> tuple[str, ...]:
    token = f" {word} "
    parts = []
    depth_paren = 0
    depth_brack = 0
    start = 0
    index = 0
    while index <= len(expression) - len(token):
        char = expression[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        if depth_paren == 0 and depth_brack == 0 and expression[index:index + len(token)] == token:
            parts.append(expression[start:index].strip())
            start = index + len(token)
            index = start
            continue
        index += 1
    if not parts:
        return (expression.strip(),)
    parts.append(expression[start:].strip())
    return tuple(part for part in parts if part)


def _strip_outer_parens(expression: str) -> str:
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        wrapped = True
        for index, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(expression) - 1:
                    wrapped = False
                    break
        if not wrapped:
            break
        expression = expression[1:-1].strip()
    return expression


def _collect_parameters(property_body: str) -> tuple[str, ...]:
    keywords = {
        "and",
        "or",
        "throughout",
        "until",
        "until_with",
        "intersect",
        "first_match",
        "disable",
        "iff",
        "posedge",
        "negedge",
    }
    params = []
    for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", property_body):
        if token not in keywords and token not in params:
            params.append(token)
    return tuple(params)


def _normalize_property_body(structure, property_body: str) -> str:
    normalized = property_body.strip()
    implication = _split_implication(normalized)
    if implication is not None:
        op, antecedent, consequent = implication
        return f"{antecedent} {op} {consequent}"
    normalized = re.sub(r"@\s*\([^)]+\)", "", normalized)
    normalized = re.sub(r"disable\s+iff\s*\([^)]+\)", "", normalized)
    return normalized.strip()


def _build_signal_decls(structure, property_body: str, clock_signal: str) -> List[SignalDecl]:
    bus_candidates = set()
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=)\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+'[bdhoBDHO][0-9A-Fa-f_xzXZ]+|\w+)",
        property_body,
    ):
        if match.group(3) not in {"0", "1"}:
            bus_candidates.add(match.group(1))
    for match in re.finditer(
        r"(?<![A-Za-z0-9_$])\$(?:stable|changed|past)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
        property_body,
    ):
        bus_candidates.add(match.group(1))

    reset_signal = None
    if getattr(structure, "has_explicit_reset", False):
        reset_signal = _extract_reset_signal_name(getattr(structure, "reset_expr", None))
    source_signals = [_coerce_signal_name(signal) for signal in getattr(structure, "signals", ())]
    if not source_signals:
        source_signals = list(_collect_signal_names(property_body))

    signals = []
    seen = set()
    for signal_name in sorted(name for name in source_signals if name):
        if signal_name in {clock_signal, reset_signal} or signal_name in seen:
            continue
        seen.add(signal_name)
        kind = SignalKind.BUS if signal_name in bus_candidates else SignalKind.BIT
        signals.append(SignalDecl(name=signal_name, kind=kind))
    return signals


def _split_implication(property_body: str) -> tuple[str, str, str] | None:
    depth_paren = 0
    depth_brack = 0
    for index in range(len(property_body) - 2):
        char = property_body[index]
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren = max(0, depth_paren - 1)
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack = max(0, depth_brack - 1)
        if depth_paren == 0 and depth_brack == 0:
            token = property_body[index:index + 3]
            if token in {"|->", "|=>"}:
                return token, property_body[:index].strip(), property_body[index + 3:].strip()
    return None


def _has_temporal_skeleton(expression: str) -> bool:
    lowered = expression.lower()
    if "##" in expression:
        return True
    if any(f" {token} " in lowered for token in ("throughout", "until", "until_with", "intersect", "and", "or")):
        return True
    return bool(re.search(r"\[(?:\*|=|->)", expression))


def _extract_reset_signal_name(reset_expr: str | None) -> str | None:
    if not reset_expr:
        return None
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", reset_expr)
    if not match:
        return None
    return match.group(0)


def _collect_signal_names(property_body: str) -> tuple[str, ...]:
    reserved = {
        "and",
        "or",
        "throughout",
        "until",
        "until_with",
        "intersect",
        "disable",
        "iff",
        "posedge",
        "negedge",
        "if",
        "else",
        "not",
    }
    names = []
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", property_body):
        if token in reserved or token.startswith("$"):
            continue
        if token not in names:
            names.append(token)
    return tuple(names)


def _coerce_signal_name(signal) -> str | None:
    if isinstance(signal, str):
        return signal
    return getattr(signal, "name", None)


def _merge_status(*statuses: ExtractionStatus) -> ExtractionStatus:
    if ExtractionStatus.UNSUPPORTED in statuses:
        return ExtractionStatus.UNSUPPORTED
    if ExtractionStatus.LOSSY in statuses:
        return ExtractionStatus.LOSSY
    return ExtractionStatus.EXACT


def _signal_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _compatible_documents(cluster: Sequence[ScenarioDocument], candidate: ScenarioDocument) -> bool:
    for existing in cluster:
        if existing.clocking != candidate.clocking:
            return False
        if _has_conflicting_constraints(existing, candidate):
            return False
    return True


def _has_conflicting_constraints(left: ScenarioDocument, right: ScenarioDocument) -> bool:
    left_facts = {
        _constraint_identity(constraint): constraint.value
        for constraint in left.lane_constraints
        if constraint.relation in {"eq", "high", "low"}
    }
    right_facts = {
        _constraint_identity(constraint): constraint.value
        for constraint in right.lane_constraints
        if constraint.relation in {"eq", "high", "low"}
    }
    for key, value in left_facts.items():
        if key in right_facts and right_facts[key] != value:
            return True
    return False


def _constraint_identity(constraint: LaneConstraint) -> tuple:
    return (
        tuple(constraint.signals),
        constraint.region,
        constraint.anchor,
        constraint.window,
        constraint.start_anchor,
        constraint.end_anchor,
    )


def _independent_chain_count(documents: Sequence[ScenarioDocument]) -> int:
    chains = 0
    for document in documents:
        chains += max(1, len(document.properties))
    return chains


def _merge_cluster(documents: tuple[ScenarioDocument, ...]) -> ScenarioDocument:
    primary = documents[0]
    
    # 1. Collect and Merge CEGs
    cegs = [doc.ceg for doc in documents if doc.ceg is not None]
    if not cegs:
        # Fallback to old behavior if no CEGs are present
        return _merge_cluster_legacy(documents)
        
    merged_ceg = CEGSolver.merge_graphs(cegs)
    solver = CEGSolver(merged_ceg)
    solver.solve()
    
    full_name = "__".join(doc.name for doc in documents)
    if len(full_name) <= 60:
        new_name = full_name
    else:
        new_name = f"{documents[0].name}__bundle_{len(documents)}"
    anchors, windows, constraints = solver.project(new_name)

    # Carry over FROM_UNTIL/BEFORE lane_constraints from source documents
    # by remapping their anchor references to the new merged anchor names.
    merged_cond_map: dict[tuple[str, int | None], str] = {}
    for a in anchors:
        cond_text = condition_to_sva(a.condition) if a.condition else "1'b1"
        merged_cond_map[(cond_text, a.absolute_tick)] = a.name

    anchor_rename: dict[str, str] = {}
    for doc in documents:
        for a in doc.anchors:
            cond_text = condition_to_sva(a.condition) if a.condition else "1'b1"
            key = (cond_text, a.absolute_tick)
            if key in merged_cond_map:
                anchor_rename[a.name] = merged_cond_map[key]

    # Compute the set of valid anchor names in the merged document
    merged_anchor_names = {a.name for a in anchors}

    for doc in documents:
        for lc in doc.lane_constraints:
            if lc.region == ConstraintRegion.AT:
                continue  # AT constraints are re-created by project()
            new_start = anchor_rename.get(lc.start_anchor, lc.start_anchor) if lc.start_anchor else None
            new_end = anchor_rename.get(lc.end_anchor, lc.end_anchor) if lc.end_anchor else None
            new_anchor = anchor_rename.get(lc.anchor, lc.anchor) if lc.anchor else None
            # Drop constraints that reference nonexistent anchors
            if new_anchor and new_anchor not in merged_anchor_names:
                continue
            if new_start and new_start not in merged_anchor_names:
                continue
            if new_end and new_end not in merged_anchor_names:
                continue
            new_lc = replace(lc, anchor=new_anchor, start_anchor=new_start, end_anchor=new_end)
            constraints.append(new_lc)

    # Also carry over hold windows (OMITTED bound) from source documents,
    # but only when both anchors can be resolved to valid merged anchors.
    for doc in documents:
        for w in doc.windows:
            if w.bound.kind == WindowBoundKind.OMITTED:
                new_start = anchor_rename.get(w.start_anchor, w.start_anchor)
                new_end = anchor_rename.get(w.end_anchor, w.end_anchor)
                if new_start not in merged_anchor_names or new_end not in merged_anchor_names:
                    continue  # anchor no longer exists in the merged document
                windows.append(replace(w, start_anchor=new_start, end_anchor=new_end))

    # 2. Merge signals and parameters — prefer BUS over BIT, strip per-property
    # samples (the joint witness will repopulate them if it succeeds).
    merged_signals_map: dict[str, SignalDecl] = {}
    for doc in documents:
        for s in doc.signals:
            s_clean = replace(s, samples=()) if s.samples else s
            if s.name not in merged_signals_map:
                merged_signals_map[s.name] = s_clean
            else:
                existing = merged_signals_map[s.name]
                if existing.kind == SignalKind.BIT and s.kind == SignalKind.BUS:
                    merged_signals_map[s.name] = s_clean
    merged_signals = list(merged_signals_map.values())
                
    merged_params = []
    seen_params = set()
    for doc in documents:
        for p in doc.params:
            if p.name not in seen_params:
                merged_params.append(p)
                seen_params.add(p.name)

    # 3. Merge properties with updated related refs
    # Note: For an industrial tool, we'd need a stable node mapping back to properties.
    # Here we use the fact that project() uses a deterministic naming scheme.
    merged_properties = []
    for doc in documents:
        for prop in doc.properties:
            # We don't rename the property itself, but we should update its related refs
            # Project names anchors as {prefix}__node_{id}
            # For simplicity, we just keep them for now, but in a real tool, 
            # we'd map original anchor names to the new merged ones.
            merged_properties.append(prop)

    overlap = 1.0
    if len(documents) > 1:
        overlap = _signal_overlap(
            {signal.name for signal in documents[0].signals},
            {signal.name for signal in documents[1].signals},
        )

    merged_doc = ScenarioDocument(
        name=new_name,
        clocking=primary.clocking,
        params=tuple(merged_params),
        signals=tuple(merged_signals),
        anchors=tuple(anchors),
        windows=tuple(windows),
        lane_constraints=tuple(constraints),
        properties=tuple(merged_properties),
        extraction_status=_merge_status(*(doc.effective_status for doc in documents)),
        bundle=merge_bundle_metadata(documents, overlap),
        notes=tuple(set(note for doc in documents for note in doc.notes)),
        ceg=merged_ceg,
    )

    merged_doc = _try_joint_witness(documents, merged_doc)
    return merged_doc


def _signal_widths_from_doc(document: ScenarioDocument) -> dict[str, int]:
    widths: dict[str, int] = {}
    for s in document.signals:
        if s.width is not None:
            try:
                widths[s.name] = int(s.width)
            except ValueError:
                widths[s.name] = 8
        elif s.kind == SignalKind.BUS:
            widths[s.name] = 8
    return widths


def _doc_to_formal_property(doc: ScenarioDocument):
    """Return a FormalProperty for the first property overlay in doc, or None."""
    from sva_toolkit.formal.parse import parse_property
    if not doc.properties:
        return None
    overlay = doc.properties[0]
    source = overlay.source
    if source:
        try:
            return parse_property(source)
        except Exception:
            pass
    if overlay.body:
        from sva_toolkit.formal.model import FormalProperty as _FP
        return _FP(
            body=overlay.body,
            clock_edge=doc.clocking.edge,
            clock_name=doc.clocking.signal,
            reset_expr=doc.clocking.disable_iff or "!rst_n",
        )
    return None


def _try_refine_with_witness(
    formal_prop,
    document: ScenarioDocument,
) -> ScenarioDocument:
    """Attempt EBMC witness refinement on a single-property document."""
    synthesizer = EbmcWitnessSynthesizer()
    if not synthesizer.available:
        return document
    signal_widths = _signal_widths_from_doc(document)
    try:
        trace = synthesizer.synthesize(formal_prop, signal_widths=signal_widths)
    except Exception:
        return document
    if trace is None:
        return document
    return refine_document_from_witness(document, trace)


def _try_joint_witness(
    documents: tuple[ScenarioDocument, ...],
    merged_doc: ScenarioDocument,
) -> ScenarioDocument:
    """Attempt joint EBMC witness synthesis for a bundle of documents."""
    synthesizer = EbmcWitnessSynthesizer()
    if not synthesizer.available:
        return merged_doc
    props = []
    for doc in documents:
        fp = _doc_to_formal_property(doc)
        if fp is not None:
            props.append(fp)
    if not props:
        return merged_doc
    # Linear causal chain: 0→1→2→...→N-1
    causal_order = [(i, i + 1) for i in range(len(props) - 1)]
    signal_widths = _signal_widths_from_doc(merged_doc)
    try:
        trace = synthesizer.synthesize_joint(
            props,
            causal_order=causal_order,
            signal_widths=signal_widths,
        )
    except Exception:
        return merged_doc
    if trace is None:
        return merged_doc
    return refine_document_from_witness(merged_doc, trace)


def _merge_cluster_legacy(documents: tuple[ScenarioDocument, ...]) -> ScenarioDocument:
    primary = documents[0]
    rename_maps = []
    merged_signals = []
    merged_anchors = []
    merged_windows = []
    merged_cuts = []
    merged_constraints = []
    merged_properties = []
    signal_names = set()
    reserved_names = {param.name for document in documents for param in document.params}
    reserved_names.update(signal.name for document in documents for signal in document.signals)

    for document in documents:
        rename_map = {}
        for collection in (
            document.anchors,
            document.windows,
            document.cuts,
            document.lane_constraints,
            document.properties,
        ):
            for item in collection:
                rename_map[item.name] = _reserve_unique_name(
                    item.name,
                    document.name,
                    reserved_names,
                )
        rename_maps.append(rename_map)

    for document, rename_map in zip(documents, rename_maps):
        for signal in document.signals:
            if signal.name not in signal_names:
                merged_signals.append(signal)
                signal_names.add(signal.name)
        for anchor in document.anchors:
            merged_anchors.append(replace(anchor, name=rename_map.get(anchor.name, anchor.name)))
        for window in document.windows:
            merged_windows.append(
                replace(
                    window,
                    name=rename_map.get(window.name, window.name),
                    start_anchor=rename_map.get(window.start_anchor, window.start_anchor),
                    end_anchor=rename_map.get(window.end_anchor, window.end_anchor),
                )
            )
        for cut in document.cuts:
            merged_cuts.append(
                replace(
                    cut,
                    name=rename_map.get(cut.name, cut.name),
                    anchor=rename_map.get(cut.anchor, cut.anchor) if cut.anchor else None,
                    left_window=rename_map.get(cut.left_window, cut.left_window) if cut.left_window else None,
                    right_window=rename_map.get(cut.right_window, cut.right_window) if cut.right_window else None,
                )
            )
        for constraint in document.lane_constraints:
            merged_constraints.append(
                replace(
                    constraint,
                    name=rename_map.get(constraint.name, constraint.name),
                    anchor=rename_map.get(constraint.anchor, constraint.anchor) if constraint.anchor else None,
                    window=rename_map.get(constraint.window, constraint.window) if constraint.window else None,
                    start_anchor=rename_map.get(constraint.start_anchor, constraint.start_anchor)
                    if constraint.start_anchor
                    else None,
                    end_anchor=rename_map.get(constraint.end_anchor, constraint.end_anchor)
                    if constraint.end_anchor
                    else None,
                )
            )
        for prop in document.properties:
            body_ast = prop.body_ast
            if body_ast is not None:
                body_ast = RenameIdentifiersTransformer(rename_map).visit(body_ast)
            merged_properties.append(
                replace(
                    prop,
                    name=rename_map.get(prop.name, prop.name),
                    body=emit_property_body(body_ast) if body_ast is not None else rebind_names(rename_map, prop.body),
                    body_ast=body_ast,
                    related_anchors=tuple(rename_map.get(anchor, anchor) for anchor in prop.related_anchors),
                    related_windows=tuple(rename_map.get(window, window) for window in prop.related_windows),
                    related_constraints=tuple(rename_map.get(item, item) for item in prop.related_constraints),
                )
            )

    overlap = 1.0
    if len(documents) > 1:
        overlap = _signal_overlap(
            {signal.name for signal in documents[0].signals},
            {signal.name for signal in documents[1].signals},
        )

    return ScenarioDocument(
        name="__".join(document.name for document in documents),
        clocking=primary.clocking,
        params=tuple(dict.fromkeys(param for document in documents for param in document.params)),
        signals=tuple(merged_signals),
        anchors=tuple(merged_anchors),
        windows=tuple(merged_windows),
        cuts=tuple(merged_cuts),
        lane_constraints=tuple(merged_constraints),
        properties=tuple(merged_properties),
        extraction_status=_merge_status(*(document.effective_status for document in documents)),
        bundle=merge_bundle_metadata(documents, overlap),
        notes=tuple(note for document in documents for note in document.notes),
    )


def _reserve_unique_name(name: str, document_name: str, reserved_names: set[str]) -> str:
    if name not in reserved_names:
        reserved_names.add(name)
        return name

    base_name = f"{document_name}__{name}"
    candidate = base_name
    index = 2
    while candidate in reserved_names:
        candidate = f"{base_name}_{index}"
        index += 1
    reserved_names.add(candidate)
    return candidate
