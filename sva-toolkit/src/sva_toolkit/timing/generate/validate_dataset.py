"""Dataset utility validation for generated timing Image-DSL records."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.core.scenario import (
    ConstraintRegion,
    LaneConstraint,
    ScenarioDocument,
    SignalKind,
    TimeBound,
)
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.frontend.validate import validate_diagram
from sva_toolkit.timing.generate.coverage import CoverageTracker
from sva_toolkit.timing.generate.waveform import _constraint_holds
from sva_toolkit.timing.projection.wavedrom_view import evaluate_condition


_DIVERSITY_THRESHOLDS = {
    "tick_count": 2,
    "lane_count": 2,
    "cut": 2,
    "rendering": 2,
    "predicate": 3,
    "region": 3,
}
_RESPONSE_OVERLAY_CLASSES = ("timing-response-arrow", "timing-rule-overlays", "timing-hold-fill")


def validate_dataset(
    dataset: str | Path,
    *,
    coverage_thresholds: Mapping[str, int] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate generated records for syntactic, semantic, visual, and utility issues."""

    dataset_path = Path(dataset)
    records_path = dataset_path / "records.jsonl"
    if not records_path.is_file():
        raise FileNotFoundError(f"dataset records file not found: {records_path}")

    failures: list[dict[str, Any]] = []
    failed_ids: set[str] = set()
    useless_diagrams: list[dict[str, Any]] = []
    skipped_symbolic: list[dict[str, str]] = []
    coverage_values: dict[str, set[str]] = defaultdict(set)
    canonical_hashes: dict[str, list[str]] = defaultdict(list)
    svg_hashes: dict[str, list[str]] = defaultdict(list)
    feature_signature_distribution: Counter[str] = Counter()
    records_total = 0

    def add_failure(record_id: str, reason: str, detail: str) -> None:
        failures.append({"id": record_id, "reason": reason, "detail": detail})
        failed_ids.add(record_id)

    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        records_total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            record_id = f"line_{line_number}"
            add_failure(record_id, "record_json", str(exc))
            continue

        record_id = str(record.get("id") or f"line_{line_number}")
        features = record.get("features") or {}
        if isinstance(features, dict):
            _record_coverage_values(features, coverage_values)
            feature_signature_distribution[_feature_signature_hash(features)] += 1

        canonical_dsl = (record.get("target") or {}).get("canonical_dsl")
        if not isinstance(canonical_dsl, str) or not canonical_dsl.strip():
            add_failure(record_id, "dsl_missing", "record target is missing canonical_dsl")
            continue

        canonical_hashes[_sha256(canonical_dsl)].append(record_id)
        _validate_dsl_file(dataset_path, record, canonical_dsl, record_id, add_failure)

        parsed: ScenarioDocument | None = None
        try:
            parsed = parse_diagram(canonical_dsl)
        except Exception as exc:
            add_failure(record_id, "dsl_syntax", f"{exc.__class__.__name__}: {exc}")
            continue

        try:
            roundtrip = emit_timing_dsl(parsed)
            if roundtrip != canonical_dsl:
                add_failure(record_id, "dsl_roundtrip", "emit_timing_dsl(parse(canonical_dsl)) changed the text")
        except Exception as exc:
            add_failure(record_id, "dsl_roundtrip", f"{exc.__class__.__name__}: {exc}")

        try:
            validate_diagram(parsed)
        except Exception as exc:
            add_failure(record_id, "dsl_semantic", f"{exc.__class__.__name__}: {exc}")

        _validate_samples(parsed, record_id, add_failure)
        if _has_any_samples(parsed):
            occurrences = _anchor_occurrences(parsed, record_id, add_failure, useless_diagrams)
            _validate_windows(parsed, occurrences, record_id, add_failure, strict=strict)
            _validate_constraints(parsed, occurrences, record_id, add_failure, strict=strict)
        else:
            skipped_symbolic.append({"id": record_id, "reason": "partial_or_missing_samples"})

        svg_text = _read_svg(dataset_path, record, record_id, add_failure)
        if svg_text is not None:
            svg_hashes[_sha256(svg_text)].append(record_id)
            _validate_svg(parsed, svg_text, record_id, add_failure)

    duplicate_canonical_records = _duplicate_records(canonical_hashes)
    duplicate_svg_records = _duplicate_records(svg_hashes)
    for duplicate in duplicate_canonical_records:
        for duplicate_id in duplicate["ids"][1:]:
            add_failure(duplicate_id, "duplicate_canonical_dsl", f"duplicate canonical DSL hash {duplicate['hash']}")
    for duplicate in duplicate_svg_records:
        for duplicate_id in duplicate["ids"][1:]:
            add_failure(duplicate_id, "duplicate_svg", f"duplicate SVG hash {duplicate['hash']}")

    coverage_summary = {bucket: len(values) for bucket, values in sorted(coverage_values.items())}
    thresholds = dict(_DIVERSITY_THRESHOLDS)
    if coverage_thresholds:
        thresholds.update(coverage_thresholds)
    coverage_failures = [
        {
            "bucket": bucket,
            "distinct_count": coverage_summary.get(bucket, 0),
            "threshold": threshold,
        }
        for bucket, threshold in sorted(thresholds.items())
        if coverage_summary.get(bucket, 0) < threshold
    ]

    return {
        "records_total": records_total,
        "records_failed": len(failed_ids),
        "failures": failures,
        "coverage_summary": coverage_summary,
        "coverage_failures": coverage_failures,
        "duplicate_canonical_dsl": sum(len(item["ids"]) - 1 for item in duplicate_canonical_records),
        "duplicate_canonical_dsl_records": duplicate_canonical_records,
        "duplicate_svg": sum(len(item["ids"]) - 1 for item in duplicate_svg_records),
        "duplicate_svg_records": duplicate_svg_records,
        "feature_signature_distribution": dict(sorted(feature_signature_distribution.items())),
        "useless_diagrams": useless_diagrams,
        "skipped_symbolic": skipped_symbolic,
    }


def validation_has_failures(result: Mapping[str, Any]) -> bool:
    """Return whether a validation result contains any strict-failing category."""

    return bool(
        result.get("records_failed")
        or result.get("coverage_failures")
        or result.get("duplicate_canonical_dsl")
        or result.get("duplicate_svg")
        or result.get("useless_diagrams")
    )


def format_validation_summary(result: Mapping[str, Any]) -> str:
    """Format a compact human-readable validation summary."""

    lines = [
        f"records_total: {result.get('records_total', 0)}",
        f"records_failed: {result.get('records_failed', 0)}",
        f"duplicate_canonical_dsl: {result.get('duplicate_canonical_dsl', 0)}",
        f"duplicate_svg: {result.get('duplicate_svg', 0)}",
        f"useless_diagrams: {len(result.get('useless_diagrams') or ())}",
        f"coverage_summary: {json.dumps(result.get('coverage_summary', {}), sort_keys=True)}",
    ]
    coverage_failures = result.get("coverage_failures") or []
    if coverage_failures:
        lines.append(f"coverage_failures: {json.dumps(coverage_failures, sort_keys=True)}")
    failures = result.get("failures") or []
    if failures:
        lines.append("failures:")
        for failure in failures[:10]:
            lines.append(f"  {failure['id']}: {failure['reason']} - {failure['detail']}")
        if len(failures) > 10:
            lines.append(f"  ... {len(failures) - 10} more")
    return "\n".join(lines)


def _record_coverage_values(features: Mapping[str, Any], coverage_values: dict[str, set[str]]) -> None:
    tracker = CoverageTracker()
    for bucket, value in tracker.features_to_pairs(dict(features)):
        coverage_values[bucket].add(value)


def _validate_dsl_file(
    dataset_path: Path,
    record: Mapping[str, Any],
    canonical_dsl: str,
    record_id: str,
    add_failure,
) -> None:
    dsl_path = record.get("dsl_path")
    if not isinstance(dsl_path, str) or not dsl_path:
        add_failure(record_id, "dsl_file_missing", "record is missing dsl_path")
        return
    full_path = dataset_path / dsl_path
    if not full_path.is_file():
        add_failure(record_id, "dsl_file_missing", f"DSL file does not exist: {dsl_path}")
        return
    try:
        file_dsl = full_path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as exc:
        add_failure(record_id, "dsl_file_read", str(exc))
        return
    if file_dsl != canonical_dsl:
        add_failure(record_id, "dsl_file_mismatch", "dsl_path contents differ from target.canonical_dsl")


def _validate_samples(document: ScenarioDocument, record_id: str, add_failure) -> None:
    for signal in document.signals:
        if signal.kind == SignalKind.BIT:
            invalid = [sample for sample in signal.samples if sample.lower() not in {"0", "1", "x", "z"}]
            if invalid:
                add_failure(record_id, "bit_samples", f"lane {signal.name} has illegal bit sample {invalid[0]!r}")
        if document.ticks is not None and signal.samples and len(signal.samples) != document.ticks:
            add_failure(
                record_id,
                "sample_length",
                f"lane {signal.name} has {len(signal.samples)} samples but diagram declares {document.ticks} ticks",
            )


def _has_full_samples(document: ScenarioDocument) -> bool:
    if document.ticks is None or not document.signals:
        return False
    return all(signal.samples and len(signal.samples) == document.ticks for signal in document.signals)


def _has_any_samples(document: ScenarioDocument) -> bool:
    if document.ticks is None:
        return False
    return any(signal.samples and len(signal.samples) == document.ticks for signal in document.signals)


def _anchor_occurrences(
    document: ScenarioDocument,
    record_id: str,
    add_failure,
    useless_diagrams: list[dict[str, Any]],
) -> dict[str, tuple[int, ...]]:
    samples = _sample_map(document)
    occurrences: dict[str, tuple[int, ...]] = {}
    assert document.ticks is not None
    for anchor in document.anchors:
        ticks: list[int] = []
        error: Exception | None = None
        for tick in range(document.ticks):
            try:
                if evaluate_condition(anchor.condition, samples, tick):
                    ticks.append(tick)
            except Exception as exc:
                error = exc
                break
        occurrences[anchor.name] = tuple(ticks)
        if error is not None:
            add_failure(record_id, "anchor_recoverability", f"{anchor.name}: {error.__class__.__name__}: {error}")
            useless_diagrams.append({"id": record_id, "anchor": anchor.name, "detail": str(error)})
        elif not ticks:
            detail = f"anchor {anchor.name} predicate is never satisfied in samples"
            add_failure(record_id, "anchor_recoverability", detail)
            useless_diagrams.append({"id": record_id, "anchor": anchor.name, "detail": detail})
    return occurrences


def _validate_windows(
    document: ScenarioDocument,
    occurrences: Mapping[str, Sequence[int]],
    record_id: str,
    add_failure,
    *,
    strict: bool,
) -> None:
    for window in document.windows:
        if strict:
            start = _first_occurrence(occurrences, window.start_anchor)
            end = _first_occurrence(occurrences, window.end_anchor)
            if start is None or end is None or start >= end or not _bound_allows(window.bound, end - start):
                add_failure(
                    record_id,
                    "window_recoverability",
                    f"window {window.name} first occurrences do not satisfy bound {window.bound.label}",
                )
            continue
        if not _window_recoverable(window.bound, occurrences.get(window.start_anchor, ()), occurrences.get(window.end_anchor, ())):
            add_failure(
                record_id,
                "window_recoverability",
                f"window {window.name} has no recoverable start/end occurrence pair within its bound",
            )


def _window_recoverable(bound: TimeBound, starts: Sequence[int], ends: Sequence[int]) -> bool:
    for start in starts:
        for end in ends:
            if start < end and _bound_allows(bound, end - start):
                return True
    return False


def _bound_allows(bound: TimeBound, delay: int) -> bool:
    min_delay = bound.min_delay
    max_delay = bound.max_delay
    if min_delay and max_delay and min_delay.isdigit() and max_delay.isdigit():
        return int(min_delay) <= delay <= int(max_delay)
    return True


def _validate_constraints(
    document: ScenarioDocument,
    occurrences: Mapping[str, Sequence[int]],
    record_id: str,
    add_failure,
    *,
    strict: bool,
) -> None:
    samples = _sample_map(document)
    assert document.ticks is not None
    for constraint in document.lane_constraints:
        if strict:
            if not _constraint_holds_strict(constraint, document, occurrences, samples, document.ticks):
                add_failure(
                    record_id,
                    "constraint_recoverability",
                    f"constraint {constraint.name} does not hold over its first-occurrence {constraint.region.value} region",
                )
            continue
        if not _constraint_recoverable(constraint, document, occurrences, samples, document.ticks):
            add_failure(
                record_id,
                "constraint_recoverability",
                f"constraint {constraint.name} does not hold over any recoverable {constraint.region.value} region",
            )


def _constraint_recoverable(
    constraint: LaneConstraint,
    document: ScenarioDocument,
    occurrences: Mapping[str, Sequence[int]],
    samples: Mapping[str, Sequence[str]],
    ticks: int,
) -> bool:
    for start, end in _constraint_candidate_ranges(constraint, document, occurrences, ticks):
        if all(
            signal_name in samples
            and _constraint_holds(constraint.relation, constraint.value, tuple(samples[signal_name]), start, end)
            for signal_name in constraint.signals
        ):
            return True
    return False


def _constraint_holds_strict(
    constraint: LaneConstraint,
    document: ScenarioDocument,
    occurrences: Mapping[str, Sequence[int]],
    samples: Mapping[str, Sequence[str]],
    ticks: int,
) -> bool:
    range_pair = _constraint_first_occurrence_range(constraint, document, occurrences, ticks)
    if range_pair is None:
        return False
    start, end = range_pair
    return all(
        signal_name in samples
        and _constraint_holds(constraint.relation, constraint.value, tuple(samples[signal_name]), start, end)
        for signal_name in constraint.signals
    )


def _constraint_first_occurrence_range(
    constraint: LaneConstraint,
    document: ScenarioDocument,
    occurrences: Mapping[str, Sequence[int]],
    ticks: int,
) -> tuple[int, int] | None:
    if constraint.region == ConstraintRegion.AT:
        tick = _first_occurrence(occurrences, constraint.anchor or "")
        return (tick, tick) if tick is not None else None
    if constraint.region == ConstraintRegion.BEFORE:
        tick = _first_occurrence(occurrences, constraint.anchor or "")
        return (0, tick - 1) if tick is not None else None
    if constraint.region == ConstraintRegion.AFTER:
        tick = _first_occurrence(occurrences, constraint.anchor or "")
        return (tick, ticks - 1) if tick is not None else None
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        start = _first_occurrence(occurrences, constraint.start_anchor or "")
        end = _first_occurrence(occurrences, constraint.end_anchor or "")
        if start is None or end is None or start > end:
            return None
        return start, end
    if constraint.region == ConstraintRegion.IN:
        window = document.window_map.get(constraint.window or "")
        if window is None:
            return None
        start = _first_occurrence(occurrences, window.start_anchor)
        end = _first_occurrence(occurrences, window.end_anchor)
        if start is None or end is None or start > end:
            return None
        return start, end
    return None


def _first_occurrence(occurrences: Mapping[str, Sequence[int]], anchor_name: str) -> int | None:
    values = occurrences.get(anchor_name, ())
    return values[0] if values else None


def _constraint_candidate_ranges(
    constraint: LaneConstraint,
    document: ScenarioDocument,
    occurrences: Mapping[str, Sequence[int]],
    ticks: int,
) -> list[tuple[int, int]]:
    if constraint.region == ConstraintRegion.AT:
        return [(tick, tick) for tick in occurrences.get(constraint.anchor or "", ())]
    if constraint.region == ConstraintRegion.BEFORE:
        return [(0, tick - 1) for tick in occurrences.get(constraint.anchor or "", ())]
    if constraint.region == ConstraintRegion.AFTER:
        return [(tick, ticks - 1) for tick in occurrences.get(constraint.anchor or "", ())]
    if constraint.region == ConstraintRegion.FROM_UNTIL:
        return [
            (start, end)
            for start in occurrences.get(constraint.start_anchor or "", ())
            for end in occurrences.get(constraint.end_anchor or "", ())
            if start <= end
        ]
    if constraint.region == ConstraintRegion.IN:
        window = document.window_map.get(constraint.window or "")
        if window is None:
            return []
        return [
            (start, end)
            for start in occurrences.get(window.start_anchor, ())
            for end in occurrences.get(window.end_anchor, ())
            if start <= end
        ]
    return []


def _sample_map(document: ScenarioDocument) -> dict[str, tuple[str, ...]]:
    return {signal.name: tuple(signal.samples) for signal in document.signals if signal.samples}


def _read_svg(
    dataset_path: Path,
    record: Mapping[str, Any],
    record_id: str,
    add_failure,
) -> str | None:
    svg_path = record.get("svg_path")
    if not isinstance(svg_path, str) or not svg_path:
        add_failure(record_id, "visual_recoverability", "record is missing svg_path")
        return None
    full_path = dataset_path / svg_path
    if not full_path.is_file():
        add_failure(record_id, "visual_recoverability", f"SVG file does not exist: {svg_path}")
        return None
    try:
        return full_path.read_text(encoding="utf-8")
    except OSError as exc:
        add_failure(record_id, "visual_recoverability", str(exc))
        return None


def _validate_svg(document: ScenarioDocument, svg_text: str, record_id: str, add_failure) -> None:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        add_failure(record_id, "visual_recoverability", f"SVG parse failed: {exc}")
        return

    if not _has_timing_group(root):
        add_failure(record_id, "visual_recoverability", "SVG has no <g> element with a timing-* class")
    if (document.windows or document.properties) and not _has_class(root, "timing-rule-summary"):
        add_failure(record_id, "visual_recoverability", "SVG is missing timing-rule-summary for rendered rules")
    for window in document.windows:
        has_window_label = f"p_{window.name}" in svg_text
        has_overlay_class = any(class_name in svg_text for class_name in _RESPONSE_OVERLAY_CLASSES)
        if not (has_window_label or has_overlay_class):
            add_failure(
                record_id,
                "visual_recoverability",
                f"window {window.name} has no response or hold overlay class in SVG",
            )


def _has_timing_group(root: ET.Element) -> bool:
    return any(
        _local_name(element.tag) == "g"
        and any(token.startswith("timing-") for token in element.attrib.get("class", "").split())
        for element in root.iter()
    )


def _has_class(root: ET.Element, class_name: str) -> bool:
    return any(class_name in element.attrib.get("class", "").split() for element in root.iter())


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _duplicate_records(records_by_hash: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    return [
        {"hash": digest, "ids": list(ids)}
        for digest, ids in sorted(records_by_hash.items())
        if len(ids) > 1
    ]


def _feature_signature_hash(features: Mapping[str, Any]) -> str:
    payload = [
        features.get("topology"),
        sorted(features.get("idioms") or ()),
        sorted(features.get("predicates") or ()),
    ]
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
