"""Coverage tracking for the timing dataset generator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


_BUCKETS: tuple[str, ...] = (
    "topology",
    "idiom",
    "tick_count",
    "lane_count",
    "lane_kind",
    "anchor_count",
    "window_count",
    "bound_kind",
    "predicate",
    "region",
    "cut",
    "rendering",
    "naming",
)


class CoverageTracker:
    """Tally feature usage across accepted items and score new candidates."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_accepted = 0

    def score(self, features: dict[str, Any]) -> float:
        score = 1.0
        for bucket, value in self._features_to_pairs(features):
            current = self.counts[bucket].get(value, 0)
            score += 1.0 / (1.0 + current)
        return score

    def update(self, features: dict[str, Any]) -> None:
        for bucket, value in self._features_to_pairs(features):
            self.counts[bucket][value] += 1
        self.total_accepted += 1

    def buckets_seen(self) -> dict[str, set[str]]:
        return {bucket: set(values.keys()) for bucket, values in self.counts.items()}

    def is_target_satisfied(self, target_per_bucket: int) -> bool:
        for bucket in _BUCKETS:
            for value, count in self.counts.get(bucket, {}).items():
                if count < target_per_bucket:
                    return False
        return True

    def _features_to_pairs(self, features: dict[str, Any]):
        topology = features.get("topology")
        if topology:
            yield "topology", str(topology)
        for idiom in features.get("idioms", ()) or ():
            yield "idiom", str(idiom)
        ticks = features.get("ticks")
        if isinstance(ticks, int):
            yield "tick_count", _bucket_range(ticks, ((0, 6), (7, 12), (13, 20), (21, 9999)))
        lane_count = features.get("lane_count")
        if isinstance(lane_count, int):
            yield "lane_count", _bucket_range(lane_count, ((0, 3), (4, 6), (7, 12), (13, 9999)))
        lane_kind = features.get("lane_kind")
        if lane_kind:
            yield "lane_kind", str(lane_kind)
        anchor_count = features.get("anchor_count")
        if isinstance(anchor_count, int):
            yield "anchor_count", _bucket_range(anchor_count, ((0, 2), (3, 5), (6, 9999)))
        window_count = features.get("window_count")
        if isinstance(window_count, int):
            yield "window_count", _bucket_range(window_count, ((0, 0), (1, 1), (2, 4), (5, 9999)))
        for bound in features.get("bound_kinds", ()) or ():
            yield "bound_kind", str(bound)
        for predicate in features.get("predicates", ()) or ():
            yield "predicate", str(predicate)
        for region in features.get("constraint_regions", ()) or ():
            yield "region", str(region)
        cuts = features.get("cuts")
        if cuts:
            for cut in cuts:
                yield "cut", str(cut)
        else:
            cut = features.get("cut")
            if cut:
                yield "cut", str(cut)
        rendering = features.get("rendering")
        if rendering:
            yield "rendering", str(rendering)
        naming = features.get("naming")
        if naming:
            yield "naming", str(naming)


def _bucket_range(value: int, ranges: tuple[tuple[int, int], ...]) -> str:
    for low, high in ranges:
        if low <= value <= high:
            if low == high:
                return str(low)
            return f"{low}-{high if high < 9999 else 'plus'}"
    return str(value)
