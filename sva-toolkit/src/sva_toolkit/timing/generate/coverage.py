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

TICK_COUNT_BUCKETS: tuple[tuple[int, int], ...] = ((0, 6), (7, 12), (13, 20), (21, 9999))
LANE_COUNT_BUCKETS: tuple[tuple[int, int], ...] = ((0, 3), (4, 6), (7, 12), (13, 9999))


class CoverageTracker:
    """Tally feature usage across accepted items and score new candidates."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_accepted = 0

    def score(self, features: dict[str, Any]) -> float:
        score = 1.0
        for bucket, value in self.features_to_pairs(features):
            current = self.counts[bucket].get(value, 0)
            score += 1.0 / (1.0 + current)
        return score

    def update(self, features: dict[str, Any]) -> None:
        for bucket, value in self.features_to_pairs(features):
            self.counts[bucket][value] += 1
        self.total_accepted += 1

    def buckets_seen(self) -> dict[str, set[str]]:
        return {bucket: set(values.keys()) for bucket, values in self.counts.items()}

    def deficient_buckets(self, target_distinct: int) -> set[str]:
        if target_distinct <= 0:
            return set()
        return {
            bucket
            for bucket in _BUCKETS
            if self.is_bucket_deficient(bucket, target_distinct)
        }

    def is_bucket_deficient(self, bucket: str, target_distinct: int) -> bool:
        if target_distinct <= 0:
            return False
        return len(self.counts.get(bucket, {})) < target_distinct

    def feature_increases_deficient_bucket(
        self,
        features: dict[str, Any],
        target_distinct: int,
    ) -> bool:
        deficient = self.deficient_buckets(target_distinct)
        if not deficient:
            return True
        for bucket, value in self.features_to_pairs(features):
            if bucket in deficient and value not in self.counts.get(bucket, {}):
                return True
        return False

    def missing_values(self, bucket: str, values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
        seen = self.counts.get(bucket, {})
        return tuple(value for value in values if value not in seen)

    def is_target_satisfied(self, target_per_bucket: int) -> bool:
        for bucket in _BUCKETS:
            for value, count in self.counts.get(bucket, {}).items():
                if count < target_per_bucket:
                    return False
        return True

    def features_to_pairs(self, features: dict[str, Any]):
        topology = features.get("topology")
        if topology:
            yield "topology", str(topology)
        for idiom in features.get("idioms", ()) or ():
            yield "idiom", str(idiom)
        ticks = features.get("ticks")
        if isinstance(ticks, int):
            yield "tick_count", bucket_range(ticks, TICK_COUNT_BUCKETS)
        lane_count = features.get("lane_count")
        if isinstance(lane_count, int):
            yield "lane_count", bucket_range(lane_count, LANE_COUNT_BUCKETS)
        lane_kind = features.get("lane_kind")
        if lane_kind:
            yield "lane_kind", str(lane_kind)
        anchor_count = features.get("anchor_count")
        if isinstance(anchor_count, int):
            yield "anchor_count", bucket_range(anchor_count, ((0, 2), (3, 5), (6, 9999)))
        window_count = features.get("window_count")
        if isinstance(window_count, int):
            yield "window_count", bucket_range(window_count, ((0, 0), (1, 1), (2, 4), (5, 9999)))
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


def bucket_range(value: int, ranges: tuple[tuple[int, int], ...]) -> str:
    for low, high in ranges:
        if low <= value <= high:
            if low == high:
                return str(low)
            return f"{low}-{high if high < 9999 else 'plus'}"
    return str(value)
