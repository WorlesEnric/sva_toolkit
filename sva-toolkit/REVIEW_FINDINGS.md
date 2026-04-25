# Timing Dataset Generator — Independent Review

## Summary

The generator has a useful scaffold, but it is not ready to scale as an Image-DSL dataset source. The largest risks are that requested size bounds are ignored, generated timing windows are largely invisible in SVG output, and coverage-guided sampling is only recorded after acceptance rather than used to steer generation. I found 3 critical issues, 8 important issues, 7 nice-to-have improvements, 14 spec items that are missing or only partial, 10 concrete test gaps, and 5 scale concerns.

## Critical issues (must fix before scaling)

1. Size limits from both the API and CLI are ignored or violated. `generate_dataset()` accepts `min_ticks`, `max_ticks`, `min_lanes`, and `max_lanes` at `src/sva_toolkit/timing/generate/dataset.py:92`, but `_sample_spec()` only uses tick bounds at `src/sva_toolkit/timing/generate/dataset.py:230` and the lane bounds are never checked; the CLI dutifully forwards the lane flags at `src/sva_toolkit/cli/main.py:408`, but they have no effect. Worse, `assign_ticks()` expands `total_ticks` beyond the sampled budget with `max(spec.tick_budget, max_tick + 2)` at `src/sva_toolkit/timing/generate/waveform.py:83`, so `--max-ticks 4` can produce 9-tick diagrams. This violates the defaults and tick-budget policy in spec sections 5 and 13. Fix by validating min/max arguments, rejecting candidates whose assigned anchors do not fit the sampled tick budget, and rejecting candidates outside the requested lane-count range.

2. Generated windows are not visually recoverable in the SVG. The generator emits `TimeWindow` objects in `_build_document()` at `src/sva_toolkit/timing/generate/dataset.py:332`, but it never creates `PropertyOverlay` metadata. The WaveDrom projection only turns windows into visible response arrows when they are referenced by classified properties at `src/sva_toolkit/timing/projection/wavedrom_view.py:440` and `src/sva_toolkit/timing/projection/wavedrom_view.py:625`; otherwise windows exist only in the target DSL. This contradicts the Image-DSL goal in spec sections 3, 8, and 11 because a model cannot infer the exact `window ... [min:max]` statements from pixels. Fix by adding visually grounded response properties for generated windows, or by teaching the renderer to draw standalone windows.

3. Coverage-guided sampling is not implemented. `CoverageTracker.score()` exists at `src/sva_toolkit/timing/generate/coverage.py:33`, and `coverage_target` is accepted at `src/sva_toolkit/timing/generate/dataset.py:100`, but the generation loop only calls `coverage.update()` after accepting an item at `src/sva_toolkit/timing/generate/dataset.py:177`; no score, target, or under-covered bucket affects sampling or acceptance. This misses spec section 10 and makes many rare buckets effectively random or unreachable. Fix by selecting or accepting candidates based on coverage score and by making `coverage_target` define an actual required per-bucket quota.

## Important issues (should fix)

1. Anchor and constraint semantic checks are incomplete. Waveform synthesis encodes some high/low requirements and checks high/low conflicts at `src/sva_toolkit/timing/generate/waveform.py:193`, but it never evaluates each anchor condition at the assigned tick or verifies each constraint over its intended region after samples are built. The renderer later finds anchor occurrences by scanning every tick at `src/sva_toolkit/timing/projection/wavedrom_view.py:428`, so an accidentally earlier predicate occurrence can move overlays without any rejection. This misses spec section 6 steps 7 and 8. Fix by adding a generator-local predicate evaluator and rejecting candidates whose assigned anchor tick or generated constraints do not hold.

2. Parameterized bound coverage is recorded incorrectly. `_build_window()` creates parameterized windows as `WindowBoundKind.RANGE` with a symbolic max token at `src/sva_toolkit/timing/generate/idioms.py:195`, but `_extract_features()` records `window.bound.kind.value` at `src/sva_toolkit/timing/generate/dataset.py:376`. A `[1:MAX_LAT]` window is therefore counted as `"range"` instead of `"parameterized"`, making the spec section 10 `bound_kind=parameterized` bucket invisible. Fix feature extraction to classify non-numeric bound tokens as `parameterized`.

3. The recorded `response` idiom has almost no implementation. `_sample_idioms()` may append `"response"` at `src/sva_toolkit/timing/generate/dataset.py:276`, but `_add_idiom_constraints()` only handles `hold_until`, `stable_while`, and `not_before` at `src/sva_toolkit/timing/generate/idioms.py:245`; no response property or overlay is generated. This partially misses spec section 3 Response. Fix by emitting response overlays/properties tied to the graph windows.

4. Several idiom coverage buckets are unreachable or misleading. `_DEFAULT_IDIOMS` contains only `hold_until`, `stable_while`, and `not_before` at `src/sva_toolkit/timing/generate/dataset.py:64`; `_sample_idioms()` never emits `burst`, `backpressure`, or `cut` at `src/sva_toolkit/timing/generate/dataset.py:274`, even when the topology or cuts use those concepts. This misses the spec section 10 `idiom` bucket. Fix by attaching topology-specific idioms and adding `cut` when cuts are actually generated.

5. Constraint-region coverage is very narrow. `_add_idiom_constraints()` generates only `FROM_UNTIL` and `BEFORE` constraints at `src/sva_toolkit/timing/generate/idioms.py:245`; `AT`, `IN`, and `AFTER` are supported by the parser/validator but never sampled. This misses spec section 4 and makes the region buckets in spec section 10 unreachable. Fix by adding small, type-aware `at`, `in`, and `after` constraints and extending waveform synthesis for those regions where needed.

6. Predicate coverage is very narrow. The topology library mostly hard-codes `rise`, `high`, and `all_high` predicates at `src/sva_toolkit/timing/generate/topology.py:44`, with no sampling for `fall`, `low`, `change`, `stable`, `eq`, or `neq` despite the type-aware predicate matrix in spec section 2 and the predicate coverage bucket in spec section 10. Fix by sampling compatible predicate kinds during semantic decoration and applying the corresponding waveform obligations.

7. Cut generation and cut coverage are incomplete. `_add_cuts()` only emits before/after omitted cuts with fixed labels at `src/sva_toolkit/timing/generate/idioms.py:312`, never between-window, compressed, lookback, or unlabeled cuts from spec section 3. `_extract_features()` collapses all cuts into one prioritized scalar at `src/sva_toolkit/timing/generate/dataset.py:379`, so a diagram with both before and after cuts is only recorded as `"before"`. Fix by varying cut placement/meaning/labels and recording all placements for coverage.

8. Rendering-mode ratios are only probabilistic and not coverage-aware. `_normalize_rendering_weights()` normalizes user ratios at `src/sva_toolkit/timing/generate/dataset.py:211` and `_sample_spec()` calls `rng.choices()` at `src/sva_toolkit/timing/generate/dataset.py:237`, but there is no quota or correction when small datasets miss symbolic or mixed examples. This partially misses spec sections 7 and 10. Fix with per-run quotas or coverage-weighted rendering-mode selection.

## Nice-to-have improvements

1. Bus distractor activity is missing. `synthesize_waveforms()` initializes buses to `x` and only fills stable ranges at `src/sva_toolkit/timing/generate/waveform.py:210`; it does not add bus value changes before or after stable regions as recommended in spec section 6. Fix by adding deterministic non-conflicting bus distractors outside constrained ranges.

2. Bit distractors are too weak. `_add_bit_distractors()` returns immediately for any constrained signal at `src/sva_toolkit/timing/generate/waveform.py:229`, so signals that participate in anchors are visually idle outside the interesting region. Fix by allowing toggles outside constrained ticks.

3. The selected `split` is only a label. `split` is written into records at `src/sva_toolkit/timing/generate/dataset.py:451`, but no random/topology/flavor/bound/size/rendering split policy from spec section 12 is implemented. Fix by adding split planners or by documenting that holdout flags are the only current split mechanism.

4. Duplicate detection only uses canonical DSL hashes. `seen_canonical_hashes` is checked at `src/sva_toolkit/timing/generate/dataset.py:165`, but no SVG hash, feature signature, or image/perceptual hash from spec section 11 is used. Fix by adding stable SVG normalization and a feature-signature key.

5. Visual filters are minimal. `_passes_visual_filter()` only checks SVG text length at `src/sva_toolkit/timing/generate/dataset.py:360`, so empty-looking SVGs, pathological dimensions, or severe label overlap are not rejected as requested in spec section 11. Fix by parsing dimensions and checking for expected overlay/lane elements.

6. Naming-style coverage has no short-name source. `NameFlavor.naming_style` can be `"short"` per `GenerationSpec` at `src/sva_toolkit/timing/generate/model.py:82`, but `FLAVORS` defines only snake_case, uppercase, and protocol_like styles at `src/sva_toolkit/timing/generate/names.py:25`. Fix by adding a compact flavor or a naming transform that produces short identifiers.

7. Records are written only after the whole run. `accepted_records` is accumulated in memory and flushed at `src/sva_toolkit/timing/generate/dataset.py:189`. This is convenient for tests but fragile for long jobs. Fix by streaming JSONL records as items are accepted and writing a summary at the end.

## Spec items not yet implemented

1. Spec section 3 Response: response windows exist, but response properties/overlays are missing or partial.

2. Spec section 3 Hold Until: generated only for start anchors whose predicate kind is `rise`, at `src/sva_toolkit/timing/generate/idioms.py:245`; it does not vary held relation, multiple lanes, bus-only holds, or parameterized holds.

3. Spec section 3 Stable While: generated only as `from/until` over the first edge at `src/sva_toolkit/timing/generate/idioms.py:261`; no named-window `in` form or bundle of bus lanes.

4. Spec section 3 Not Before: generated only as `low(first_response_signal) before first_trigger` at `src/sva_toolkit/timing/generate/idioms.py:277`; no forbidden event/window-end variants.

5. Spec section 3 Backpressure: topology exists at `src/sva_toolkit/timing/generate/topology.py:141`, but the decorator does not guarantee ready-low stall cycles or stable metadata during stall.

6. Spec section 3 Burst: topology exists at `src/sva_toolkit/timing/generate/topology.py:129`, but there are no middle beat anchors, payload-per-beat changes, explicit last-count variation, or response-after-last behavior.

7. Spec section 3 Setup/Hold: topology exists at `src/sva_toolkit/timing/generate/topology.py:153`, but clock-edge flavor and pre-only/post-only variants are not sampled.

8. Spec section 3 Cut/Omitted Region: only fixed before/after omitted cuts are generated; no between-window, compressed, lookback, no-label, or label-length variation.

9. Spec section 4 Constraint Generator: no compatibility matrix object exists, no `AT`/`IN`/`AFTER` sampling, no `eq`/`neq` constraint sampling, and only high/low conflict detection is implemented.

10. Spec section 6 Waveform Synthesis: bus stable conflicts, anchor truth checks, constraint truth checks, and rich distractors are incomplete.

11. Spec section 8 Property Overlay Generation: no generated `PropertyOverlay` objects are produced for primary examples.

12. Spec section 10 Coverage-Guided Sampling: buckets are tallied post-hoc, not used to bias sampling; several bucket values are unreachable.

13. Spec section 11 Rejection Filters: many filters are missing, including semantic waveform checks, rendered dimensions/emptiness beyond length, label overlap, SVG hash duplicate detection, feature duplicate detection, and triviality filters.

14. Spec section 12 Train/Validation/Test Splits: only topology and flavor holdouts exist; random, bound, size, and rendering holdouts are not implemented.

## Test gaps

1. Add tests that `--min-ticks/--max-ticks` and `--min-lanes/--max-lanes` are enforced; current tests never exercise tight bounds.

2. Add tests that generated parameterized bounds appear as `parameterized` in feature coverage, not only as `range`.

3. Add tests that each generated anchor predicate holds at the generator-assigned tick, and that the first rendered occurrence pair for each window is not outside the declared bound.

4. Add tests for `AT`, `IN`, `AFTER`, `eq`, `neq`, `fall`, `change`, and bus-stable constraints once those buckets are implemented.

5. Add tests that response/window overlays are present in SVG for generated windows, e.g. by checking for response arrow overlay elements.

6. Add tests that cut coverage records before and after placements when both are generated.

7. Add tests that `coverage_target` changes behavior or fails clearly when impossible.

8. Add tests for rendering-mode quota behavior with deterministic seeds and small counts.

9. Add tests that mixed examples keep at least one sampled lane and at least one symbolic lane.

10. Add CLI tests for invalid flag combinations such as `--min-ticks > --max-ticks`, `--min-lanes > --max-lanes`, and all rendering ratios set to zero.

## Performance / scale concerns

1. Tight size bounds can create retry-loop blowups once the bounds are enforced, because delay sampling is independent of the tick budget at `src/sva_toolkit/timing/generate/idioms.py:191`. Consider sampling delays from the remaining budget or rejecting early before rendering.

2. `accepted_records` stores full canonical DSL strings for the whole run at `src/sva_toolkit/timing/generate/dataset.py:133`; for 10,000 examples this is manageable but unnecessary and makes interrupted runs lose all record output. Stream records as they are accepted.

3. SVG rendering occurs before coverage acceptance could be applied at `src/sva_toolkit/timing/generate/dataset.py:157`; once coverage-guided rejection exists, score cheaper structural candidates before rendering when possible.

4. Rejection counts are too coarse because all `GenerationError` instances are grouped by class at `src/sva_toolkit/timing/generate/dataset.py:158`, hiding the actual failure mode during large runs. Include a stable reason code in `GenerationError` or rejection accounting.

5. Duplicate checks are O(1) for canonical DSL hashes, but the lack of SVG/feature duplicate keys means a large run can accumulate many visually redundant examples even when text differs.

## Phase B — Applied Fixes

- `src/sva_toolkit/timing/generate/dataset.py:113` — added validation for tick and lane bound arguments — invalid CLI/API bounds now fail before generation.
- `src/sva_toolkit/timing/generate/dataset.py:180` — reject accepted candidates whose lane count falls outside `min_lanes`/`max_lanes` — lane-count flags now have real effect.
- `src/sva_toolkit/timing/generate/waveform.py:83` — stop expanding `total_ticks` beyond the sampled budget and reject over-budget anchor assignments — `max_ticks` is now enforced by retries.
- `src/sva_toolkit/timing/generate/dataset.py:254` — sample `cuts_enabled` before idiom selection — cut idiom metadata can reflect actual cut generation.
- `src/sva_toolkit/timing/generate/dataset.py:289` — record topology-specific `burst` and `backpressure` idioms and append `cut` when cuts are enabled — idiom coverage is less misleading.
- `src/sva_toolkit/timing/generate/dataset.py:337` — render response overlays from temporary render-only properties — generated windows now have visible SVG response summaries without polluting the canonical DSL target.
- `src/sva_toolkit/timing/generate/dataset.py:417` — classify feature bound kinds with `_bound_feature()` — parameterized bounds are now counted as `parameterized` instead of `range`.
- `src/sva_toolkit/timing/generate/dataset.py:420` and `src/sva_toolkit/timing/generate/coverage.py:82` — record all cut placements and count each placement in coverage — before/after cuts are both visible in coverage.
- `src/sva_toolkit/timing/generate/waveform.py:216` — detect overlapping bus stable ranges with different values — contradictory bus constraints are rejected.
- `src/sva_toolkit/timing/generate/waveform.py:227` — verify synthesized samples against assigned anchor predicates and lane constraints — invalid generated waveforms are rejected before emission.
- `tests/timing/test_generate.py:50` — added size-bound regression tests — tick and lane bounds cannot silently regress.
- `tests/timing/test_generate.py:93` — added SVG response-overlay regression coverage — generated windows must produce visible response summaries.
- `tests/timing/test_generate.py:103` — added parameterized-bound feature regression coverage — symbolic bounds keep their intended bucket.
- `tests/timing/test_generate.py:113` — added semantic verifier regression tests — anchor and constraint mismatches are caught directly.
