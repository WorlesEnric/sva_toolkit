# Timing Diagram Dataset Generation Design

## Purpose

This document describes the procedural generator that produces large
Image-DSL datasets for the timing-diagram DSL. The target task is to train
a Vision-Language Model that recovers canonical timing DSL code from a
rendered timing-diagram image.

The pipeline implements the redesign described in
[`docs/refactor_render.md`](refactor_render.md): a stochastic observation
model rather than a single fixed renderer, paired with a strict
visual-recoverability contract on the supervised target.

```text
canonical visual DSL
  -> semantic timing scene (TimingScene)
       -> sampled renderer/style/page/degradation (RenderSpec)
            -> image (SVG/PNG)
```

The generator does not sample raw DSL text directly. The DSL has semantic
cross-references between lanes, anchors, windows, cuts, constraints,
properties, and rendered overlays. A direct grammar sampler would produce
many invalid, trivial, or visually ambiguous examples. The pipeline is
therefore scenario-first, then visually lowered, then rendered through a
randomly sampled renderer/style:

```text
primitive temporal idioms
  -> temporal event graph
  -> typed signal schema
  -> concrete or symbolic constraints
  -> waveform synthesis
  -> ScenarioDocument (semantic)
  -> lower_to_visual_document(...)         (Phase 1)
  -> visual ScenarioDocument               (the target)
  -> build_timing_scene(...)               (Phase 2)
  -> TimingScene IR
  -> sample_render_spec(...)               (Phase 7)
  -> RenderSpec
  -> render(scene, spec)                   (Phase 4)
  -> RenderResult + audits
  -> compose_record(...)                   (Phase 6)
  -> degraded image bytes
  -> JSONL record + per-record metadata
```

The canonical source of truth remains
`sva_toolkit.timing.core.scenario.ScenarioDocument`. DSL text is emitted
with `sva_toolkit.timing.bridge.to_dsl.emit_timing_dsl`, then parsed and
validated again before rendering.

## Non-Goals

- The generator does not need hundreds of hand-written protocol templates.
- The generator does not need to prove all generated properties formally.
- The generator does not require one-to-one recovery of invisible
  information such as comments, formatting, or raw property text that is
  not present in the rendered figure.
- The generator does not train on arbitrary raw DSL variants unless the
  rendered image contains enough visual evidence to recover them.

## Key Principles

1. **Compositional content** — a small idiom library combined with a
   temporal-graph generator yields broad variation without large manual
   motif catalogs.
2. **Visual recoverability of the target** — every token in the supervised
   DSL must be visible in the image, geometrically inferable, or
   deterministically canonical by visual convention. Hidden semantic
   tokens (arbitrary anchor names, plain-English summaries, parameter
   names like `MAX_LAT`) are removed from the target by
   `lower_to_visual_document`.
3. **Stochastic observation model** — render through one of many sampled
   renderers/styles/page contexts/degradations rather than a single fixed
   pipeline. Visual coverage is tracked alongside semantic coverage.
4. **No target leakage on the image** — the leakage audit rejects records
   whose rendered text contains target tokens that are not also lane
   names, bus values, geometric guides, or canonical conventions.
5. **Auditable per record** — every record carries a leakage audit, a
   target-visibility audit, contrast/occlusion/layout/reproducibility
   audits, and the full sampled `RenderSpec`.

## Per-record Output Layout

```
out_dir/
  records.jsonl
  summary.json
  dsl/             td_000123.td        # canonical *visual* DSL (target)
  semantic/        td_000123.td        # canonical semantic DSL (debug)
  svg/             td_000123.svg       # rendered SVG (when applicable)
  png/             td_000123.png       # rasterized + composed + degraded
  render_specs/    td_000123.json      # full sampled RenderSpec
  audits/          td_000123.json      # all audit reports for the record
```

Each `records.jsonl` line:

```json
{
  "id": "td_000123",
  "seed": 12345,
  "split": "train",
  "dsl_path": "dsl/td_000123.td",
  "semantic_dsl_path": "semantic/td_000123.td",
  "svg_path": "svg/td_000123.svg",
  "image_path": "png/td_000123.png",
  "render_spec_path": "render_specs/td_000123.json",
  "audits_path": "audits/td_000123.json",
  "renderer_id": "native_svg",
  "profile": "native-random",
  "style_family": "native_random",
  "annotation_policy": "geometric_guides",
  "degradation_profile": "native",
  "target": {
    "canonical_dsl": "diagram ... { ... anchor a0 = rise(req); ... }",
    "policy": "visual"
  },
  "features": {
    "topology": "single_response",
    "idioms": ["response", "hold_until"],
    "ticks": 9,
    "lane_count": 5,
    "anchor_count": 2,
    "window_count": 1,
    "has_bus": true,
    "has_params": false,
    "bound_kinds": ["range"],
    "constraint_regions": ["from_until", "before"],
    "predicates": ["rise", "stable"],
    "lane_kind": "mixed",
    "rendering": "concrete",
    "naming": "snake_case",
    "cut": "before",
    "cuts": ["before"]
  },
  "visibility": {
    "anchor_names": "canonical_visual",
    "bounds": "visible_geometry",
    "bus_values": "visible_text",
    "rule_summaries": "not_rendered"
  },
  "difficulty": {
    "occlusion": 0.08,
    "contrast": 0.62,
    "crop": "loose",
    "dpi": 150
  },
  "audit_status": {
    "leakage": "pass",
    "target_visibility": "pass",
    "contrast": "pass",
    "occlusion": "pass",
    "layout_overflow": "pass",
    "reproducibility": "pass"
  }
}
```

`summary.json` carries the same fields the legacy generator had, plus:

```json
{
  "coverage": {
    "semantic": { ... },
    "visual": {
      "renderer_id":      {"native_svg": 25, "wavedrom": 5, ...},
      "profile":          {"native-random": 20, "clean-wavedrom": 5, ...},
      "annotation_policy":{"geometric_guides": 23, "none": 5, ...},
      "color_mode":       { ... },
      "raster_dpi_bucket":{ ... },
      ...
    }
  },
  "rejection_reasons": {
    "render_text_leakage": 0,
    "target_not_visible": 0,
    "low_contrast": 2,
    "external_renderer_unavailable": 5,
    "holdout_style": 0,
    "rendering_mode_quota": 19,
    ...
  },
  "profile_set": "train_v2",
  "split_plan": "train_v2",
  "audits_strict": true
}
```

## Generator Architecture

### 0. Visual Lowering (Phase 1 — required for VLM targets)

`lower_to_visual_document(document, policy)` converts a fully-decorated
semantic `ScenarioDocument` into a visual-target `ScenarioDocument`:

- **Anchors** are renamed to `a0, a1, ...` by deterministic visual order
  (sorted by absolute tick → predicate role → primary signal).
- **Windows** are renamed to `w0, w1, ...` after anchors resolve.
- **Lane constraints** are renamed to `c0, c1, ...`.
- **Property overlays** without a parsed AST (paraphrase / renderer
  hints) are dropped under `visual` policy and kept under
  `debug_keep_all`.
- **Notes** and **bundle metadata** are dropped under `visual`.
- All references through `TimeWindow.start_anchor` /
  `LaneConstraint.anchor` / `Cut.anchor` / etc. are rewritten via the
  rename map.

Bound visibility is classified per window:

| Bound shape                         | Visibility class       |
| ----------------------------------- | ---------------------- |
| `EXACT` literal integer             | `visible_geometry`     |
| `RANGE` with literal endpoints      | `visible_text`         |
| Parameter (`MAX_LAT`, `SETUP`, ...) | `hidden_semantic`      |

The leakage audit will refuse to print `hidden_semantic` bound text on
the image unless `AnnotationPolicy.DEBUG_LEAKY` is in effect.

### 1. Structure Generator (unchanged)

The structure generator creates a temporal event graph. Nodes become
anchors. Edges become timing windows.

| Topology          | Shape                       | Use                                    |
| ----------------- | --------------------------- | -------------------------------------- |
| `single_response` | `A -> B`                    | req/ack, irq/clear, valid/ready        |
| `chain`           | `A -> B -> C`               | multi-phase transactions               |
| `fork`            | `A -> B`, `A -> C`          | one request, multiple responses        |
| `join`            | `A -> C`, `B -> C`          | two prerequisites before completion    |
| `parallel`        | `A -> B`, `C -> D`          | independent activity in one diagram    |
| `burst`           | `first -> beat -> last`     | packet/data burst behavior             |
| `backpressure`    | `valid_rise -> handshake`   | stall followed by acceptance           |
| `setup_hold`      | `setup -> sample -> hold`   | stability around sample point          |

The graph must be acyclic for simple tick assignment. Cyclic protocol
behavior is unrolled into multiple beat anchors.

### 2. Semantic Decorator, Idioms, Constraints, Tick Assignment, Waveform Synthesis

Unchanged from prior versions; see Sections 2–6 below for details. The
semantic decorator chooses a domain flavor, allocates names from role
pools, picks predicates, applies idioms (`response`, `hold_until`,
`stable_while`, `not_before`, `backpressure`, `burst`, `setup_hold`,
`cut`), assigns ticks, and synthesizes waveforms.

### 3. Visual Scene Construction

`build_timing_scene(visual_document, semantic_document=...)` projects the
visual document into a renderer-independent IR:

- one `LaneScene` per `SignalDecl`, with samples compressed into runs
- a `TickModel` carrying `total_ticks` and an optional grid-pitch hint
- `VisualEvent`s for each anchor (canonical name + tick)
- `VisualConstraint`s for each lane constraint
- `CutRegion`s translated from semantic cuts
- optional `Decoration` tuples (vertical guides, measurement brackets,
  hold highlights, callouts, nuisance text, captions, hand-drawn marks)

The scene is the hand-off point between content generation and rendering.
No renderer touches `ScenarioDocument` directly.

### 4. Render-Profile Sampling

A render profile bundles together a renderer id, an `AnnotationPolicy`,
default style/page/degradation choices, and any per-renderer overrides.

The canonical profiles are:

| Profile id          | Renderer      | Annotation policy      | Notes                                                          |
| ------------------- | ------------- | ---------------------- | -------------------------------------------------------------- |
| `debug-current`     | wavedrom      | `debug_leaky`          | Legacy look (prints `req_rise`, `RULES`). **Audit fails by design.** |
| `clean-wavedrom`    | wavedrom      | `none`                 | Same backbone, overlays stripped.                              |
| `native-random`     | native_svg    | `geometric_guides`     | Stochastic native; lane labels + unlabelled vertical guides.   |
| `datasheet-native`  | native_svg    | `natural_measurements` | Datasheet style; bracket text for `visible_text` bounds.       |
| `document-native`   | native_svg    | `geometric_guides`     | Native + page composition + document degradation.              |
| `ood-native`        | native_svg    | `natural_measurements` | Exotic axes (curved transitions, hatched buses, low contrast). |
| `undulate-random`   | undulate      | `geometric_guides`     | Requires `undulate` package.                                   |
| `tikz-datasheet`    | tikz_timing   | `natural_measurements` | Requires `pdflatex`.                                           |
| `plantuml-ood`      | plantuml      | `none`                 | Requires `plantuml` (held out of training).                    |
| `gtkwave-ood`       | gtkwave       | `none`                 | Requires `gtkwave` (held out of training).                     |
| `ascii-rfc`         | ascii         | `none`                 | UTF-8 ASCII waveform + monospace PNG (no SVG).                 |

Profile sets:

| Set id              | Members and weights                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `train_v2`          | `native-random` 0.50 / `clean-wavedrom` 0.15 / `undulate-random` 0.10 / `document-native` 0.10 / `datasheet-native` 0.07 / `tikz-datasheet` 0.05 / `ascii-rfc` 0.03 |
| `val_seen_style`    | same renderers, different RNG seeds + semantic holdouts                                                                                |
| `val_unseen_style`  | `plantuml-ood` 0.50 / `gtkwave-ood` 0.50                                                                                               |
| `test_ood`          | `tikz-datasheet` 0.20 / `plantuml-ood` 0.20 / `gtkwave-ood` 0.20 / `ascii-rfc` 0.15 / `ood-native` 0.25                                 |

`sample_render_spec(rng, profile=, scene=)` then samples the continuous
visual axes — font family/size, stroke width, transition shape (sharp /
slanted / curved / step), grid mode, bus style, unknown style, cut style,
color mode (color / grayscale / monochrome / low-contrast / inverted),
DPI, blur, perspective, JPEG quality, etc. Sampling uses the supplied
rng — same seed ⇒ same `RenderSpec`.

### 5. Render + Audits

`render2.pipeline.render(scene, spec, *, target_dsl_text, enforce_audits)`:

1. Look up the renderer in `DEFAULT_REGISTRY` (`native_svg`, `wavedrom`,
   `undulate`, `tikz_timing`, `plantuml`, `gtkwave`, `ascii`).
2. Build the renderer's source dict / scene from the `TimingScene`.
3. Render. The renderer must populate a `VisualVisibilityReport` listing
   every text primitive it drew, classified by role
   (`lane_label`, `bus_value_text`, `measurement_bracket`,
   `nuisance_text`, `debug_overlay`, ...).
4. Run audits:
   - **leakage** — `rendered_tokens & target_tokens - allowed_tokens`
     must be empty. Allowed tokens = lane names ∪ bus values ∪ printed
     bound brackets ∪ canonical anchor/window names that the
     decoration layer intentionally drew.
   - **target_visibility** — every required signal name in the target
     must appear in `rendered_text`.
   - **contrast** — minimum sRGB-luminance contrast between waveform
     stroke and background ≥ 0.45.
   - **occlusion** — per-lane occluded fraction by decorations and
     nuisance text ≤ 0.15.
   - **layout_overflow** — primitive bbox union ⊆ canvas.
   - **reproducibility** — same `RenderSpec` and rng ⇒ identical SVG/PNG
     digest on a second pass.
5. Return a `RenderOutcome` with `audits_passed` and a
   `rejection_reason` populated when `enforce_audits=True`.

When `enforce_audits=False`, the audits still run and their results are
recorded per record, but no rejection occurs.

### 6. Page Composition and Degradation

`compose_record(scene, spec, result, *, rng)`:

1. Rasterize the SVG (cairosvg → resvg-py → wand → synthetic fallback).
2. Apply `PageComposer` per `RenderSpec.page`:
   - tight crop, loose-with-caption, off-center, page-fragment
   - caption above/below, surrounding paragraph fragments, table border,
     page header/footer (all tagged `nuisance_text` for the audit)
3. Apply `DegradationPipeline` per `RenderSpec.degradation`:
   1. geometry: rotation, perspective
   2. document: photocopy/scan/fax/paper texture/bleed-through
      (`augraphy` if installed; PIL fallback otherwise)
   3. color: grayscale / monochrome / low-contrast / inverted
   4. morphology: line thinning / thickening / broken strokes
   5. compression: jpeg/webp/png re-encode
   6. crop/resize: final canvas tuning
4. Save to bytes per `RasterSpec.output_format`.

Optional dependencies (`numpy`, `cv2`, `albumentations`, `augraphy`,
`resvg-py`, `wand`) are all guarded — the pipeline falls back to
Pillow-only ops when they are missing and records a per-record
composition warning.

### 7. Rejection Filters

A candidate is rejected (and the reason counted in `summary.json`) for any
of these:

Existing semantic checks (preserved):

- DSL parse fails / `validate_diagram` fails
- SVG/PNG rendering fails
- Lane sample length mismatch
- Bit lane contains values other than `0`/`1`/`x`/`z`
- Window/constraint references unknown anchors/windows
- Trivial diagram for the target split
- Duplicate of a prior canonical-DSL hash, SVG hash, or feature signature

New render2 / visual checks:

- `render_text_leakage`
- `target_not_visible`
- `low_contrast`
- `required_label_cropped`
- `required_bus_value_occluded`
- `external_renderer_failed`
- `external_renderer_unavailable`
- `layout_overflow`
- `unsupported_scene_for_renderer`
- `holdout_style` / `holdout_degradation` / `holdout_annotation`

### 8. Coverage-Guided Sampling

The semantic `CoverageTracker` (unchanged) buckets topology, idiom,
tick_count, lane_count, lane_kind, anchor_count, window_count,
bound_kind, predicate, region, cut, rendering, naming.

The new `VisualCoverageTracker` (Phase 7) buckets:

```
renderer_id, profile, style_family, font_family_bucket, font_size_bucket,
stroke_width_bucket, grid_mode, tick_label_mode, bus_style, unknown_style,
cut_style, annotation_policy, helper_line_count_bucket,
nuisance_text_count_bucket, page_context_mode, color_mode,
raster_dpi_bucket, compression_bucket, blur_bucket, perspective_bucket,
crop_bucket, occlusion_bucket, recoverability_class, leakage_audit_status
```

Both trackers update on every accepted record. `summary.json` carries
both as `coverage.semantic` and `coverage.visual`.

### 9. Splits

`sva_toolkit.timing.generate.splits` defines the canonical plans:

| Split                | Profile set         | Held-out profiles                                    |
| -------------------- | ------------------- | ---------------------------------------------------- |
| `train_v2`           | `train_v2`          | `plantuml-ood`, `gtkwave-ood`                        |
| `val_seen_style`     | `val_seen_style`    | none                                                 |
| `val_unseen_style`   | `val_unseen_style`  | `document-native`, `native-random`, `clean-wavedrom`, `datasheet-native` |
| `test_synthetic_ood` | `test_ood`          | none (full OOD)                                      |
| `test_real`          | (user-supplied)     | scaffold for manually annotated diagrams             |

The split machinery enforces holdouts by counting unselected profiles as
`holdout_style` rejections and resampling.

## CLI

### Generate one split

```bash
sva timing generate-dataset \
    --count 50000 \
    --seed 1 \
    --out /data/timing-v2/train \
    --split train \
    --render-profile-set train_v2 \
    --target-policy visual \
    --emit-render-specs \
    --audit-strict \
    --style-holdout plantuml-ood,gtkwave-ood \
    --format both
```

All flags accepted by the legacy generator (`--min-ticks`, `--max-ticks`,
`--min-lanes`, `--max-lanes`, `--concrete-ratio`, `--symbolic-ratio`,
`--mixed-ratio`, `--max-retries`, `--coverage-target`,
`--coverage-required`, `--holdout-topology`, `--holdout-flavor`,
`--holdout-bound`, `--holdout-size`, `--holdout-rendering`,
`--split-policy`, `--cuts-probability`, `--distractor-probability`)
continue to work.

New render2 flags:

- `--render-profile-set <id>` — default `train_v2`.
- `--render-profile <id>` — pin one profile for every record.
- `--target-policy {visual|debug_keep_all}` — default `visual`.
- `--emit-render-specs` / `--no-emit-render-specs` — default emit.
- `--audit-strict` / `--no-audit-strict` — default strict.
- `--style-holdout <ids>` — CSV; profiles to drop from the active set.
- `--degradation-holdout <families>` — CSV.
- `--annotation-holdout <policies>` — CSV.

### Generate all canonical splits in one command

Use the helper script (`scripts/generate_timing_dataset.py`):

```bash
python scripts/generate_timing_dataset.py \
    --root /data/timing-v2 \
    --train-count 50000 \
    --val-seen-count 2000 \
    --val-unseen-count 1000 \
    --test-ood-count 2000 \
    --seed 1 \
    --format both
```

This wraps `sva timing generate-dataset` and applies the right
`--render-profile-set` and `--style-holdout` per split. Use
`--skip-ood` on machines without PlantUML / GTKWave / pdflatex.

### Validate

```bash
sva timing validate-dataset \
    --dataset /data/timing-v2/train \
    --strict \
    --strict-visual
```

`--strict-visual` validates `render_specs/`, `audits/`, and visual
coverage axes alongside the existing semantic validation.

### Visual inspection

```bash
python scripts/render_profile_gallery.py \
    examples/td/01_simple_handshake.td \
    examples/td/06_bus_protocol.td \
    --out-dir /tmp/sva_gallery \
    --seeds 1,7,42 \
    --format svg
```

Produces one SVG per (input × profile × seed) triple plus a
`gallery.json` manifest.

## Recommended Defaults

| Setting                        | Default                                  |
| ------------------------------ | ---------------------------------------- |
| ticks                          | 6 to 20                                  |
| lanes                          | 3 to 12                                  |
| anchors                        | 2 to 8                                   |
| windows                        | 1 to 6                                   |
| bus lane ratio                 | 30% to 50%                               |
| concrete examples              | 80%                                      |
| symbolic examples              | 10%                                      |
| mixed examples                 | 10%                                      |
| cuts                           | 20% to 35%                               |
| parameterized bounds           | 25%                                      |
| unbounded bounds               | 5% to 10%                                |
| distractor lanes               | 0 to 3                                   |
| max retries per accepted item  | 100                                      |
| render-profile-set             | `train_v2`                               |
| target-policy                  | `visual`                                 |
| audit-strict                   | true                                     |

## Evaluation Recommendations

Evaluate with multiple metrics rather than only exact string match:

- **Parse validity** — model output parses as DSL.
- **Validation success** — model output passes `validate_diagram`.
- **Canonical exact match** — strict target match.
- **Semantic equivalence** — ignores harmless renaming / canonicalization.
- **Lane recovery F1** — signal names, lane order, lane types.
- **Waveform edit distance** — per-lane state/run recovery.
- **Event recovery F1** — anchor positions, predicates.
- **Bound accuracy** — exact or tolerance-based timing window recovery.
- **Bus value accuracy** — OCR-style value recovery.
- **Robustness slices** — accuracy by renderer, degradation, annotation
  policy, crop, DPI.
- **Leakage sensitivity** — accuracy drop when debug overlays are
  removed at eval time. A large drop signals the model was relying on
  leaked overlay text rather than waveform geometry.

The most important diagnostic: train with and without leaky overlays,
then evaluate on no-overlay diagrams. If performance collapses, the
model was not learning waveform semantics — fix data, not model.

## Reproducibility

- Master seed → per-record `GenerationRng` derives child RNGs by
  `sha256(seed:label)`.
- Per-record `seed` is stored in the JSONL record.
- Per-record `RenderSpec` is serialized to `render_specs/td_*.json` and
  round-trips exactly via
  `sva_toolkit.timing.render2.serialization.spec_from_dict`.
- The reproducibility audit re-renders the same `RenderSpec` and asserts
  the digest matches; failures are recorded as
  `audits.reproducibility.passed = false`. External adapters that shell
  out (PlantUML, GTKWave) are not always byte-reproducible — this is
  detected, not silently accepted.

## File Map

```
sva_toolkit/timing/
  visual/                    # Phase 1 — visual lowering + visibility
    lowering.py
    policy.py
    visibility.py
  render2/                   # Phases 2-7 — renderer-independent core + adapters
    scene.py                 # TimingScene IR
    primitives.py            # Line/Polyline/Path/Text/Rect/Group + role allowlist
    decorations.py           # Decoration + AnnotationPolicy
    spec.py                  # RenderSpec + Style/Layout/Annotation/Page/Raster/Degradation
    result.py                # RenderResult + DiagramLayout + VisualVisibilityReport
    protocol.py              # TimingRenderer Protocol + RendererRegistry + DEFAULT_REGISTRY
    scene_builder.py         # build_timing_scene(...)
    serialization.py         # spec_to_dict / spec_from_dict
    profiles.py              # PROFILE_* constants and PROFILE_SET_*
    spec_sampler.py          # sample_profile / sample_render_spec
    visual_coverage.py       # VisualCoverageTracker
    rasterize.py             # rasterize_svg(...)
    page_composer.py         # PageComposer
    compose.py               # compose_record(scene, spec, result, *, rng)
    pipeline.py              # render(scene, spec, *, target_dsl_text, enforce_audits)
    decoration_layer.py      # decoration -> primitive emitter
    native/                  # NativeSvgRenderer (scene-graph SVG)
    adapters/                # WaveDromAdapter, ASCIIAdapter, optional externals
    audit/                   # leakage, target_visibility, contrast, occlusion,
                             # layout_overflow, reproducibility
    legacy/                  # legacy debug overlays for DEBUG_LEAKY profile
    degrade/                 # geometry, document, color, morphology, compression
  generate/                  # Phase 8 — dataset orchestration
    dataset.py               # generate_dataset(...) (refactored)
    render_pipeline.py       # generate_one_record(...) using render2
    splits.py                # SplitPlan + canonical split definitions
    validate_dataset.py      # extended with --strict-visual checks
    coverage.py              # semantic CoverageTracker (unchanged)
    ...
```

## Related Docs

- [Timing diagrams](sva-timing.md) — CLI surface, programmatic API,
  legacy renderer, render2 profiles.
- [Refactor design notes](refactor_render.md) — full motivation and
  detailed design rationale for the stochastic observation model.
- [Problem brief](problem.md) — original failure-mode analysis (visual
  monoculture + overlay leakage).
- [Helper scripts](../sva-toolkit/scripts/README.md) — gallery + dataset
  orchestrator scripts.
