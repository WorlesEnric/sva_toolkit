# Timing Diagrams

## Purpose

`sva_toolkit.timing` owns the timing-diagram DSL, rendering pipeline, and bridges to and from SVA. In V3 it remains the canonical way to move between diagram-oriented protocol documentation and assertion-oriented verification artifacts.

The render layer is split in two:

- **Legacy renderer** — `sva_toolkit.timing.render` (WaveDrom-backed, with anchor pills, response arrows, and the `RULES` footer). Kept intact for human inspection and backwards compatibility.
- **render2** — `sva_toolkit.timing.render2`, the stochastic, audit-aware pipeline introduced for VLM dataset generation. It separates a renderer-independent `TimingScene` IR from pluggable renderer adapters, lowers the supervised target through `lower_to_visual_document`, and runs leakage / target-visibility / contrast / occlusion / overflow / reproducibility audits on every record.

Default `sva timing render` behavior is unchanged: omitting `--render-profile` falls back to the legacy `debug-current` look.

## CLI Commands

### Validate

```bash
sva timing validate examples/td/01_simple_handshake.td
```

### Render

Legacy SVG (anchor pills, RULES footer):

```bash
sva timing render examples/td/01_simple_handshake.td -o examples/out/01_simple_handshake.svg
```

Legacy PNG:

```bash
sva timing render examples/td/01_simple_handshake.td --format png -o examples/out/01_simple_handshake.png
```

Render through a render2 profile (clean WaveDrom — no leaky overlays):

```bash
sva timing render examples/td/01_simple_handshake.td \
    --render-profile clean-wavedrom \
    --seed 0 \
    -o /tmp/clean.svg
```

Render through the native scene-graph renderer with stochastic style:

```bash
sva timing render examples/td/01_simple_handshake.td \
    --render-profile native-random \
    --seed 7 \
    -o /tmp/native.svg
```

Render with audit-strict — fail when leakage, contrast, occlusion, layout
overflow, or reproducibility checks regress:

```bash
sva timing render examples/td/01_simple_handshake.td \
    --render-profile native-random \
    --seed 7 \
    --audit-strict \
    -o /tmp/native_strict.svg
```

#### Available render2 profiles

| Profile id          | Renderer      | Annotation policy      | Notes |
| ------------------- | ------------- | ---------------------- | ----- |
| `debug-current`     | wavedrom      | `debug_leaky`          | Legacy look; prints anchor names + `RULES`. **Audit fails by design.** Human inspection only. |
| `clean-wavedrom`    | wavedrom      | `none`                 | Same WaveDrom backbone, overlays stripped. No leakage. |
| `native-random`     | native_svg    | `geometric_guides`     | Stochastic native renderer. Lane labels + unlabelled vertical guides. |
| `datasheet-native`  | native_svg    | `natural_measurements` | Datasheet-flavored native style; bracket text for `visible_text` bounds. |
| `document-native`   | native_svg    | `geometric_guides`     | Native + page composition (caption / paragraph / page chrome) + degradation. |
| `ood-native`        | native_svg    | `natural_measurements` | Exotic axes (curved transitions, hatched buses, low contrast). |
| `undulate-random`   | undulate      | `geometric_guides`     | Requires `pip install undulate`. |
| `tikz-datasheet`    | tikz_timing   | `natural_measurements` | Requires `pdflatex` (and ideally `dvisvgm`/`pdftoppm`). |
| `plantuml-ood`      | plantuml      | `none`                 | Requires `plantuml` on PATH or `PLANTUML_JAR`. Held out of training. |
| `gtkwave-ood`       | gtkwave       | `none`                 | Requires `gtkwave` on PATH. Held out of training. |
| `ascii-rfc`         | ascii         | `none`                 | UTF-8 ASCII waveform + monospace PNG (when Pillow is installed). Profile produces no SVG; use programmatic API or `--format png`. |

Run `python -c "from sva_toolkit.timing.render2.adapters.registry_bootstrap import bootstrap_external_renderers; print(bootstrap_external_renderers())"` to see which adapters are available locally; profiles whose adapter is missing report `missing_dependency:<name>` or `missing_executable:<name>` and are skipped automatically by the dataset sampler.

### Generate dataset

```bash
sva timing generate-dataset \
    --count 10 \
    --seed 1 \
    --out /tmp/sva-tiny \
    --format svg \
    --render-profile-set train_v2 \
    --emit-render-specs \
    --audit-strict
```

The full set of render2 flags on `generate-dataset`:

- `--render-profile-set <name>` — one of `train_v2`, `val_seen_style`,
  `val_unseen_style`, `test_ood`. Default `train_v2`.
- `--render-profile <name>` — pin one profile for every record (mutually
  exclusive with `--render-profile-set`).
- `--target-policy <visual|debug_keep_all>` — `visual` (default)
  canonicalizes anchor / window / constraint names to `a0/a1/...`,
  `w0/w1/...`, `c0/c1/...` and drops paraphrase property overlays.
- `--emit-render-specs` / `--no-emit-render-specs` — write per-record
  `render_specs/td_*.json`. Default emit.
- `--audit-strict` / `--no-audit-strict` — reject records whose
  rendered text leaks target tokens, whose contrast is below threshold,
  whose required signals are not visible, or whose layout overflows.
  Default strict.
- `--style-holdout <ids>` — CSV of profile ids to skip (resampling
  counts as `holdout_style` rejection).
- `--degradation-holdout <families>` — CSV of degradation family names
  to skip (`holdout_degradation`).
- `--annotation-holdout <policies>` — CSV of annotation policy names to
  skip (`holdout_annotation`).

### Validate generated dataset

```bash
sva timing validate-dataset --dataset /tmp/sva-tiny --strict --strict-visual
```

`--strict-visual` adds checks for `render_specs/`, `audits/` and visual
coverage axes alongside the existing semantic checks.

### SVA bridges (unchanged)

```bash
sva timing emit-sva    examples/td/11_emit_sva_bridge.td -o examples/out/11_emit_sva_bridge.sv
sva timing extract-sva examples/sva/11_emit_sva_bridge.sv -o examples/out/11_emit_sva_bridge.td
sva timing bundle-sva  examples/sva/11_emit_sva_bridge.sv examples/sva/12_extract_sva_bridge.sv -o examples/out/bundled.td
```

## Programmatic Usage

### Parsing (unchanged)

```python
from pathlib import Path
from sva_toolkit.timing.frontend.parser import parse_diagram

document = parse_diagram(Path("examples/td/01_simple_handshake.td").read_text())
print(document.name, document.ticks)
```

### Legacy SVG renderer (unchanged)

```python
from sva_toolkit.timing import render_diagram_svg
svg = render_diagram_svg(document)
```

### render2 — full clean pipeline

```python
import random

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.visual import lower_to_visual_document, TargetPolicy
from sva_toolkit.timing.render2 import (
    DEFAULT_REGISTRY,
    build_timing_scene,
)
from sva_toolkit.timing.render2.profiles import PROFILE_NATIVE_RANDOM
from sva_toolkit.timing.render2.spec_sampler import sample_render_spec
from sva_toolkit.timing.render2.pipeline import render

semantic = parse_diagram(open("examples/td/01_simple_handshake.td").read())

# 1. lower the semantic document into a visual-recoverable target
lowered = lower_to_visual_document(semantic, TargetPolicy.visual())
visual_dsl = emit_timing_dsl(lowered.visual_document)

# 2. build a renderer-independent scene
scene = build_timing_scene(lowered.visual_document, semantic_document=semantic)

# 3. sample a deterministic RenderSpec for the chosen profile
rng = random.Random(7)
spec = sample_render_spec(rng, profile=PROFILE_NATIVE_RANDOM, scene=scene)

# 4. render with audits
outcome = render(scene, spec, target_dsl_text=visual_dsl, enforce_audits=True)
print("audits passed:", outcome.audits_passed)
print("leaked tokens:", outcome.leakage.leaked_tokens)
print("contrast:", outcome.contrast.minimum_contrast)
print("svg length:", len(outcome.result.svg_text or ""))
```

Choosing a renderer manually:

```python
renderer = DEFAULT_REGISTRY.get(spec.renderer_id)  # "native_svg" / "wavedrom" / "ascii" / ...
result = renderer.render(scene, spec)
```

### Visual lowering only (no rendering)

```python
from sva_toolkit.timing.visual import lower_to_visual_document, TargetPolicy

lowered = lower_to_visual_document(document, TargetPolicy.visual())
print(lowered.anchor_renames)            # {'req_rise': 'a0', 'ack_rise': 'a1', ...}
print(lowered.visibility.bound_visibility)  # {'w0': VisibilityClass.VISIBLE_GEOMETRY}
print(lowered.dropped_properties)        # tuple of dropped paraphrase overlays
```

### Page composition + degradation

```python
from sva_toolkit.timing.render2 import compose_record

composed = compose_record(scene, spec, outcome.result, rng=random.Random(7))
open("/tmp/td_000000.png", "wb").write(composed.image_bytes)
print(composed.degradation_chain)
```

## Audits and per-record metadata

Every dataset record is accompanied by:

- `dsl/td_*.td` — canonical visual DSL (target).
- `semantic/td_*.td` — canonical semantic DSL (debug).
- `svg/td_*.svg` — rendered SVG (when applicable).
- `png/td_*.png` — rasterized + composed + degraded image.
- `render_specs/td_*.json` — full `RenderSpec` (round-trippable via
  `sva_toolkit.timing.render2.serialization`).
- `audits/td_*.json` — every audit report:
  - `leakage` — rendered tokens vs. target tokens vs. allowed.
  - `target_visibility` — required signals all rendered.
  - `contrast` — minimum waveform/background luminance contrast.
  - `occlusion` — per-lane fraction occluded by decorations / nuisance.
  - `layout_overflow` — primitive bboxes within canvas bounds.
  - `reproducibility` — same RenderSpec ⇒ same SVG digest twice.
  - `lowering_visibility` — per-field visibility class + rationale.

Inspect the leakage report on a sample record:

```bash
jq '.leakage | {passed, leaked_tokens, allowed_tokens, rendered_tokens}' /tmp/sva-tiny/audits/td_000000.json
```

## API Reference

Stable public exports (legacy):

- `render_diagram_svg(document) -> str`
- `render_diagram_png(document, path) -> None`

render2 surface:

- `sva_toolkit.timing.visual.lower_to_visual_document(doc, policy=...)`
- `sva_toolkit.timing.visual.TargetPolicy.visual() / .debug_keep_all()`
- `sva_toolkit.timing.render2.build_timing_scene(visual_doc, semantic_document=...)`
- `sva_toolkit.timing.render2.profiles.PROFILE_*` and `PROFILE_SET_*`
- `sva_toolkit.timing.render2.spec_sampler.sample_render_spec(rng, profile=, scene=)`
- `sva_toolkit.timing.render2.spec_sampler.sample_profile(rng, profile_set)`
- `sva_toolkit.timing.render2.pipeline.render(scene, spec, target_dsl_text=, enforce_audits=)`
- `sva_toolkit.timing.render2.compose.compose_record(scene, spec, result, rng=)`
- `sva_toolkit.timing.render2.protocol.DEFAULT_REGISTRY` — `RendererRegistry`
- `sva_toolkit.timing.render2.adapters.registry_bootstrap.bootstrap_external_renderers()`
- `sva_toolkit.timing.render2.audit.{leakage,target_visibility,contrast,layout_overflow,reproducibility}`
- `sva_toolkit.timing.render2.visual_coverage.VisualCoverageTracker`

Frequently used internal modules:

- `sva_toolkit.timing.frontend.parser.parse_diagram`
- `sva_toolkit.timing.frontend.validate.validate_diagram`
- `sva_toolkit.timing.bridge.emit_sva.emit_parameterized_sva`
- `sva_toolkit.timing.bridge.from_sva.extract_sva_scenario`
- `sva_toolkit.timing.bridge.from_sva.bundle_sva_scenarios`
- `sva_toolkit.timing.bridge.to_dsl.emit_timing_dsl`

Core model classes live under `sva_toolkit.timing.core.scenario`, including:

- `ScenarioDocument`
- `ClockingSpec`
- `SignalDecl`
- `Anchor`
- `TimeWindow`
- `PropertyOverlay`

## V3 Notes

- SVG rendering is available in the base install.
- PNG rendering and the Phase 6 raster/degradation pipeline both expect a
  working SVG-to-PNG path. Install `cairosvg` (with libcairo native lib) for
  the legacy PNG. The render2 composer additionally tries `resvg-py` and
  `wand`; if none are loadable it falls back to a blank synthetic raster
  and records a composition warning per record.
- WaveDrom is required for the legacy renderer and the `clean-wavedrom`,
  `debug-current` profiles. Optional adapters (`undulate`, `tikz_timing`,
  `plantuml`, `gtkwave`) are auto-registered when their tooling is found
  on the system.
- The `examples/td/` and `examples/sva/` directories in V3 are ported from
  the V2 timing example suite.
- Extraction and bundling are best-effort structure recovery workflows,
  not full semantic proof steps.

## Helper scripts

- `scripts/render_profile_gallery.py` — render a `.td` across every render2
  profile for visual inspection. Writes a `gallery.json` manifest.
- `scripts/generate_timing_dataset.py` — build the canonical
  train / val_seen_style / val_unseen_style / test_synthetic_ood splits in
  one command. Thin wrapper over `sva timing generate-dataset` with the
  right profile-sets and holdouts per split.

See `scripts/README.md` for full usage.

## Related Docs

- [Architecture](architecture.md)
- [Formal verification](sva-formal.md)
- [Timing diagram dataset generation](timing-dataset-generation.md)
- [Examples](../examples/README.md)
