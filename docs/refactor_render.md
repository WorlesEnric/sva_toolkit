## Core recommendation

Do **not** implement “the expected render” as one improved WaveDrom renderer. Implement it as a **stochastic observation model**:

[
\text{canonical visual DSL} \rightarrow \text{semantic timing scene} \rightarrow \text{sampled renderer/style/page/degradation} \rightarrow \text{image}
]

The dataset should learn invariants of timing diagrams, not the quirks of one renderer. Your current brief already identifies the two biggest failure modes: the visual distribution is effectively one point in style space, and the overlay layer writes target-side tokens such as anchor names, delay labels, and rule summaries into the image. Those two issues must be fixed before adding more renderers, otherwise the model will simply learn a larger set of shortcuts. 

The right redesign is:

1. **Split semantic content from visual presentation.**
2. **Define a visual-recoverability contract.**
3. **Make renderers pluggable.**
4. **Generate many styles at the vector level, not only by PNG augmentation.**
5. **Add realistic annotations and helper lines as controlled nuisances, not as label leaks.**
6. **Track visual coverage just like you track semantic coverage.**
7. **Evaluate on held-out renderers and real diagrams, not only random samples from the same renderer family.**

---

## 1. Fix the target first: the model can only recover what is visible

The current target is too strong for a pure image-to-DSL task. Arbitrary anchor names, named windows, rule prose, and property labels are not recoverable from waveform geometry unless they are printed in the figure. A VLM cannot infer that a transition should be named `req_rise` rather than `a0`, `event_1`, or `trigger`, unless that string appears somewhere in the image.

So you need two internal DSL layers:

### A. Semantic DSL

This is the full internal generator document: rich names, idioms, constraints, generated anchors, hidden metadata, property overlays, and so on. This is useful for generation and validation.

### B. Visual DSL

This is the supervised image target. It must contain only fields that are one of:

| Visibility class     | Meaning                                                                               | Safe for image-to-DSL target? |
| -------------------- | ------------------------------------------------------------------------------------- | ----------------------------- |
| `visible_geometry`   | Recoverable from waveform shape, lane order, tick grid, bus run positions             | Yes                           |
| `visible_text`       | Printed naturally in the figure, e.g. signal names, bus values, timing bracket labels | Yes                           |
| `visible_convention` | Deterministically canonicalized by visual order, e.g. `a0`, `a1`, `w0`                | Yes                           |
| `hidden_semantic`    | Exists only in generator metadata                                                     | No                            |
| `debug_overlay`      | Printed only by your renderer to help the model                                       | No for normal training        |

The training target can still be canonical DSL, but it should be canonical DSL **after lowering to the visual contract**:

```text
ScenarioDocument
  └─ lower_to_visual_document(...)
       └─ emit_timing_dsl
            └─ parse
                 └─ validate
                      └─ emit_timing_dsl
```

That preserves your existing round-trip/validate constraint while preventing impossible labels.

### Concrete example

Bad target:

```td
anchor req_rise when req rises
anchor handshake when req && ack
show ack in [1:4] after req_rise
```

If the image does not literally print `req_rise` or `handshake`, the visual target should become something like:

```td
anchor a0 when req rises
anchor a1 when req && ack
show ack in [1:4] after a0
```

Here `a0` and `a1` are assigned by deterministic visual order, such as left-to-right event order, then lane order, then predicate type. The model can infer that convention.

For exact bounds like `[1:4]`, require one of two things: either the grid/tick spacing makes the number geometrically inferable, or the bracket text is naturally printed. If neither is true, do not ask the model to output the exact bound.

---

## 2. Remove target leakage, but keep realistic annotations

You should not remove all annotations. Real timing diagrams often contain helper lines, arrows, measurement brackets, circled regions, callouts, table boundaries, captions, and handwritten marks. The problem is not annotations themselves; the problem is **annotations that directly expose the target in an artificial synthetic-only way**.

Create an explicit annotation policy:

```python
class AnnotationPolicy(Enum):
    NONE = "none"
    NUISANCE_ONLY = "nuisance_only"
    GEOMETRIC_GUIDES = "geometric_guides"
    NATURAL_MEASUREMENTS = "natural_measurements"
    DEBUG_LEAKY = "debug_leaky"  # never in train/val/test, only manual inspection
```

Recommended behavior:

| Current overlay             |     Keep? | Replacement                                                                                                |
| --------------------------- | --------: | ---------------------------------------------------------------------------------------------------------- |
| Anchor pill with `req_rise` |        No | Optional unlabeled marker, or canonical `a0` only if target uses `a0`                                      |
| Arrow label `[1:4]`         | Sometimes | Keep only when the visual target expects that bound and the label is rendered in a natural datasheet style |
| Hold highlight              | Sometimes | Keep as realistic shaded/hatched region, but do not always use the same color/style                        |
| `RULES` footer              |        No | Move to debug-only render profile                                                                          |
| Plain-English summaries     |        No | Debug-only; this is target paraphrase leakage                                                              |
| Vertical helper lines       |       Yes | Use unlabeled, dashed, dotted, faint, or noisy styles                                                      |
| Timing measurement brackets |       Yes | Use realistic labels like `tSU`, `tH`, `1 cycle`, `≤ 4 clk`, but only when target accounts for them        |
| Captions and figure labels  |       Yes | Treat as document-context distractors unless the label is part of the target                               |

Add a **leakage audit** before accepting a record:

```python
def audit_rendered_text(record):
    target_tokens = tokenize(record.visual_dsl)
    rendered_tokens = collect_svg_text_nodes(record.svg)

    allowed = {
        *record.visible_signal_names,
        *record.visible_bus_values,
        *record.visible_timing_labels,
        *record.canonical_visual_anchor_names,  # e.g. a0/a1 only when intended
    }

    leaked = (target_tokens & rendered_tokens) - allowed
    if leaked:
        reject("render_text_leakage", tokens=sorted(leaked))
```

For raster-only outputs, do not depend on OCR for the audit. Since you control the synthetic renderer, audit the vector/text primitives before rasterization.

---

## 3. Introduce a renderer abstraction

Right now, your public path is hardwired around WaveDrom layout extraction. That makes visual diversification painful. Instead, define a renderer interface that accepts a renderer-independent timing scene and returns image data plus metadata.

```python
@dataclass(frozen=True)
class RenderSpec:
    renderer_id: str
    style: "StyleSpec"
    layout: "LayoutSpec"
    annotations: "AnnotationSpec"
    page: "PageSpec"
    raster: "RasterSpec"
    degradation: "DegradationSpec"
    seed: int

@dataclass
class RenderResult:
    svg_text: str | None
    png_bytes: bytes | None
    layout: "DiagramLayout"
    rendered_text: list["TextPrimitive"]
    bboxes: dict[str, "BBox"]
    visibility: "VisibilityReport"
    render_spec: RenderSpec
    warnings: list[str]

class TimingRenderer(Protocol):
    id: str
    capabilities: set[str]

    def supports(self, scene: "TimingScene", spec: RenderSpec) -> bool: ...
    def render(self, scene: "TimingScene", spec: RenderSpec) -> RenderResult: ...
```

The key intermediate object is **not WaveJSON**. It should be your own scene IR:

```python
@dataclass
class TimingScene:
    lanes: list[LaneScene]
    ticks: TickModel
    cuts: list[CutRegion]
    events: list[VisualEvent]
    constraints: list[VisualConstraint]
    visible_target: ScenarioDocument
```

The scene should express:

```text
lane name
lane type: bit / bus / clock / analog-ish / unknown / high-Z
sample runs
bus value runs
unknown/high-Z regions
cuts/compressed regions
event positions
time windows
optional measurement brackets
optional nuisance decorations
```

Then each backend lowers `TimingScene` into its own format.

---

## 4. Use multiple renderers, but treat them as style families, not truth

I did not find one open-source tool that solves the whole problem. The useful tools are partial backends you can put behind your renderer interface.

| Tool/backend                   | Why it is useful                                                                                                                                                                                                                                                          | Limitation                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Your native SVG renderer**   | Best long-term backbone. You control every primitive, bbox, style, and recoverability rule.                                                                                                                                                                               | You must implement style diversity yourself.                                                          |
| **WaveDrom / YoWASP WaveDrom** | Keep as one common style. WaveDrom is open source and turns WaveJSON into SVG; YoWASP provides a self-contained Python/JS renderer with `render(source) -> str`. ([GitHub][1])                                                                                            | Should not be the only style. Existing Python package geometry coupling is fragile.                   |
| **Undulate**                   | Very relevant. It is WaveJSON-compatible-ish but adds features you care about: annotations, global time compression, vertical/horizontal lines, and style overloading for font size, fill, stroke, stroke width, dash arrays, etc.; it outputs SVG/PDF/PNG. ([GitHub][2]) | Still a diagram-code renderer, not a scanned-document simulator. Needs adapter and capability checks. |
| **tikz-timing**                | Good for datasheet/paper/LaTeX-looking diagrams. CTAN describes it as a TikZ package for generating digital timing diagrams, including larger tabular timing diagrams. ([CTAN][3])                                                                                        | Requires TeX toolchain. Harder to get precise layout metadata unless you control parameters.          |
| **PlantUML timing diagrams**   | Adds a UML-ish timing style. PlantUML supports timing participants such as `binary`, `clock`, `concise`, `rectangle`, `robust`, and `analog`. ([PlantUML.com][4])                                                                                                         | Semantics differ from hardware waveform diagrams; use for subset/OOD style only.                      |
| **GTKWave screenshot backend** | Important for EDA-viewer style. GTKWave reads standard VCD/EVCD plus FST/GHW and other waveform dump formats. ([GitHub][5])                                                                                                                                               | Headless screenshot automation is more brittle than SVG renderers. Use as optional/held-out style.    |
| **pyvcd + GTKWave save files** | Useful for generating `.gtkw` files that specify displayed traces, aliases, colors, ordering, etc. ([PyVCD][6])                                                                                                                                                           | Only helps configure GTKWave; it is not itself a renderer.                                            |
| **asciiwave**                  | Useful for ASCII waveform examples embedded in RFCs, comments, Markdown, or monospace docs. It converts WaveDrom JSON into ASCII art and exposes formatting knobs such as `hscale` and graphics style. ([GitHub][7])                                                      | Supports a WaveJSON subset. Not a replacement for graphical renderers.                                |
| **Augraphy**                   | Strong fit for scanned/faxed/photocopied document degradation. It simulates print, scan, fax, paper texture, ink/toner degradation, folds, and related document effects. ([GitHub][8])                                                                                    | Post-processing only; it cannot create waveform geometry diversity by itself.                         |
| **Albumentations / OpenCV**    | Good for fast image augmentation, geometric transforms, pixel transforms, resizing, affine/perspective warps, and integration into CV training pipelines. ([Albumentations][9])                                                                                           | Generic augmentation; must be constrained by recoverability checks.                                   |

The most immediately useful external renderer for your case is probably **Undulate**, because it explicitly targets style overrides and annotations/helper lines. The most important renderer overall is still a **native renderer you own**, because you need precise metadata, leakage auditing, visibility checking, and recoverability validation.

---

## 5. Native renderer should become the reference renderer

A native renderer should not be one style. It should be a **scene-graph renderer with randomized style kernels**.

### Scene graph primitives

Use a small vector primitive library:

```python
@dataclass
class Primitive:
    role: str
    id: str | None
    z: int

@dataclass
class Line(Primitive):
    p0: Point
    p1: Point
    stroke: Stroke

@dataclass
class Polyline(Primitive):
    points: list[Point]
    stroke: Stroke

@dataclass
class Path(Primitive):
    d: str
    stroke: Stroke
    fill: Fill | None

@dataclass
class Text(Primitive):
    text: str
    anchor: Point
    font: FontSpec
    bbox_policy: str
    visibility_class: str

@dataclass
class Rect(Primitive):
    bbox: BBox
    stroke: Stroke | None
    fill: Fill | None
    radius: float = 0
```

Logical roles should include:

```text
background
outer_card
lane_label
lane_separator
grid_major
grid_minor
tick_label
bit_wave_high
bit_wave_low
bit_transition
bus_region
bus_region_edge
bus_value_text
unknown_region
hiz_region
cut_marker
measurement_bracket
vertical_helper_line
horizontal_helper_line
annotation_arrow
nuisance_text
caption_text
```

Then style randomization maps logical roles to actual visual choices.

### Style axes to randomize

Do not only randomize colors. Randomize geometry too.

| Axis           | Examples                                                                                             |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| Font           | Helvetica-like, Times-like, Courier-like, condensed, small caps, bold labels, all-monospace          |
| Stroke         | 0.6–3 px, rounded vs square caps, solid/dashed/dotted, anti-aliased vs pixelated                     |
| Wave geometry  | sharp transitions, slanted transitions, curved transitions, step-like bus diamonds, boxed bus values |
| Grid           | no grid, faint vertical ticks, dense grid, only major ticks, tick labels above/below                 |
| Lane layout    | compact, spacious, labels left/right/inside, multiline labels, grouped lanes                         |
| Bus rendering  | yellow fill, white fill, no fill, hatched unknown, inline value text, value boxes                    |
| Unknown/high-Z | X hatch, gray block, green hatch, orange hatch, diagonal stripes, dashed outline                     |
| Cuts           | zigzag break, vertical ellipsis, gray omitted band, double slash                                     |
| Annotations    | thin callouts, measurement brackets, arrows, hand-drawn circles, vertical guides                     |
| Page context   | figure caption, surrounding paragraph text, table border, cropped page fragment                      |
| Color mode     | full color, grayscale, black-and-white, low-contrast photocopy, inverted dark EDA viewer             |
| Rasterization  | DPI, antialiasing, JPEG quality, blur, sharpening, thresholding, halftone                            |

A critical rule: **sample style continuously**, not from ten named presets only. Domain randomization works by exposing the model to broad visual variability; the original domain-randomization argument is that with enough simulator variability, real images can appear as another variation. ([ar5iv][10]) Document-domain randomization work similarly shows that style mismatch hurts document extraction and that randomized pseudo-pages can transfer to real document layouts. ([arXiv][11])

---

## 6. Add page-level composition, not just diagram-level augmentation

Real diagrams are rarely isolated perfect crops. Add a `PageComposer` after vector rendering:

```text
diagram SVG/PNG
  └─ place on synthetic page
       ├─ caption above/below
       ├─ paragraph fragments around it
       ├─ table cell or figure border
       ├─ page number/header/footer
       ├─ nearby unrelated labels
       ├─ crop window
       └─ document degradation
```

This matters because many VLM failures come from **context confusion**, not waveform recognition. The model must learn which text belongs to the timing diagram and which text is a caption, page header, or unrelated paragraph.

Add nuisance text such as:

```text
Figure 7. Read transaction timing
tCLK min = 10 ns
valid sampled on rising edge
not to scale
reserved
see Table 12
```

But annotate it as `nuisance_text`, not target text, unless you explicitly want a task variant that uses captions as evidence.

---

## 7. Model helper lines and annotations as separate object types

Real helper lines are hard because some are semantic and some are noise. You need both.

### Semantic helper lines

These encode timing information that should be reflected in the target.

Examples:

```text
vertical line at trigger event
vertical line at response event
measurement bracket spanning 1..4 cycles
setup/hold window around clock edge
```

Use these only when the visual target contains the corresponding relation.

### Nuisance helper lines

These make the image harder but should not alter the target.

Examples:

```text
faint vertical cursor
random document table border
editor selection line
hand-drawn circle
arrow pointing to a signal without useful text
highlight rectangle around a region
```

For nuisance overlays, enforce recoverability constraints:

```python
max_occluded_edge_fraction_per_lane <= 0.15
min_visible_signal_label_fraction >= 0.85
min_wave_contrast >= threshold
no_crop_of_required_bus_value_text
no_crop_of_required_tick_labels
```

The generator can accept hard examples, but it should know why they are hard.

---

## 8. Add a raster/degradation pipeline with recoverability checks

Post-render augmentation is necessary but insufficient. Use it after vector-level variation.

Recommended pipeline:

```text
vector render
  └─ rasterize at sampled DPI
       └─ optional page composition
            └─ geometric transform
                 └─ document degradation
                      └─ compression/noise
                           └─ crop/resize
                                └─ recoverability check
```

Useful degradation families:

| Family      | Operations                                                      |
| ----------- | --------------------------------------------------------------- |
| DPI/scale   | 72, 96, 150, 200, 300, 600 DPI; downsample/upsample             |
| Compression | JPEG artifacts, PNG quantization, WebP-like blur                |
| Scan        | blur, threshold, speckle, dust, streaks, faded ink              |
| Photocopy   | low contrast, uneven illumination, paper texture, bleed-through |
| Camera      | rotation, perspective warp, shadow, glare, focus blur           |
| Crop        | partial surrounding context, tight crop, off-center diagram     |
| Color       | grayscale, monochrome threshold, faded colors, dark mode        |
| Morphology  | line thinning/thickening, broken strokes, small gaps            |

Augraphy is especially useful for document-like degradation, while OpenCV/Albumentations are better for fast geometric and pixel transforms. ([GitHub][8])

---

## 9. Store render metadata with every record

Each record should include more than image + DSL. Store the sampled render spec and visibility report.

```json
{
  "id": "train_000123",
  "seed": 12345,
  "semantic_dsl_path": "semantic/train_000123.td",
  "target_dsl_path": "dsl/train_000123.td",
  "image_path": "images/train_000123.png",
  "render_spec_path": "render_specs/train_000123.json",
  "renderer_id": "native_svg",
  "style_family": "datasheet_bw_dense_grid",
  "annotation_policy": "natural_measurements",
  "degradation_profile": "photocopy_low_contrast",
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
  }
}
```

This lets you debug failures by renderer, style, degradation, and visibility class instead of treating all errors as model errors.

---

## 10. Extend coverage tracking to visual coverage

Your existing coverage tracker covers semantic axes. Add visual axes:

```text
renderer_id
style_family
font_family_bucket
font_size_bucket
stroke_width_bucket
grid_mode
tick_label_mode
bus_style
unknown_style
cut_style
annotation_policy
helper_line_count_bucket
nuisance_text_count_bucket
page_context_mode
color_mode
raster_dpi_bucket
compression_bucket
blur_bucket
perspective_bucket
crop_bucket
occlusion_bucket
recoverability_class
leakage_audit_status
```

Then summary metrics should include both semantic and visual histograms:

```json
{
  "coverage": {
    "semantic": { "...": "..." },
    "visual": {
      "renderer_id": {
        "native_svg": 42000,
        "wavedrom": 9000,
        "undulate": 11000,
        "tikz_timing": 7000,
        "plantuml": 3000,
        "gtkwave": 2000,
        "ascii": 1000
      },
      "annotation_policy": {
        "none": 22000,
        "nuisance_only": 18000,
        "geometric_guides": 19000,
        "natural_measurements": 16000
      }
    }
  }
}
```

Also add rejection reasons:

```text
render_text_leakage
unrecoverable_hidden_anchor_name
required_label_cropped
required_bus_value_occluded
low_contrast
external_renderer_failed
layout_overflow
unsupported_scene_for_renderer
```

---

## 11. Suggested renderer mix

A good first production mix would be:

| Bucket                     |        Percentage | Purpose                               |
| -------------------------- | ----------------: | ------------------------------------- |
| Native SVG randomized      |            45–55% | Main controllable distribution        |
| WaveDrom / YoWASP WaveDrom |            10–15% | Common modern WaveJSON style          |
| Undulate                   |            10–15% | Annotation/helper-line-rich style     |
| tikz-timing                |             5–10% | Paper/datasheet/LaTeX style           |
| PlantUML timing            |              3–7% | UML-ish OOD style                     |
| GTKWave screenshots        |              3–7% | EDA viewer style                      |
| ASCII/monospace            |              1–3% | RFC/comment/plain-text waveform style |
| Real annotated eval set    | not mixed blindly | Evaluation and calibration            |

Keep at least one renderer family completely held out from training. For example, train without GTKWave and test on GTKWave screenshots. Also reserve a real-world manually annotated test set.

---

## 12. Concrete implementation plan inside `sva-toolkit`

### Step 1: Add render profiles without breaking the CLI

Keep the current CLI but add profile arguments:

```bash
sva timing render input.td \
  --render-profile debug-current

sva timing generate-dataset \
  --render-profile-set train_v2 \
  --target-policy visual \
  --image-format png \
  --emit-render-specs
```

Profiles:

```text
debug-current         # current overlays, for humans only
clean-wavedrom        # WaveDrom without target-leaking overlays
native-random         # native randomized vector renderer
document-random       # page composition + degradation
ood-gtkwave           # optional held-out eval profile
```

### Step 2: Add `RenderSpec` and seeded sampling

Use your existing seed discipline:

```python
render_rng = generation_rng.child(f"record:{record_id}:render")
spec = render_spec_sampler.sample(render_rng, scene, profile_set)
```

Store the sampled `RenderSpec` beside the image.

### Step 3: Build `lower_to_visual_document`

This is the most important semantic change.

```python
def lower_to_visual_document(doc: ScenarioDocument, policy: TargetPolicy) -> ScenarioDocument:
    """
    Convert full semantic document to image-recoverable canonical document.
    - Replace hidden arbitrary anchor/window names with visual canonical names.
    - Drop rule summaries and non-visible prose.
    - Keep exact bounds only when geometry/text policy makes them recoverable.
    - Preserve parse/validate/emit canonicalization.
    """
```

### Step 4: Build `TimingScene`

```python
semantic_doc = generated_doc
visual_doc = lower_to_visual_document(semantic_doc, policy="visual")
scene = build_timing_scene(visual_doc, semantic_doc=semantic_doc)
```

The scene carries both the target and optional non-target metadata used for rendering nuisance context.

### Step 5: Implement native renderer first

The native renderer should produce:

```text
SVG
layout metadata
text primitives
primitive role map
bbox map
visibility report
```

Avoid reverse-engineering external renderer geometry. Your own renderer is the source of truth.

### Step 6: Add external adapters

Adapters should be optional and capability-based:

```python
class UndulateRenderer:
    capabilities = {"bit", "bus", "cuts", "annotations", "style_overrides"}

class TikzTimingRenderer:
    capabilities = {"bit", "bus_basic", "clock", "latex_style"}

class GTKWaveRenderer:
    capabilities = {"vcd_bit", "vcd_bus", "eda_screenshot"}
```

If a scene requires unsupported features, reject that renderer for that scene and resample.

### Step 7: Add augmentation pipeline

```python
result = renderer.render(scene, spec)
image = rasterize(result.svg_text, spec.raster)
image = compose_page(image, spec.page)
image = degrade(image, spec.degradation)
visibility = check_visibility(image, result.layout, spec)
```

### Step 8: Add audits

Required audits:

```text
DSL parse/validate/emit round trip
target visibility audit
rendered text leakage audit
layout overflow audit
minimum contrast audit
crop/occlusion audit
external renderer reproducibility audit
```

---

## 13. How to handle annotations/helper lines safely

Create a small annotation DSL internal to rendering:

```python
@dataclass
class Decoration:
    kind: Literal[
        "vertical_guide",
        "horizontal_guide",
        "measurement_bracket",
        "callout_arrow",
        "highlight_region",
        "caption",
        "nuisance_text",
        "handdrawn_mark"
    ]
    semantic: bool
    target_ref: str | None
    text: str | None
    visibility_class: str
    style: DecorationStyle
```

Examples:

```python
Decoration(
    kind="vertical_guide",
    semantic=True,
    target_ref="anchor:a0",
    text=None,
    visibility_class="visible_geometry",
)
```

```python
Decoration(
    kind="nuisance_text",
    semantic=False,
    target_ref=None,
    text="see timing notes below",
    visibility_class="nuisance",
)
```

```python
Decoration(
    kind="measurement_bracket",
    semantic=True,
    target_ref="window:w0",
    text="1–4 cycles",
    visibility_class="visible_text",
)
```

Then the target builder can reason about what is safe.

---

## 14. Dataset split strategy

Use multiple split dimensions:

### Semantic split

No shared generated topology/template IDs between train and test.

### Style split

Hold out complete style families.

Example:

```text
train: native, wavedrom, undulate, tikz
val_seen_style: same renderer families, new semantics
val_unseen_style: plantuml, gtkwave
test_synthetic_ood: held-out style profiles + stronger degradation
test_real: manually annotated real diagrams
```

### Degradation split

Hold out some degradation profiles:

```text
train: clean, mild scan, mild JPEG, slight crop
test: severe photocopy, perspective camera, dense page context
```

### Annotation split

Hold out annotation-heavy diagrams:

```text
train: no annotations + mild helper lines
test: dense helper lines, callouts, measurement brackets
```

This prevents “random sample from same distribution” validation from overestimating robustness.

---

## 15. Evaluation should not be only exact string match

Use several metrics:

| Metric                 | Purpose                                                         |
| ---------------------- | --------------------------------------------------------------- |
| Parse validity         | Did the model produce syntactically valid DSL?                  |
| Validation success     | Does DSL satisfy toolkit validator?                             |
| Canonical exact match  | Strict target match                                             |
| Semantic equivalence   | Ignores harmless renaming/canonicalization differences          |
| Lane recovery F1       | Signal names, lane order, lane types                            |
| Waveform edit distance | Per-lane state/run recovery                                     |
| Event recovery F1      | Anchor positions/predicates                                     |
| Bound accuracy         | Exact or tolerance-based timing window recovery                 |
| Bus value accuracy     | OCR/value recovery                                              |
| Robustness slices      | Accuracy by renderer, degradation, annotation policy, crop, DPI |
| Leakage sensitivity    | Accuracy drop when debug overlays are removed                   |

The most important diagnostic is: **train with and without leaky overlays, then evaluate on no-overlay diagrams**. If performance collapses, the model was not learning waveform semantics.

---

## 16. Practical MVP

A good first version does not need every backend.

Build this first:

```text
visual target lowering
native randomized SVG renderer
annotation policy without leakage
rendered-text audit
raster degradation pipeline
visual coverage summary
held-out WaveDrom/no-overlay evaluation
```

Then add:

```text
Undulate adapter
tikz-timing adapter
page composition
Augraphy document degradation
GTKWave optional backend
ASCII backend
real eval set
```

The first milestone is not “more beautiful images.” It is:

```text
No target leakage.
All target fields are visually recoverable.
Same DSL can render into many geometries/styles.
Renderer choice is recorded and balanced.
Held-out styles expose failure modes.
```

---

## 17. Bottom line

Your problem is not that WaveDrom is slightly too clean. The deeper issue is that the current dataset defines the task as “read a WaveDrom-like synthetic figure with target text printed on top.” That is not the same task as “recover timing DSL from arbitrary real specification diagrams.”

The redesign should make rendering a sampled distribution:

```text
P(image | visual DSL, renderer, style, page context, degradation, nuisance annotations)
```

and it should make the supervised target obey a strict rule:

```text
Every target token must be visible, geometrically inferable, or deterministically canonical by visual convention.
```

Use WaveDrom as one backend, not the backbone. Use Undulate, tikz-timing, PlantUML, GTKWave, ASCII, Augraphy, Albumentations, and OpenCV as style/degradation components where they fit. But the center of the system should be your own renderer-independent `TimingScene`, `RenderSpec`, `RenderResult`, visibility audit, and visual coverage tracker.

[1]: https://github.com/wavedrom/wavedrom "GitHub - wavedrom/wavedrom: :ocean: Digital timing diagram rendering engine · GitHub"
[2]: https://github.com/LudwigCRON/undulate "GitHub - LudwigCRON/undulate · GitHub"
[3]: https://ctan.org/pkg/tikz-timing "CTAN: Package tikz-timing"
[4]: https://plantuml.com/timing-diagram "Timing Diagram syntax and features"
[5]: https://github.com/gtkwave/gtkwave/blob/master/README.md "gtkwave/README.md at master · gtkwave/gtkwave · GitHub"
[6]: https://pyvcd.readthedocs.io/en/latest/vcd.gtkw.html "vcd.gtkw — pyvcd  documentation"
[7]: https://github.com/Wren6991/asciiwave "GitHub - Wren6991/asciiwave: Turn WaveDrom timing diagrams into ASCII art · GitHub"
[8]: https://github.com/sparkfish/augraphy "GitHub - sparkfish/augraphy: Augmentation pipeline for rendering synthetic paper printing, faxing, scanning and copy machine processes · GitHub"
[9]: https://albumentations.readthedocs.io/ "albumentations — albumentations 1.1.0 documentation"
[10]: https://ar5iv.labs.arxiv.org/html/1703.06907 "[1703.06907] Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
[11]: https://arxiv.org/abs/2105.14931 "[2105.14931] Document Domain Randomization for Deep Learning Document Layout Extraction"
