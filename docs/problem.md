# Problem Brief: Visual Distribution of the Timing Image-DSL Dataset

This document is written for an external reviewer. It describes the relevant
parts of the toolkit, the current rendering mechanism, and the actual problem
we are trying to solve. It deliberately does not propose solutions.

---

## 1. Tool Overview: `sva-toolkit` and `sva timing`

`sva-toolkit` is a Python 3.11+ CLI distributed under the entry point `sva`.
It groups several subcommands; the relevant group here is `sva timing`.

`sva timing` owns:

- A timing-diagram DSL stored in `.td` files. The DSL describes signal lanes,
  anchors (named events with predicates over signals), time windows between
  anchors, lane constraints (`show ... at | in | before | after | from..until`),
  cuts (omitted/compressed regions), and optional property overlays.
- The canonical in-memory model: `sva_toolkit.timing.core.scenario.ScenarioDocument`.
- A frontend (`parse_diagram`, `validate_diagram`).
- Bridges between the timing DSL and SystemVerilog Assertions: `emit-sva`,
  `extract-sva`, `bundle-sva`.
- A rendering pipeline that turns a `ScenarioDocument` into SVG and, optionally,
  PNG via cairosvg.
- A procedural dataset generator: `sva timing generate-dataset`. This is the
  pipeline relevant to this brief.

The dataset generator is intended to produce supervised training data for a
Vision-Language Model. Given a rendered timing-diagram image, the model must
recover the canonical timing DSL.

The pipeline implemented in `sva_toolkit.timing.generate.dataset` is:

```text
GenerationSpec  ──►  topology graph  ──►  semantic decoration (idioms)
        │                                       │
        ▼                                       ▼
  tick assignment  ──►  waveform synthesis  ──►  ScenarioDocument
                                                    │
                                                    ▼
                                          emit_timing_dsl  ──►  parse  ──►
                                          validate  ──►  emit_timing_dsl
                                                    │
                                                    ▼
                                          canonical DSL  +  one SVG image
```

The training target is the round-tripped canonical DSL. Each accepted record
emits one DSL file and one image (SVG, optionally a single PNG produced by
`cairosvg.svg2png(svg_text)`).

---

## 2. Current Rendering Mechanism

### 2.1 Public renderer entry point

The dataset generator and the `sva timing render` command both call:

```text
sva_toolkit.timing.render.svg.render_diagram_svg(document)
  └─ build_wavedrom_view(document)         # projection layer
  └─ render_wavedrom_svg(view)             # WaveDrom-backed renderer
```

There is exactly one rendering implementation in the public path. A second
native SVG renderer exists at `sva_toolkit/timing/render/waveform.py` but it is
not reachable from the dataset generator and is not used to produce any record.
PNG output is produced exclusively by rasterizing the SVG with cairosvg; there
is no second raster pipeline.

### 2.2 How `render_wavedrom_svg` builds an image

The renderer is a thin orchestration layer on top of the upstream `wavedrom`
PyPI package (`wavedrom>=2.0.3.post3`). It performs five steps:

1. **Build a WaveDrom source dict.** Each lane is converted into a
   WaveDrom-style signal entry. Bit lanes use the wave alphabet
   `{0, 1, x, z, .}` where `.` repeats the previous sample. Bus lanes encode
   value runs as `=` and emit the values into a separate `data` array. The dict
   also carries `head: {tick: 0}` and a heuristically chosen `hscale`.
2. **Hand the dict to `wavedrom.waveform.WaveDrom().render_waveform(...)`.**
   This upstream call produces an SVG drawing using svgwrite and the WaveDrom
   library's hardcoded geometry: lane height, lane pitch, tick width, font
   metrics, lane label positioning, transition shapes, etc.
3. **Reverse-engineer WaveDrom's layout.** Before serializing, the renderer
   stashes WaveDrom's internal layout constants (`xg`, `yh0`, `yh1`, `xs`,
   `ys`, `yo`, `hscale`) onto the SVG root as `data-timing-*` attributes,
   then reads them back as a `WaveDromLayout` dataclass so subsequent overlays
   can be placed correctly.
4. **Layer semantic overlays as additional SVG groups.** Four overlay
   constructions are appended:
   - `timing-event-overlays`: dashed vertical guides plus pill-shaped boxes
     containing the anchor name (e.g. `req_rise`, `handshake`).
   - `timing-rule-overlays`: arrows between trigger and response anchors with
     a delay label (e.g. `[1:4]`, `[0:MAX_LAT]`).
   - `timing-hold-highlights`: translucent green rectangles painted over lanes
     where a `from..until` hold span applies.
   - `timing-rule-summary`: a footer block titled `RULES` listing the
     classified rules in plain English.
5. **Wrap in an outer card.** The renderer composes a final SVG: a white
   background, a gray rounded card containing the WaveDrom drawing and the
   overlay groups, a title row, and a clocking metadata line such as
   `@(posedge clk)`.

### 2.3 Fixed visual choices

Every visual property other than waveform content is hardcoded:

| Axis | Source | Fixed value |
| --- | --- | --- |
| Font family | injected CSS + outer wrap | `Helvetica, Arial, sans-serif` |
| Title size / weight | outer wrap CSS | `16px / 700` |
| Metadata size | outer wrap CSS | `12px` |
| Lane label color | injected CSS override on `#lanes_0 text` | `#0066CC` (blue) |
| Lane label size | injected CSS override | `13px` |
| Waveform stroke | injected CSS override on `#waves_0 path` | `#000000` at `1.5px` |
| Bus value rendering | upstream WaveDrom | WaveDrom default (yellow-ish fill, equal-sign syntax) |
| Anchor pill style | overlay code | rounded blue box (`#eff6ff` fill, `#2563eb` stroke), label at `11px / 600` blue |
| Response arrow color | overlay code constant `RESPONSE_COLOR` | `#b45309` (amber) |
| Hold highlight | overlay code constant `HOLD_COLOR` | `#bbf7d0` fill at 0.42 opacity |
| Not-before color | overlay code constant `NOT_BEFORE_COLOR` | `#b91c1c` (red) |
| Outer card | outer wrap | white background, `#d8d8d8` rounded card border, `12px` corner radius |
| Margins / padding | constants | `OUTER_MARGIN_X=20`, `OUTER_MARGIN_Y=16`, `HEADER_HEIGHT=54`, `CARD_PADDING=12` |
| Overlay track pitch | constants | `EVENT_TRACK_PITCH=24`, `RULE_TRACK_PITCH=22`, `SUMMARY_LINE_PITCH=18` |
| Footer summary | overlay code | titled `RULES`, plain-English bullet list |
| Antialiasing / DPI | cairosvg default | clean vector rasterization, no scan/JPEG/blur |

The unused native renderer at `render/waveform.py` is similarly a single fixed
style: `GRID_COLOR=#E8E8E8`, `NAME_COLOR=#0066CC`, `BUS_FILL=#FFFFB0`, an
unknown-region green hatch (`#02D98A`), a high-Z orange hatch (`#FF8C00`), and
constants `LANE_HEIGHT=28`, `TICK_WIDTH=40`, `STROKE_WIDTH=1.5`.

### 2.4 What the dataset generator varies, and what it does not

`GenerationSpec` (in `sva_toolkit.timing.generate.model`) parameterizes
*content* across topology, flavor, idioms, bound kinds, naming style, cuts,
distractor lanes, tick budget, clock edge, predicate bias, region bias, and
rendering mode (`concrete` / `symbolic` / `mixed`). Across thousands of
records, lane count, lane ordering, signal names, sample values, anchor
positions, response/hold spans, and summary lines therefore vary widely.

`GenerationSpec` does **not** carry any rendering parameter. There is no
sampling at the visual level. Every accepted record passes through the same
`render_diagram_svg → render_wavedrom_svg` codepath with all of the constants
in 2.3 frozen, and then through one cairosvg rasterization with no
augmentation. A user inspecting any two PNGs in the dataset cannot distinguish
them by font, color, line weight, anchor pill style, arrow style, lane label
color, footer style, card chrome, or rasterization quality — only by content.

---

## 3. The Problem

The dataset is intended to train a VLM that reads a waveform image from a
real specification document and emits the canonical timing DSL. The current
dataset cannot meet that goal, for the following structural reasons.

### 3.1 The training distribution is a single point in style space

Every image in the dataset is rendered by one pipeline with frozen visual
constants. There is exactly one font, one palette, one line weight, one
overlay style, one card chrome, and one rasterization profile. As a training
distribution this is a single point. A model fit to it has no incentive to
learn invariants — it learns a renderer-specific mapping from pixels to DSL.

### 3.2 The deployment distribution is open-ended and unknown

Real specification documents contain timing diagrams produced by, at least:
WaveDrom, EDA-tool screenshots, hand-drawn datasheet figures from arbitrary
decades, PDF rasterizations at varied DPIs and antialiasing levels,
black-and-white scans, photocopies, photographs of whiteboards, charts pasted
into Word/PowerPoint, ASCII waveforms in plain-text RFCs, paper-figure
diagrams without grids, and so on. There is no finite, enumerable set of
"styles" that bounds this distribution. No matter how many style profiles a
dataset enumerates, the long tail is unbounded and the tail is where real
documents live.

This means the gap between the training distribution (one point) and the
deployment distribution (unbounded long tail) is essentially the entire
problem. Any visual signal the model relies on that is not also present in the
deployment distribution becomes a shortcut feature that fails in the wild.

### 3.3 Overlays leak the target through rendered text

The overlay layer draws anchor pills, arrow labels, hold highlights, and a
"RULES" footer that contains DSL-derived text on the image:

- Anchor pills carry anchor names verbatim (`req_rise`, `handshake`,
  `aw_hs`). The DSL declares the same names. The model can learn a transcription
  shortcut: read the pill text, copy it into `anchor` declarations.
- Response arrows carry the literal bound text (`[1:4]`, `[0:MAX_LAT]`,
  `[1:$]`). The DSL contains the same tokens.
- The `RULES` footer contains plain-English summaries that are direct
  paraphrases of `show` constraints and property bodies.
- Hold highlights are drawn precisely on the lane regions named by
  `from..until` constraints.

The image therefore *labels itself* with target-side tokens that real
specification figures generally do not contain. A model that learns to read
these overlays learns the wrong policy: it learns to transcribe text that the
synthetic renderer drew, not to recover semantics from waveform geometry. At
deployment, that text is missing, and the policy collapses.

This is a more severe failure mode than visual monoculture. Visual monoculture
makes the model brittle to style. Overlay leakage makes the model **dependent
on signals that do not exist in the target domain**, regardless of style
augmentation.

### 3.4 The architecture binds visual diversity to one upstream renderer

The renderer reverse-engineers `wavedrom`'s internal geometry by stashing
private layout state on the SVG root and reading it back to position
overlays. The overlay layer, the layout extraction, and the WaveDrom call are
coupled. Producing a non-WaveDrom rendering — for example, an EDA-tool-style
or scanned-datasheet-style image — would require either reimplementing the
overlay placement against a different geometry source or removing the overlays
entirely. There is no internal abstraction for "renderer" that takes a
`ScenarioDocument` and a visual configuration and returns an image; the public
function `render_diagram_svg` is hardwired to the WaveDrom path.

A dataset generator that wants to vary rendering must therefore either fork
the rendering module or live with the constraint that all variation happens
inside one drawing implementation.

### 3.5 PNGs are clean vector rasterizations

PNG output is `cairosvg.svg2png(svg_text)` with no augmentation. There is no
resolution variance, no JPEG compression, no blur, no contrast shift, no
rotation, no cropping, no scan-style degradation. Even if the SVG layer were
diversified, the raster path adds no further variation.

### 3.6 The supervised target presumes overlay-grounded recovery

The dataset's correctness story relies on `recoverability = "visual"`: the
canonical DSL must be inferable from the image. Today this is true only
because the overlays explicitly draw most of the symbolic content the DSL
contains. Strip the overlays — which is what a real datasheet figure looks
like — and many features the DSL records (anchor names, exact bound numbers,
named windows, rule descriptions) are no longer present in the image. The
current dataset implicitly defines "visually recoverable" as "recoverable from
the overlay band," not "recoverable from the waveform itself." Shrinking the
overlay band would invalidate many existing target labels; keeping it
preserves the leakage in 3.3.

### 3.7 Coverage controls do not address the visual axis

`CoverageTracker` buckets cover topology, idiom, tick count, lane count,
lane kind, anchor count, window count, bound kind, predicate, region, cut,
rendering mode, and naming. None of these are visual. The generator can
balance content composition but has no concept of visual coverage — there is
nothing to balance, because there is one visual mode.

---

## 4. Constraints that any redesign has to respect

These are not solutions; they are properties the current pipeline depends on
that a redesign must preserve, otherwise the rest of the toolkit breaks.

- The training target must remain canonical DSL produced by
  `emit_timing_dsl(parse_diagram(emit_timing_dsl(document)))`. This is what
  `sva timing` consumers consume; deviating from it makes the dataset useless
  for the surrounding tooling.
- The DSL must round-trip and validate. The generator already enforces this
  through `parse_diagram` + `validate_diagram` rejection.
- Records must remain seedable and reproducible. `GenerationRng` derives child
  RNGs by `sha256(seed:label)`; any new sampling axis needs the same property.
- The CLI surface (`sva timing generate-dataset`, `sva timing render`) is part
  of the user contract.
- Coverage and rejection metrics in `summary.json` are consumed by downstream
  validation (`sva timing validate-dataset`).

---

## 5. Question for review

Given the structural problems above — single-point training distribution,
open-ended deployment distribution, overlay-driven target leakage, renderer
coupling, clean rasterization, and the visually blind coverage tracker — what
is the right way to redesign the dataset and rendering pipeline so that a VLM
trained on it can actually recover canonical timing DSL from waveform images
in arbitrary real specification documents?
