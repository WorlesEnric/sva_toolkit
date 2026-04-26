# Problems

## Timing Diagram Dataset Visual Diversity

The current timing diagram dataset generator produces semantically varied
Image-DSL pairs, but the rendered figures all come from one visual style. This
can become a training problem for a VLM whose target use case is recovering DSL
code from waveform images in real specification documents.

A model trained only on one renderer may overfit to that renderer's visual
grammar: font choices, line widths, lane spacing, grid layout, annotation
placement, bus rendering, arrow style, colors, margins, and clean SVG/PNG
rasterization. Real documents may contain WaveDrom-style diagrams, black and
white datasheet figures, screenshots from EDA tools, compact PDF timing charts,
low-resolution scans, cropped captions, inconsistent fonts, or degraded
antialiasing. The current dataset therefore risks testing semantic recovery
inside a narrow visual distribution rather than robust diagram understanding.

Suggested mitigations:

- Add renderer style profiles for the same canonical DSL target, such as
  `spec_bw`, `wavedrom_like`, `dense_pdf`, `eda_screenshot`, and
  `annotated_protocol`.
- Generate multiple image variants per DSL record while keeping one canonical
  target DSL, so the model learns visual invariance.
- Vary fonts, colors, line widths, grid visibility, lane spacing, tick labels,
  bus value rendering, arrow/window labels, margins, titles, captions, and
  background treatment.
- Add safe raster augmentations that mimic documents: grayscale, lower contrast,
  slight blur, JPEG compression, scale changes, margin/crop variation, and
  PDF-like antialiasing.
- Prefer at least one alternate renderer, not only style changes in the current
  SVG renderer, so visual diversity is not tied to a single drawing
  implementation.
- Build a small real-spec evaluation set from manually cropped waveform
  diagrams to measure the synthetic-to-real gap directly.

Recommended next step: extend `timing generate-dataset` with style selection
options such as `--style-profile` or `--styles`, write `style_id` into each
record, and support generating several rendered images for each canonical DSL.
