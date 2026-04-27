# Helper scripts

These scripts complement the `sva` CLI for repeatable timing-diagram dataset
workflows and visual inspection of the render2 pipeline.

## `render_profile_gallery.py`

Render one or more `.td` files across every render2 profile and write the
results into a gallery directory. Use this to eyeball the visual diversity
the new pipeline produces and to compare profiles side by side.

```bash
python scripts/render_profile_gallery.py \
    examples/td/01_simple_handshake.td \
    examples/td/06_bus_protocol.td \
    --out-dir /tmp/sva_gallery \
    --seeds 1,7,42 \
    --format svg
```

Useful flags:

- `--profiles native-random,clean-wavedrom,...` — restrict to a subset.
- `--debug-current-only` — render only the legacy profile (the figure
  with `RULES` footer) for visual diff against clean profiles.
- `--audit-strict` — skip records whose render2 audits fail (leakage,
  contrast, occlusion, etc.). Useful to confirm a profile produces clean
  records on your machine.
- `--format png` — only meaningful when a working SVG-to-PNG rasterizer is
  installed (cairosvg + libcairo, resvg-py, or wand). Without one the
  composer falls back to a blank synthetic raster.

The script writes a `gallery.json` manifest summarizing each render
attempt: profile, renderer status, audit status (if available), output
path, byte size, or skip/error reason.

## `generate_timing_dataset.py`

Build the canonical multi-split dataset (`train` / `val_seen_style` /
`val_unseen_style` / `test_synthetic_ood`) defined in
`docs/timing-dataset-generation.md`. Thin wrapper over
`sva timing generate-dataset` that applies the right `--render-profile-set`,
holdouts, and `--audit-strict` flags per split.

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

Useful flags:

- `--skip-ood` — skip `val_unseen_style` and `test_synthetic_ood` (the
  splits that depend on external renderers like PlantUML, GTKWave,
  tikz-timing); use on machines where those tools are not installed.
- `--only train,val_seen_style` — restrict to a subset of splits.
- `--dry-run` — print the underlying `sva` invocations without executing.
- `--extra "--coverage-target 5"` — append extra flags verbatim to every
  split.
- `--no-audit-strict` — turn off audit-strict mode globally (debugging
  only — not recommended for shipping datasets).

Each split runs with a deterministic seed derived from `--seed`, so
re-running the script with the same arguments reproduces the dataset.

## `expand_signal_pool.py`

Pre-existing helper for curating signal-name presets out of an existing
`records.jsonl`. Unrelated to the render2 refactor; see the file
docstring for usage.
