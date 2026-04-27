#!/usr/bin/env python3
"""Build the standard multi-split timing Image-DSL dataset.

This is a thin orchestrator over `sva timing generate-dataset` that produces
the four canonical splits described in `docs/timing-dataset-generation.md`:

    train / val_seen_style / val_unseen_style / test_synthetic_ood

It applies the right `--render-profile-set`, `--style-holdout`, and
`--audit-strict` settings for each split and writes the four output
directories under one root.

Typical usage:

    python scripts/generate_timing_dataset.py \\
        --root /data/timing-v2 \\
        --train-count 50000 \\
        --val-seen-count 2000 \\
        --val-unseen-count 1000 \\
        --test-ood-count 2000 \\
        --seed 1 \\
        --format both

Re-running the same command with the same seed reproduces the dataset
deterministically (per-record seeds derive from this master seed via the
existing `GenerationRng` contract).

Notes:

- `val_unseen_style` and `test_synthetic_ood` profiles depend on external
  renderers (PlantUML, GTKWave, tikz-timing). If those tools are not
  installed locally the corresponding split sizes will fall short or fail;
  use `--skip-ood` to skip those splits when running on a machine without
  the required tools.
- All splits enforce `--audit-strict` by default. Disable with
  `--no-audit-strict` only for debugging — accepted records can leak target
  tokens or fail visibility checks.
- Set `--format png` once a working raster path (cairosvg with libcairo,
  resvg-py, or wand) is available. With `svg` the script writes vector
  files only, which is fastest and renders perfectly in browsers.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitDef:
    name: str
    count_attr: str
    profile_set: str
    style_holdout: tuple[str, ...]
    audit_strict: bool
    require_external_tools: bool


SPLITS: tuple[SplitDef, ...] = (
    SplitDef(
        name="train",
        count_attr="train_count",
        profile_set="train_v2",
        style_holdout=("plantuml-ood", "gtkwave-ood"),
        audit_strict=True,
        require_external_tools=False,
    ),
    SplitDef(
        name="val_seen_style",
        count_attr="val_seen_count",
        profile_set="val_seen_style",
        style_holdout=("plantuml-ood", "gtkwave-ood"),
        audit_strict=True,
        require_external_tools=False,
    ),
    SplitDef(
        name="val_unseen_style",
        count_attr="val_unseen_count",
        profile_set="val_unseen_style",
        style_holdout=(),
        audit_strict=True,
        require_external_tools=True,
    ),
    SplitDef(
        name="test_synthetic_ood",
        count_attr="test_ood_count",
        profile_set="test_ood",
        style_holdout=(),
        audit_strict=True,
        require_external_tools=True,
    ),
)


def _build_command(
    split: SplitDef,
    *,
    out_dir: Path,
    count: int,
    seed: int,
    format_: str,
    audit_strict: bool,
    extra_flags: list[str],
) -> list[str]:
    cmd = [
        "sva", "timing", "generate-dataset",
        "--out", str(out_dir),
        "--count", str(count),
        "--seed", str(seed),
        "--split", split.name,
        "--render-profile-set", split.profile_set,
        "--target-policy", "visual",
        "--emit-render-specs",
        "--format", format_,
    ]
    if audit_strict and split.audit_strict:
        cmd.append("--audit-strict")
    else:
        cmd.append("--no-audit-strict")
    if split.style_holdout:
        cmd.extend(["--style-holdout", ",".join(split.style_holdout)])
    cmd.extend(extra_flags)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Root output directory; one subdir per split.")
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--val-seen-count", type=int, default=500)
    parser.add_argument("--val-unseen-count", type=int, default=500)
    parser.add_argument("--test-ood-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1, help="Master seed; per-split seeds derive from this.")
    parser.add_argument("--format", choices=("svg", "png", "both", "none"), default="svg")
    parser.add_argument("--audit-strict/--no-audit-strict", dest="audit_strict", action="store_true", default=True)
    parser.add_argument("--no-audit-strict", dest="audit_strict", action="store_false")
    parser.add_argument("--skip-ood", action="store_true", help="Skip val_unseen_style + test_synthetic_ood.")
    parser.add_argument("--only", default="", help="CSV of split names to run (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--extra", default="", help="Extra flags appended to every sva command (verbatim).")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    requested = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None

    extra_flags = shlex.split(args.extra) if args.extra else []
    overall_ok = True

    for index, split in enumerate(SPLITS):
        if requested is not None and split.name not in requested:
            continue
        if args.skip_ood and split.require_external_tools:
            print(f"-- skip {split.name}: requires external tools (use without --skip-ood to enable)")
            continue

        count = getattr(args, split.count_attr)
        if count <= 0:
            print(f"-- skip {split.name}: count={count}")
            continue

        out_dir = root / split.name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_command(
            split,
            out_dir=out_dir,
            count=count,
            seed=args.seed + index * 1009,  # decorrelate split seeds
            format_=args.format,
            audit_strict=args.audit_strict,
            extra_flags=extra_flags,
        )
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"==> {split.name}: {printable}")
        if args.dry_run:
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"!! {split.name} failed (exit code {proc.returncode})", file=sys.stderr)
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
