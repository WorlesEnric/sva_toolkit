#!/usr/bin/env python3
"""Render one or more `.td` example files across every render2 profile.

This script is intended for human visual inspection of the new render2
pipeline introduced in `refactor/render-stochastic`. It renders each input
diagram through every available profile and writes the SVG (and, where
possible, PNG) outputs to a gallery directory along with a small
`gallery.json` manifest summarizing what was produced.

Usage:

    python scripts/render_profile_gallery.py \\
        examples/td/01_simple_handshake.td \\
        examples/td/06_bus_protocol.td \\
        --out-dir /tmp/sva_gallery \\
        --seeds 1,7,42

By default the script renders every locally registered renderer (native_svg,
wavedrom, ascii, plus undulate/tikz/plantuml/gtkwave when their dependencies
are installed) and skips profiles whose renderer is not available, recording
that fact in the manifest.

When `--debug-current-only` is passed the script renders only the legacy
`debug-current` profile, useful for visually diff-ing legacy vs. clean
output side by side.

The script is read-only against the package: it does not modify any cached
files or coverage trackers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render2.adapters.registry_bootstrap import (
    bootstrap_external_renderers,
)
from sva_toolkit.timing.render2.profiles import ALL_PROFILES, RenderProfile

try:
    from sva_toolkit.cli.main import _render_timing_with_profile  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"render2 CLI helper unavailable: {exc}") from exc


def _profiles_for(args: argparse.Namespace) -> list[RenderProfile]:
    if args.debug_current_only:
        return [p for p in ALL_PROFILES if p.id == "debug-current"]
    if args.profiles:
        wanted = {x.strip() for x in args.profiles.split(",") if x.strip()}
        return [p for p in ALL_PROFILES if p.id in wanted]
    return list(ALL_PROFILES)


def _seeds(args: argparse.Namespace) -> list[int]:
    raw = args.seeds.split(",") if args.seeds else ["0"]
    return [int(x.strip()) for x in raw if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Input .td files.")
    parser.add_argument("--out-dir", required=True, help="Where to write the gallery.")
    parser.add_argument("--profiles", default="", help="CSV of profile ids to include (default: all).")
    parser.add_argument("--seeds", default="0", help="CSV of seeds to render with each profile.")
    parser.add_argument("--debug-current-only", action="store_true", help="Only render the legacy debug-current profile.")
    parser.add_argument("--format", choices=("svg", "png"), default="svg", help="Output image format.")
    parser.add_argument(
        "--audit-strict",
        action="store_true",
        help="Fail (and skip) records whose render2 audits report leakage or visibility problems.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_status = bootstrap_external_renderers()
    profiles = _profiles_for(args)
    seeds = _seeds(args)

    manifest: list[dict[str, object]] = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            raise SystemExit(f"input not found: {input_path}")
        diagram = parse_diagram(input_path.read_text(encoding="utf-8"))

        for profile in profiles:
            renderer_status = bootstrap_status.get(profile.renderer_id, "registered")
            for seed in seeds:
                stem = f"{input_path.stem}__{profile.id}__seed{seed}"
                out_path = out_dir / f"{stem}.{args.format}"
                record: dict[str, object] = {
                    "input": str(input_path),
                    "profile": profile.id,
                    "renderer_id": profile.renderer_id,
                    "renderer_status": renderer_status,
                    "annotation_policy": profile.annotation_policy.value,
                    "seed": seed,
                    "output": str(out_path),
                }
                if profile.renderer_id != "native_svg" and profile.renderer_id != "wavedrom" and not renderer_status.startswith("registered"):
                    record["status"] = f"skipped: {renderer_status}"
                    manifest.append(record)
                    continue

                try:
                    rendered = _render_timing_with_profile(
                        diagram,
                        render_profile=profile.id,
                        seed=seed,
                        audit_strict=args.audit_strict,
                    )
                except Exception as exc:
                    record["status"] = f"error: {exc.__class__.__name__}: {exc}"
                    manifest.append(record)
                    continue

                if args.format == "png":
                    image_bytes = rendered.get("image_bytes")
                    if not image_bytes:
                        record["status"] = "no_png_bytes"
                        manifest.append(record)
                        continue
                    out_path.write_bytes(image_bytes)
                else:
                    svg_text = rendered.get("svg_text")
                    if not isinstance(svg_text, str) or not svg_text:
                        record["status"] = "no_svg_output"
                        manifest.append(record)
                        continue
                    out_path.write_text(svg_text, encoding="utf-8")

                audit_status = rendered.get("audit_status")
                if audit_status:
                    record["audit_status"] = audit_status
                record["status"] = "ok"
                record["bytes"] = out_path.stat().st_size
                manifest.append(record)

    manifest_path = out_dir / "gallery.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ok = sum(1 for r in manifest if r.get("status") == "ok")
    skipped = sum(1 for r in manifest if str(r.get("status", "")).startswith("skipped"))
    errored = sum(1 for r in manifest if str(r.get("status", "")).startswith("error"))
    print(f"wrote {ok} files, skipped {skipped}, errored {errored}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
