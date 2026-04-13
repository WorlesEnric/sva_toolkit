"""Test the td -> sva -> gen_td -> gen_svg roundtrip chain.

For every .td file in examples/td/:
  1. Parse TD and emit SVA          -> examples/sva/{name}.sv
  2. Extract TD back from SVA       -> examples/gen_td/{name}.td
  3. Render extracted TD to SVG     -> examples/gen_svg/{name}.svg

As described in docs/remediation_witness_synthesis.md §5.3,
the canonical regression direction is sv -> td -> sv, but this script
exercises the full rendering chain to verify no step crashes and to
produce visual artifacts for manual inspection.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sva_toolkit.formal.parse import split_property_texts
from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
from sva_toolkit.timing.bridge.from_sva import bundle_sva_scenarios, extract_sva_scenario
from sva_toolkit.timing.bridge.to_dsl import emit_timing_dsl
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.projection.wavedrom_view import build_wavedrom_view
from sva_toolkit.timing.render.wavedrom import render_wavedrom_svg


def main() -> int:
    examples_dir = Path(__file__).resolve().parent
    td_dir = examples_dir / "td"
    sva_dir = examples_dir / "sva"
    gen_td_dir = examples_dir / "gen_td"
    gen_svg_dir = examples_dir / "gen_svg"

    for d in (sva_dir, gen_td_dir, gen_svg_dir):
        d.mkdir(parents=True, exist_ok=True)

    td_files = sorted(td_dir.glob("*.td"))
    if not td_files:
        print(f"No .td files found in {td_dir}")
        return 1

    passed = 0
    failed = 0
    errors: list[tuple[str, str, str]] = []

    for td_path in td_files:
        stem = td_path.stem
        sva_path = sva_dir / f"{stem}.sv"
        gen_td_path = gen_td_dir / f"{stem}.td"
        gen_svg_path = gen_svg_dir / f"{stem}.svg"

        # -- Step 1: td -> sva ------------------------------------------------
        try:
            td_text = td_path.read_text(encoding="utf-8")
            document = parse_diagram(td_text)
            sva_text = emit_parameterized_sva(document, allow_lossy=True)
            sva_path.write_text(f"{sva_text}\n", encoding="utf-8")
            print(f"  [emit]    {stem}.td -> {stem}.sv")
        except Exception:
            msg = traceback.format_exc()
            errors.append((stem, "td->sva", msg))
            print(f"  [FAIL]    {stem}.td -> sva: emit failed")
            failed += 1
            continue

        # -- Step 2: sva -> gen_td --------------------------------------------
        try:
            property_texts = split_property_texts(sva_path.read_text(encoding="utf-8"))
            documents = tuple(
                extract_sva_scenario(pt) for pt in property_texts
            )
            if len(documents) > 1:
                documents = bundle_sva_scenarios(
                    documents,
                    overlap_threshold=0.0,
                    max_signals=max(10, sum(len(d.signals) for d in documents)),
                    max_properties=max(5, sum(len(d.properties) for d in documents)),
                    max_chains=max(3, len(documents)),
                )
            gen_td_text = "\n\n".join(emit_timing_dsl(d) for d in documents)
            gen_td_path.write_text(gen_td_text, encoding="utf-8")
            print(f"  [extract] {stem}.sv -> {stem}.td")
        except Exception:
            msg = traceback.format_exc()
            errors.append((stem, "sva->gen_td", msg))
            print(f"  [FAIL]    {stem}.sv -> gen_td: extract failed")
            failed += 1
            continue

        # -- Step 3: gen_td -> gen_svg ----------------------------------------
        try:
            gen_doc = parse_diagram(gen_td_path.read_text(encoding="utf-8"))
            view = build_wavedrom_view(gen_doc)
            svg_content = render_wavedrom_svg(view)
            gen_svg_path.write_text(svg_content, encoding="utf-8")
            print(f"  [render]  {stem}.td -> {stem}.svg")
        except Exception:
            msg = traceback.format_exc()
            errors.append((stem, "gen_td->gen_svg", msg))
            print(f"  [FAIL]    {stem}.td -> gen_svg: render failed")
            failed += 1
            continue

        passed += 1
        print(f"  [ok]      {stem}")

    # -- Summary --------------------------------------------------------------
    total = passed + failed
    print()
    print(f"Roundtrip chain: {passed}/{total} passed, {failed} failed")

    if errors:
        print("\n--- Error details ---")
        for stem, stage, msg in errors:
            print(f"\n[{stem}] stage={stage}")
            print(msg)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
