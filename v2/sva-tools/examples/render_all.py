"""Render all example timing diagrams to SVG and emit synthesized SVA."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
from sva_toolkit.timing.frontend.parser import parse_diagram
from sva_toolkit.timing.render.svg import render_diagram_svg


def main() -> int:
    examples_dir = Path(__file__).resolve().parent
    td_dir = examples_dir / "td"
    svg_dir = examples_dir / "svg"
    sva_dir = examples_dir / "sva"
    svg_dir.mkdir(parents=True, exist_ok=True)
    sva_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for diagram_path in sorted(td_dir.glob("*.td")):
        document = parse_diagram(diagram_path.read_text(encoding="utf-8"))

        svg_text = render_diagram_svg(document)
        svg_path = svg_dir / f"{diagram_path.stem}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")

        sva_text = emit_parameterized_sva(document)
        sva_path = sva_dir / f"{diagram_path.stem}.sv"
        sva_path.write_text(f"{sva_text}\n", encoding="utf-8")

        print(
            f"[ok] {diagram_path.relative_to(examples_dir)} -> "
            f"{svg_path.relative_to(examples_dir)}, {sva_path.relative_to(examples_dir)}"
        )
        print(f"[sva] {diagram_path.stem}")
        print(sva_text)
        print()
        rendered += 1

    print(f"Rendered {rendered} diagrams to {svg_dir}")
    print(f"Emitted {rendered} SVA files to {sva_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
