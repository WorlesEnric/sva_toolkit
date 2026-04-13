#!/usr/bin/env python3
"""Extract candidate signal names from JSONL datasets for manual preset curation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
LEADING_WIDTH_RE = re.compile(r"^\[[^\]]+\]\s*(.+)$")
TRAILING_INDEX_RE = re.compile(r"\[[^\]]+\]$")


def normalize_signal_name(raw_name: str) -> str | None:
    name = raw_name.strip()
    if not name:
        return None

    width_match = LEADING_WIDTH_RE.match(name)
    if width_match:
        name = width_match.group(1).strip()

    while TRAILING_INDEX_RE.search(name):
        name = TRAILING_INDEX_RE.sub("", name).strip()

    if not name or not IDENTIFIER_RE.match(name):
        return None

    return name


def iter_signal_names(jsonl_path: Path) -> Iterable[str]:
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            signals = row.get("sva_signals")
            if not isinstance(signals, list):
                continue

            for raw_signal in signals:
                if not isinstance(raw_signal, str):
                    continue
                normalized = normalize_signal_name(raw_signal)
                if normalized:
                    yield normalized


def write_signal_pool_module(
    output_path: Path,
    source_path: Path,
    signals: list[str],
    min_count: int,
) -> None:
    header = (
        '"""Auto-generated candidate signal pool for SVA generation."""\n\n'
        f'SOURCE_JSONL = "{source_path.as_posix()}"\n'
        f"MIN_COUNT = {min_count}\n\n"
        "GENERATED_SIGNALS = [\n"
    )
    body = "".join(f'    "{signal}",\n' for signal in signals)
    footer = "]\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + body + footer, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and preprocess signal names from a JSONL dataset."
    )
    parser.add_argument(
        "--input",
        default="data/src_case_sva_train_clear.jsonl",
        help="Path to source JSONL with sva_signals field.",
    )
    parser.add_argument(
        "--output",
        default="out/signal_presets_generated.py",
        help="Output Python module containing GENERATED_SIGNALS.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Keep only signals appearing at least this many times.",
    )
    args = parser.parse_args()

    if args.min_count < 1:
        raise ValueError("--min-count must be >= 1")

    input_path = Path(args.input)
    output_path = Path(args.output)

    counter = Counter(iter_signal_names(input_path))
    selected = [
        signal
        for signal, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count >= args.min_count
    ]

    write_signal_pool_module(
        output_path=output_path,
        source_path=input_path,
        signals=selected,
        min_count=args.min_count,
    )

    print(f"Scanned unique signals: {len(counter)}")
    print(f"Selected signals: {len(selected)} (min_count={args.min_count})")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
