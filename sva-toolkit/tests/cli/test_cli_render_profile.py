from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sva_toolkit.cli.main import main


EXAMPLE = Path("examples/td/01_simple_handshake.td")


def test_timing_render_default_uses_legacy_debug_svg() -> None:
    result = CliRunner().invoke(main, ["timing", "render", str(EXAMPLE)], prog_name="sva")

    assert result.exit_code == 0, result.output
    assert "req" in result.output
    assert "ack" in result.output
    assert "RULES" in result.output


def test_timing_render_profile_routes_through_render2(tmp_path: Path) -> None:
    output = tmp_path / "clean.svg"
    result = CliRunner().invoke(
        main,
        ["timing", "render", str(EXAMPLE), "--render-profile", "clean-wavedrom", "-o", str(output)],
        prog_name="sva",
    )

    assert result.exit_code == 0, result.output
    svg = output.read_text(encoding="utf-8")
    assert "req" in svg
    assert "ack" in svg
    assert "RULES" not in svg
