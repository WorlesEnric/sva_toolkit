from __future__ import annotations

from pathlib import Path
import threading

import pytest

from sva_toolkit.runtime import atomic_io


def test_atomic_write_text_writes_full_content(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    atomic_io.atomic_write_text(target, "alpha\nbeta\n")

    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_write_text_failed_publish_leaves_target_unchanged(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("original\n", encoding="utf-8")

    def fail_replace(src: str | Path, dst: str | Path) -> None:
        raise OSError(f"replace failed for {src} -> {dst}")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_io.atomic_write_text(target, "updated\n")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_write_text_missing_parent_raises_file_not_found(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "out.txt"

    with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
        atomic_io.atomic_write_text(target, "payload")


def test_atomic_write_text_concurrent_writers_produce_single_coherent_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    barrier = threading.Barrier(2)
    contents = ("A" * 2048, "B" * 2048)
    failures: list[BaseException] = []

    def writer(content: str) -> None:
        try:
            barrier.wait()
            atomic_io.atomic_write_text(target, content)
        except BaseException as exc:  # pragma: no cover - defensive capture for thread failures
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(content,)) for content in contents]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert target.read_text(encoding="utf-8") in set(contents)
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_write_text_repeated_same_content_is_byte_stable(tmp_path: Path) -> None:
    target = tmp_path / "stable.txt"

    atomic_io.atomic_write_text(target, "stable-content\n")
    first = target.read_bytes()
    atomic_io.atomic_write_text(target, "stable-content\n")
    second = target.read_bytes()

    assert first == second
    assert [path.name for path in tmp_path.iterdir()] == ["stable.txt"]


def test_atomic_write_json_delegates_to_text(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
        assert encoding == "utf-8"
        calls.append((Path(path), content))

    monkeypatch.setattr(atomic_io, "atomic_write_text", fake_atomic_write_text)

    atomic_io.atomic_write_json(tmp_path / "data.json", {"b": 2, "a": 1})

    assert calls == [
        (
            tmp_path / "data.json",
            '{\n  "a": 1,\n  "b": 2\n}\n',
        )
    ]


def test_atomic_write_jsonl_delegates_to_text(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> None:
        assert encoding == "utf-8"
        calls.append((Path(path), content))

    monkeypatch.setattr(atomic_io, "atomic_write_text", fake_atomic_write_text)

    atomic_io.atomic_write_jsonl(tmp_path / "data.jsonl", [{"b": 2, "a": 1}, {"c": 3}])

    assert calls == [
        (
            tmp_path / "data.jsonl",
            '{"a": 1, "b": 2}\n{"c": 3}\n',
        )
    ]
