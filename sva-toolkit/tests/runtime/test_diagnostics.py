from __future__ import annotations

import importlib
import logging
import threading

import pytest

import sva_toolkit.runtime.diagnostics as diagnostics_module


def test_record_increments_snapshot_and_snapshot_is_immutable() -> None:
    diagnostics = diagnostics_module.Diagnostics()
    diagnostics.record("cache_miss")

    snapshot = diagnostics.snapshot()

    assert snapshot["cache_miss"] == 1
    with pytest.raises(TypeError):
        snapshot["cache_miss"] = 2  # type: ignore[index]

    diagnostics.record("cache_miss")

    assert snapshot["cache_miss"] == 1
    assert diagnostics.snapshot()["cache_miss"] == 2


def test_record_rejects_unknown_kinds() -> None:
    diagnostics = diagnostics_module.Diagnostics()

    with pytest.raises(ValueError, match="Unsupported diagnostic kind"):
        diagnostics.record("opaque_sequence")


def test_render_summary_is_deterministic_and_alphabetical() -> None:
    diagnostics = diagnostics_module.Diagnostics()
    diagnostics.record("retry_exhausted")
    diagnostics.record("cache_miss")
    diagnostics.record("retry_exhausted")

    assert diagnostics.render_summary() == "Diagnostics summary: cache_miss=1, retry_exhausted=2"


def test_configure_cli_logging_is_idempotent_and_uses_stable_format(capsys) -> None:
    module = importlib.reload(diagnostics_module)

    logger = module.configure_cli_logging(0)
    first_handlers = tuple(logger.handlers)
    same_logger = module.configure_cli_logging(2)

    assert same_logger is logger
    assert tuple(logger.handlers) == first_handlers
    assert logger.level == logging.DEBUG

    logger.warning("warning output")

    captured = capsys.readouterr()
    assert captured.err == "WARNING warning output\n"


def test_record_is_thread_safe() -> None:
    diagnostics = diagnostics_module.Diagnostics()
    barrier = threading.Barrier(8)
    increments_per_thread = 250
    failures: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(increments_per_thread):
                diagnostics.record("lossy_extraction")
        except BaseException as exc:  # pragma: no cover - defensive capture for thread failures
            failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert diagnostics.snapshot()["lossy_extraction"] == 8 * increments_per_thread
