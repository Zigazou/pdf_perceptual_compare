"""Unit tests for CLI output and orchestration."""

from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock

import pytest
from numpy import uint8, zeros

from pdf_perceptual_compare.ansi import Ansi
from pdf_perceptual_compare.cli import (
    compare_rendered_pages,
    render_page_result
)
from pdf_perceptual_compare.comparator import (
    ComparisonOptions,
    PageResult,
    compare_page
)
from pdf_perceptual_compare.verdict import Verdict


def test_render_page_result_colors_failures_only_for_terminals() -> None:
    """Verify that failure colors are only applied in terminal output."""
    failure = PageResult(1, False, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.1, Verdict.FAIL)

    assert render_page_result(failure, use_color=True).startswith(Ansi.RED)
    assert render_page_result(failure, use_color=False).startswith("   1  FAIL")


def test_compare_rendered_pages_runs_comparisons_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that compare_rendered_pages runs comparisons concurrently."""
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    maximum_active = 0

    def fake_compare_page(a, b, page, options, failure_dir):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait(timeout=1)
        except BrokenBarrierError:
            pass
        finally:
            with lock:
                active -= 1
        return compare_page(a, b, page, options, failure_dir)

    image = zeros((16, 16, 3), dtype=uint8)

    monkeypatch.setattr(
        "pdf_perceptual_compare.cli.load_rgb",
        lambda path: image
    )

    monkeypatch.setattr(
        "pdf_perceptual_compare.cli.compare_page",
        fake_compare_page
    )

    results = list(
        compare_rendered_pages(
            [
                (tmp_path / "a-0001.png", tmp_path / "b-0001.png"),
                (tmp_path / "a-0002.png", tmp_path / "b-0002.png"),
            ],
            ComparisonOptions(),
            failure_dir=None,
            keep_rendered=None,
            jobs=2,
        )
    )

    assert maximum_active == 2
    assert {result.page for result in results} == {1, 2}
