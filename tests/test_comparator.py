import threading

import numpy as np
import pytest

pytest.importorskip("skimage")

from pdf_perceptual_compare.cli import compare_rendered_pages, render_page_result
from pdf_perceptual_compare.comparator import (
    ComparisonOptions,
    PageResult,
    compare_page,
    crop_for_shift,
)


def test_render_page_result_colors_failures_only_for_terminals() -> None:
    failure = PageResult(1, False, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.1, "FAIL")

    assert render_page_result(failure, use_color=True).startswith("\033[31m")
    assert render_page_result(failure, use_color=False).startswith("   1  FAIL")
from pdf_perceptual_compare.pdf import render_page_pairs


def test_compare_page_accepts_identical_images() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = compare_page(image, image.copy(), page=1, options=ComparisonOptions())

    assert result.verdict == "PASS"
    assert result.identical is True
    assert result.ssim == 1.0


def test_compare_page_rejects_different_dimensions() -> None:
    result = compare_page(
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((15, 16, 3), dtype=np.uint8),
        page=1,
        options=ComparisonOptions(),
    )

    assert result.verdict == "FAIL(size)"


def test_crop_for_shift_returns_matching_overlap() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    shifted_a, shifted_b = crop_for_shift(image, image, dx=2, dy=-1)

    assert shifted_a.shape == shifted_b.shape == (9, 8, 3)


def test_render_page_pairs_returns_paths_in_page_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_render_page(pdf, page, dpi, output_base):
        return output_base.with_suffix(".png")

    monkeypatch.setattr("pdf_perceptual_compare.pdf.render_page", fake_render_page)

    rendered = render_page_pairs(
        tmp_path / "original.pdf",
        tmp_path / "candidate.pdf",
        pages=2,
        dpi=150,
        output_directory=tmp_path,
        jobs=2,
    )

    assert rendered == [
        (tmp_path / "a-0001.png", tmp_path / "b-0001.png"),
        (tmp_path / "a-0002.png", tmp_path / "b-0002.png"),
    ]


def test_render_page_pairs_runs_rendering_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_render_page(pdf, page, dpi, output_base):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        finally:
            with lock:
                active -= 1
        return output_base.with_suffix(".png")

    monkeypatch.setattr("pdf_perceptual_compare.pdf.render_page", fake_render_page)

    render_page_pairs(
        tmp_path / "original.pdf",
        tmp_path / "candidate.pdf",
        pages=1,
        dpi=150,
        output_directory=tmp_path,
        jobs=2,
    )

    assert maximum_active == 2


def test_compare_rendered_pages_runs_comparisons_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_compare_page(a, b, page, options, failure_dir):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait(timeout=1)
        except threading.BrokenBarrierError:
            pass
        finally:
            with lock:
                active -= 1
        return compare_page(a, b, page, options, failure_dir)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    monkeypatch.setattr("pdf_perceptual_compare.cli.load_rgb", lambda path: image)
    monkeypatch.setattr("pdf_perceptual_compare.cli.compare_page", fake_compare_page)

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
