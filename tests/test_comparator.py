"""Tests for the pdf_perceptual_compare comparator module."""
from pdf_perceptual_compare.pdf import render_page_pairs
from pdf_perceptual_compare.comparator import (
    ComparisonOptions,
    PageResult,
    compare_page,
    crop_for_shift,
)
from pdf_perceptual_compare.cli import (
    compare_rendered_pages,
    main,
    render_page_result,
)
from pdf_perceptual_compare.verdict import Verdict

from threading import Barrier, Lock, BrokenBarrierError
from json import loads
from pathlib import Path
from shutil import which
from subprocess import CalledProcessError

from numpy import zeros, uint8
import pytest

pytest.importorskip("skimage")


def test_render_page_result_colors_failures_only_for_terminals() -> None:
    """Verify that failure colors are only applied in terminal output.

    Checks that render_page_result prefixes failures with ANSI red codes when
    use_color=True and plain text otherwise.
    """
    failure = PageResult(1, False, 0, 0, 0.9, 0.9, 0.9, 0.9, 0.1, Verdict.FAIL)

    assert render_page_result(failure, use_color=True).startswith("\033[31m")
    assert render_page_result(
        failure, use_color=False).startswith("   1  FAIL")


FIXTURES_DIRECTORY = Path(__file__).parent
ORIGINAL_PDF = FIXTURES_DIRECTORY / "original.pdf"
DIETPDF_PDF = FIXTURES_DIRECTORY / "compressed_with_dietpdf.pdf"
INVALID_PDF = FIXTURES_DIRECTORY / "bad.pdf"
DIFFERENT_PAGE_COUNT_PDF = FIXTURES_DIRECTORY / "different_number_of_pages.pdf"
COMPRESSED_PDFS_WITH_ERRORS = [
    FIXTURES_DIRECTORY / "compressed_with_adobe_standard.pdf",
    FIXTURES_DIRECTORY / "compressed_with_filevert_standard.pdf",
    FIXTURES_DIRECTORY / "compressed_with_ilovepdf_good.pdf",
    FIXTURES_DIRECTORY / "compressed_with_ilovepdf_high.pdf",
    FIXTURES_DIRECTORY / "compressed_with_smallpdf_simple.pdf",
]

requires_poppler = pytest.mark.skipif(
    which("pdfinfo") is None or which("pdftoppm") is None,
    reason="PDF fixture tests require Poppler's pdfinfo and pdftoppm commands",
)


def test_compare_page_accepts_identical_images() -> None:
    """Verify that compare_page returns PASS for identical images.

    Compares two identical 16x16 RGB images and confirms the result has verdict
    'PASS', identical=True, and SSIM=1.0.
    """
    image = zeros((16, 16, 3), dtype=uint8)

    result = compare_page(image, image.copy(), page=1,
                          options=ComparisonOptions())

    assert result.verdict.is_pass
    assert result.identical is True
    assert result.ssim == 1.0


def test_compare_page_rejects_different_dimensions() -> None:
    """Verify that compare_page returns FAIL for mismatched dimensions."""
    result = compare_page(
        zeros((16, 16, 3), dtype=uint8),
        zeros((15, 16, 3), dtype=uint8),
        page=1,
        options=ComparisonOptions(),
    )

    assert result.verdict == Verdict.FAIL_SIZE


def test_crop_for_shift_returns_matching_overlap() -> None:
    """Verify that crop_for_shift returns images with matching overlap."""
    image = zeros((10, 10, 3), dtype=uint8)

    shifted_a, shifted_b = crop_for_shift(image, image, shift_x=2, shift_y=-1)

    assert shifted_a.shape == shifted_b.shape == (9, 8, 3)


def test_render_page_pairs_returns_paths_in_page_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Verify that render_page_pairs returns paths in page order.

    Mocks the rendering function to return predictable filenames and confirms
    the output list matches expected (a-0001.png, b-0001.png), (a-0002.png,
    b-0002.png).
    """
    def fake_render_page(pdf, page, dpi, output_base):
        return output_base.with_suffix(".png")

    monkeypatch.setattr(
        "pdf_perceptual_compare.pdf.render_page", fake_render_page)

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
    """Verify that render_page_pairs runs rendering concurrently.

    Uses a Barrier and Lock to track concurrent execution and confirms
    maximum_active reaches 2 when jobs=2 is specified.
    """
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    maximum_active = 0

    def fake_render_page(pdf, page, dpi, output_base):
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
        return output_base.with_suffix(".png")

    monkeypatch.setattr(
        "pdf_perceptual_compare.pdf.render_page", fake_render_page)

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
    """Verify that compare_rendered_pages runs comparisons concurrently.

    Uses a Barrier and Lock to track concurrent execution and confirms
    maximum_active reaches 2 when jobs=2 is specified. Also verifies both pages
    are compared (page set equals {1, 2}).
    """
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
        "pdf_perceptual_compare.cli.load_rgb", lambda path: image)
    monkeypatch.setattr(
        "pdf_perceptual_compare.cli.compare_page", fake_compare_page)

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


def run_comparison(
    monkeypatch: pytest.MonkeyPatch,
    original: Path,
    candidate: Path,
    json_report: Path | None = None,
) -> int:
    """Run the CLI with deterministic, low-concurrency fixture settings.

    Executes pdf-perceptual-compare with the given files and optional JSON
    report path, patching sys.argv to ensure consistent argument parsing.

    Args:
        monkeypatch: pytest.MonkeyPatch instance for test isolation.
        original (Path): Path to the original PDF file.
        candidate (Path): Path to the candidate PDF file.
        json_report (Path | None, optional): Optional path for JSON report
            output.

    Returns:
        int: The exit code from main(). 0 indicates success, non-zero indicates
            failure.
    """
    arguments = [
        "pdf-perceptual-compare",
        str(original),
        str(candidate),
        "--jobs",
        "2",
    ]

    if json_report is not None:
        arguments.extend(["--json", str(json_report)])

    monkeypatch.setattr("sys.argv", arguments)

    return main()


@requires_poppler
def test_dietpdf_compression_preserves_all_pages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify that dietpdf compression preserves all pages.

    Runs the comparison against a dietpdf-compressed file and confirms all 3
    pages pass with zero failures in both CLI output and JSON report.
    """
    report = tmp_path / "dietpdf.json"
    assert run_comparison(monkeypatch, ORIGINAL_PDF, DIETPDF_PDF, report) == 0

    output = capsys.readouterr().out
    assert "Passed pages    : 3/3" in output
    assert "Failed pages    : 0/3" in output

    payload = loads(report.read_text(encoding="utf-8"))
    assert payload["summary"] == {
        "identical_pages": 3,
        "passed_pages": 3,
        "failed_pages": 0,
    }
    assert [result["verdict"] for result in payload["results"]] == ["PASS"] * 3


@requires_poppler
@pytest.mark.parametrize("candidate", COMPRESSED_PDFS_WITH_ERRORS)
def test_other_compressed_pdfs_report_at_least_one_failed_page(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    candidate: Path,
    tmp_path: Path,
) -> None:
    """Verify that other compressed PDFs report at least one failed page.

    Tests multiple compression tools (Adobe Standard, Filevert, iLovePDF,
    Smallpdf) and confirms each produces a non-empty failure count in both CLI
    output and JSON report.
    """
    report = tmp_path / f"{candidate.stem}.json"
    assert run_comparison(monkeypatch, ORIGINAL_PDF, candidate, report) == 1

    output = capsys.readouterr().out
    assert "Failed pages    : 0/3" not in output
    assert "Failed page(s)  :" in output

    payload = loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["failed_pages"] >= 1
    assert any(result["verdict"].startswith("FAIL")
               for result in payload["results"])


@requires_poppler
def test_invalid_pdf_raises_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that invalid PDFs raise a CalledProcessError.

    Tests with an intentionally malformed PDF and confirms the CLI exits with an
    error code without generating a JSON report file.
    """
    report = tmp_path / "invalid.json"

    with pytest.raises(CalledProcessError):
        run_comparison(monkeypatch, ORIGINAL_PDF, INVALID_PDF, report)

    assert not report.exists()


@requires_poppler
def test_different_page_counts_report_an_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Verify that different page counts report an error.

    Tests with a PDF having a mismatched page count and confirms the CLI outputs
    the specific failure message while not generating a JSON report.
    """
    report = tmp_path / "different-page-count.json"
    assert (
        run_comparison(monkeypatch, ORIGINAL_PDF,
                       DIFFERENT_PAGE_COUNT_PDF, report)
        == 1
    )

    assert "FAIL: different number of pages: 3 != 2" in capsys.readouterr().out
    assert not report.exists()
