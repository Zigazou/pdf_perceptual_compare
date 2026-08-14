"""End-to-end CLI tests using real PDF fixtures."""

from json import loads
from pathlib import Path

import pytest
from conftest import (
    COMPRESSED_PDFS_WITH_ERRORS,
    DIETPDF_PDF,
    DIFFERENT_PAGE_COUNT_PDF,
    ILOVEPDF_HIGH_PDF,
    INVALID_PDF,
    ORIGINAL_PDF,
    requires_pypdfium2,
    run_comparison,
)


@requires_pypdfium2
def test_dietpdf_compression_preserves_all_pages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    """Verify that dietpdf compression preserves all pages."""
    report = tmp_path / "dietpdf.json"

    assert run_comparison(monkeypatch, ORIGINAL_PDF, DIETPDF_PDF, report) == 0

    output = capsys.readouterr().out

    assert "Passed pages    : 3/3" in output
    assert "Failed pages    : 0/3" in output

    payload = loads(report.read_text(encoding="utf-8"))

    assert payload["summary"] == {
        "identical_pages": 0,
        "passed_pages": 3,
        "failed_pages": 0
    }

    assert [result["verdict"] for result in payload["results"]] == ["PASS"] * 3


@requires_pypdfium2
def test_same_pdf_gives_identical_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    """Verify that comparing the same PDF gives an identical verdict."""
    report = tmp_path / "same.json"

    assert run_comparison(monkeypatch, ORIGINAL_PDF, ORIGINAL_PDF, report) == 0

    output = capsys.readouterr().out

    assert "Passed pages    : 3/3" in output
    assert "Failed pages    : 0/3" in output

    payload = loads(report.read_text(encoding="utf-8"))

    assert payload["summary"] == {
        "identical_pages": 3,
        "passed_pages": 3,
        "failed_pages": 0
    }

    assert [result["verdict"] for result in payload["results"]] == ["PASS"] * 3


@requires_pypdfium2
@pytest.mark.parametrize("candidate", COMPRESSED_PDFS_WITH_ERRORS)
def test_other_compressed_pdfs_report_at_least_one_failed_page(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    candidate: Path,
    tmp_path: Path,
) -> None:
    """Verify that other compressed PDFs report at least one failed page."""
    report = tmp_path / f"{candidate.stem}.json"

    assert run_comparison(monkeypatch, ORIGINAL_PDF, candidate, report) == 1

    output = capsys.readouterr().out

    assert "Failed pages    : 0/3" not in output
    assert "Failed page(s)  :" in output

    payload = loads(report.read_text(encoding="utf-8"))

    assert payload["summary"]["failed_pages"] >= 1

    assert any(
        result["verdict"].startswith("FAIL")
        for result in payload["results"]
    )


@requires_pypdfium2
def test_ilovepdf_high_compression_passes_with_pypdfium2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    """Verify that iLovePDF high compression meets the configured thresholds."""
    report = tmp_path / "ilovepdf-high.json"

    assert run_comparison(
        monkeypatch,
        ORIGINAL_PDF,
        ILOVEPDF_HIGH_PDF,
        report
    ) == 0

    output = capsys.readouterr().out

    assert "Passed pages    : 3/3" in output
    assert "Failed pages    : 0/3" in output

    payload = loads(report.read_text(encoding="utf-8"))

    assert payload["summary"]["failed_pages"] == 0
    assert all(result["verdict"] == "PASS" for result in payload["results"])


@requires_pypdfium2
def test_invalid_pdf_raises_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    """Verify that invalid PDFs raise a user-facing error."""
    report = tmp_path / "invalid.json"

    with pytest.raises(SystemExit) as error:
        run_comparison(monkeypatch, ORIGINAL_PDF, INVALID_PDF, report)

    assert error.value.code == 2
    assert not report.exists()


@requires_pypdfium2
def test_different_page_counts_report_an_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path
) -> None:
    """Verify that different page counts report an error."""
    report = tmp_path / "different-page-count.json"

    assert run_comparison(
        monkeypatch,
        ORIGINAL_PDF,
        DIFFERENT_PAGE_COUNT_PDF,
        report
    ) == 1

    assert "FAIL: different number of pages: 3 != 2" in capsys.readouterr().out
    assert not report.exists()
