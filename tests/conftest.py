"""Shared fixtures and helpers for the test suite."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from pdf_perceptual_compare.cli import main

pytest.importorskip("skimage")

requires_pypdfium2 = pytest.mark.skipif(
    find_spec("pypdfium2") is None,
    reason="PDF fixture tests require pypdfium2",
)

FIXTURES_DIRECTORY = Path(__file__).parent
ORIGINAL_PDF = FIXTURES_DIRECTORY / "original.pdf"
DIETPDF_PDF = FIXTURES_DIRECTORY / "compressed_with_dietpdf.pdf"
INVALID_PDF = FIXTURES_DIRECTORY / "bad.pdf"
DIFFERENT_PAGE_COUNT_PDF = FIXTURES_DIRECTORY / "different_number_of_pages.pdf"
COMPRESSED_PDFS_WITH_ERRORS = [
    FIXTURES_DIRECTORY / "compressed_with_adobe_standard.pdf",
    FIXTURES_DIRECTORY / "compressed_with_filevert_standard.pdf",
    FIXTURES_DIRECTORY / "compressed_with_ilovepdf_good.pdf",
    FIXTURES_DIRECTORY / "compressed_with_smallpdf_simple.pdf",
]
ILOVEPDF_HIGH_PDF = FIXTURES_DIRECTORY / "compressed_with_ilovepdf_high.pdf"


def run_comparison(
    monkeypatch: pytest.MonkeyPatch,
    original: Path,
    candidate: Path,
    json_report: Path | None = None,
) -> int:
    """Run the CLI with deterministic, low-concurrency fixture settings."""
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
