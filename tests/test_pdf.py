"""Unit tests for PDF inspection and rendering."""

from pathlib import Path
from time import monotonic, sleep

import pytest

from pdf_perceptual_compare.pdf import (
    page_count,
    render_page,
    render_page_pairs
)


def fake_render_page(pdf: Path, page: int, dpi: int, output_base: Path) -> Path:
    """Return the path that a renderer would create without rendering a PDF."""
    return output_base.with_suffix(".png")


def fake_render_page_wait_for_peer(
    pdf: Path,
    page: int,
    dpi: int,
    output_base: Path
) -> Path:
    """Record whether another renderer starts concurrently in this directory."""
    ready = output_base.with_suffix(".ready")
    concurrent = output_base.parent / ".concurrent-render"
    ready.touch()
    deadline = monotonic() + 1

    try:
        while monotonic() < deadline:
            if len(list(output_base.parent.glob("*.ready"))) >= 2:
                concurrent.touch()
                break

            sleep(0.01)
    finally:
        ready.unlink(missing_ok=True)

    return output_base.with_suffix(".png")


def test_render_page_pairs_returns_paths_in_page_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    """Verify that render_page_pairs returns paths in page order."""
    monkeypatch.setattr(
        "pdf_perceptual_compare.pdf.render_page",
        fake_render_page
    )

    rendered = render_page_pairs(
        tmp_path / "original.pdf",
        tmp_path / "candidate.pdf",
        2,
        150,
        tmp_path,
        2
    )

    assert rendered == [
        (tmp_path / "a-0001.png", tmp_path / "b-0001.png"),
        (tmp_path / "a-0002.png", tmp_path / "b-0002.png"),
    ]


def test_pypdfium2_provides_page_count_and_png_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    """Verify the binding supplies the page count and rendered PNG output."""
    calls: list[object] = []

    class FakeDocument:
        def __len__(self) -> int:
            return 3

        def get_page(self, index: int) -> "FakePage":
            calls.append(index)
            return FakePage()

        def close(self) -> None:
            calls.append("document.close")

    class FakePage:
        def render(self, *, scale: float) -> "FakeBitmap":
            calls.append(scale)
            return FakeBitmap()

        def close(self) -> None:
            calls.append("page.close")

    class FakeBitmap:
        def to_pil(self) -> "FakeImage":
            calls.append("to_pil")
            return FakeImage()

        def close(self) -> None:
            calls.append("bitmap.close")

    class FakeImage:
        def save(
            self,
            filename: Path,
            image_format: str,
            *,
            dpi: tuple[int, int]
        ) -> None:
            calls.append((image_format, dpi))
            Path(filename).touch()

        def close(self) -> None:
            calls.append("image.close")

    document = FakeDocument()

    monkeypatch.setattr(
        "pdf_perceptual_compare.pdf.PdfDocument",
        lambda path: document
    )

    pdf = tmp_path / "input.pdf"
    output = render_page(pdf, page=2, dpi=150, output_base=tmp_path / "page")

    assert page_count(pdf) == 3
    assert output == tmp_path / "page.png"
    assert calls == [
        1,
        150 // 72,
        "to_pil",
        ("PNG", (150, 150)),
        "image.close",
        "bitmap.close",
        "page.close",
        "document.close",
        "document.close",
    ]


def test_render_page_pairs_schedules_rendering_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    """Verify that render_page_pairs schedules rendering concurrently."""
    monkeypatch.setattr(
        "pdf_perceptual_compare.pdf.render_page",
        fake_render_page_wait_for_peer
    )

    render_page_pairs(
        tmp_path / "original.pdf",
        tmp_path / "candidate.pdf",
        1,
        150,
        tmp_path,
        2
    )

    assert (tmp_path / ".concurrent-render").exists()
