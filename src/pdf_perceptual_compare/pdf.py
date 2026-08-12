"""PDF rendering helpers backed by Poppler utilities."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from shutil import which
from subprocess import CompletedProcess, run, PIPE
from sys import stderr
from pathlib import Path

from numpy import ndarray, asarray, uint8
from PIL import Image


def die(message: str, code: int = 2) -> None:
    """Print a user-facing error and stop with the supplied status code."""
    print(f"ERROR: {message}", file=stderr)
    raise SystemExit(code)


def require_command(name: str) -> None:
    """Ensure a required Poppler executable is available."""
    if which(name) is None:
        die(f"required command not found: {name}")


def run_command(*args: str, capture: bool = False) -> CompletedProcess[str]:
    """Run a Poppler command and propagate command failures."""
    return run(
        args,
        check=True,
        text=True,
        stdout=PIPE if capture else None,
        stderr=PIPE if capture else None,
    )


def page_count(pdf: Path) -> int:
    """Return the number of pages reported by ``pdfinfo``."""
    pdfinfo_lines = run_command(
        "pdfinfo",
        str(pdf),
        capture=True
    ).stdout.splitlines()

    for line in pdfinfo_lines:
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())

    die(f"could not determine page count of {pdf}")

    raise AssertionError("unreachable")


def render_page(pdf: Path, page: int, dpi: int, output_base: Path) -> Path:
    """Render one PDF page as PNG and return its path."""
    run_command(
        "pdftoppm",
        "-r", str(dpi),
        "-f", str(page),
        "-l", str(page),
        "-singlefile",
        "-png",
        str(pdf),
        str(output_base),
    )

    output = output_base.with_suffix(".png")

    if not output.exists():
        die(f"pdftoppm did not produce {output}")

    return output


def render_page_pairs(
    original: Path,
    candidate: Path,
    pages: int,
    dpi: int,
    output_directory: Path,
    jobs: int,
    progress: Callable[[int, int], None] | None = None,
) -> list[tuple[Path, Path]]:
    """Render matching pages concurrently, return their paths in page order."""
    with ThreadPoolExecutor(
        max_workers=jobs,
        thread_name_prefix="pdf-render"
    ) as executor:
        original_futures: dict[int, Future[Path]] = {}
        candidate_futures: dict[int, Future[Path]] = {}

        for page in range(1, pages + 1):
            original_futures[page] = executor.submit(
                render_page, original, page, dpi, output_directory /
                f"a-{page:04d}"
            )

            candidate_futures[page] = executor.submit(
                render_page, candidate, page, dpi, output_directory /
                f"b-{page:04d}"
            )

        futures = [*original_futures.values(), *candidate_futures.values()]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if progress is not None:
                progress(completed, len(futures))

        return [
            (original_futures[page].result(), candidate_futures[page].result())
            for page in range(1, pages + 1)
        ]


def load_rgb(path: Path) -> ndarray:
    """Load an image as an RGB uint8 NumPy array."""
    with Image.open(path) as image:
        return asarray(image.convert("RGB"), dtype=uint8)
