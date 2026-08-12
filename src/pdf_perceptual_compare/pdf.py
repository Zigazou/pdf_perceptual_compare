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
    """Print a user-facing error and stop with the supplied status code.

    Args:
        message (str): The error message to display to the user.
        code (int, optional): The exit code for SystemExit. Defaults to 2.

    Raises:
        SystemExit: With the provided exit code.
    """
    print(f"ERROR: {message}", file=stderr)
    raise SystemExit(code)


def require_command(name: str) -> None:
    """Ensure a required Poppler executable is available in PATH.

    Args:
        name (str): The command name to search for using ``which``.

    Raises:
        SystemExit: If the command cannot be found and exits with code 2.
    """
    if which(name) is None:
        die(f"required command not found: {name}")


def run_command(*args: str, capture: bool = False) -> CompletedProcess[str]:
    """Run a Poppler command and propagate command failures.

    Args:
        *args (str): Command arguments to execute.
        capture (bool, optional): Whether to capture stdout/stderr. Defaults to
            False.

    Returns:
        CompletedProcess[str]: The result of the subprocess execution.

    Raises:
        CalledProcessError: If the command exits with a non-zero status code.
    """
    return run(
        args,
        check=True,
        text=True,
        stdout=PIPE if capture else None,
        stderr=PIPE if capture else None,
    )


def page_count(pdf: Path) -> int:
    """Return the number of pages reported by ``pdfinfo``.

    Args:
        pdf (Path): The path to a PDF file.

    Returns:
        int: The total page count of the PDF document.

    Raises:
        SystemExit: If ``pdfinfo`` cannot determine the page count.
    """
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
    """Render one PDF page as PNG and return its path.

    Args:
        pdf (Path): The source PDF file.
        page (int): The 1-based page number to render.
        dpi (int): The resolution in dots per inch.
        output_base (Path): Base directory for the rendered image files.

    Returns:
        Path: The path to the generated PNG file.

    Raises:
        SystemExit: If ``pdftoppm`` fails to produce an output file.
    """
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
    """Render matching pages concurrently and return their paths in page order.

    Renders each page of both ``original`` and ``candidate`` PDFs at the
    specified DPI to PNG format using a thread pool, storing results as separate
    files prefixed with 'a-' and 'b-'.

    Args:
        original (Path): The source PDF file for comparison.
        candidate (Path): The candidate PDF file to compare against.
        pages (int): Total number of pages in the document.
        dpi (int): Resolution in dots per inch.
        output_directory (Path): Directory where rendered PNG files are saved.
        jobs (int): Maximum concurrent threads for rendering.
        progress (Callable[[int, int], None] | None, optional): Callback
            receiving current and total page counts. Defaults to None.

    Returns:
        list[tuple[Path, Path]]: List of tuples containing the rendered paths
            for each page in order, where each tuple is ``(original_page_path,
            candidate_page_path)``.

    Raises:
        SystemExit: If ``pdftoppm`` fails to produce an output file.
    """
    with ThreadPoolExecutor(
        max_workers=jobs,
        thread_name_prefix="pdf-render"
    ) as executor:
        original_futures: dict[int, Future[Path]] = {}
        candidate_futures: dict[int, Future[Path]] = {}

        for page in range(1, pages + 1):
            original_futures[page] = executor.submit(
                render_page,
                original,
                page,
                dpi,
                output_directory / f"a-{page:04d}",
            )

            candidate_futures[page] = executor.submit(
                render_page,
                candidate,
                page,
                dpi,
                output_directory / f"b-{page:04d}",
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
    """Load an image as an RGB uint8 NumPy array.

    Opens the given path using PIL and converts to RGB mode before returning
    a NumPy array with dtype ``uint8``.

    Args:
        path (Path): The path to the image file.

    Returns:
        ndarray: A 3D NumPy array of shape ``(height, width, 3)`` containing
            RGB pixel values in the range [0, 255].
    """
    with Image.open(path) as image:
        return asarray(image.convert("RGB"), dtype=uint8)
