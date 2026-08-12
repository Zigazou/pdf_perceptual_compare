"""Command-line interface for perceptual PDF comparison."""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from os import cpu_count
from json import dumps
from shutil import copy2
from sys import stdout
from tempfile import TemporaryDirectory
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .comparator import compare_page
from .page_result import PageResult
from .comparison_options import ComparisonOptions
from .pdf import die, load_rgb, page_count, render_page_pairs, require_command

# ANSI escape codes for terminal colorization
RED = "\033[31m"
RESET = "\033[0m"


def positive_integer(value: str) -> int:
    """Return a strictly positive command-line integer.

    Args:
        value (str): The string representation of the integer to parse.

    Returns:
        int: A positive integer greater than or equal to 1.

    Raises:
        ArgumentTypeError: If the input cannot be parsed as an integer
            or if the resulting integer is less than 1.
    """
    integer = int(value)

    if integer < 1:
        raise ArgumentTypeError("must be at least 1")

    return integer


def parse_args() -> Namespace:
    """Parse command-line arguments for the PDF comparison tool.

    Returns:
        Namespace: Parsed argument namespace containing all CLI options.
    """
    parser = ArgumentParser(
        description="Perceptual page-by-page comparison of two PDFs."
    )

    parser.add_argument("original", type=Path)

    parser.add_argument("candidate", type=Path)

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="rendering resolution (default: 150)"
    )

    parser.add_argument(
        "--jobs",
        type=positive_integer,
        default=cpu_count() or 1,
        help="maximum concurrent page operations (default: CPU count)",
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=128,
        help="tile size for local SSIM statistics (default: 128)",
    )

    parser.add_argument(
        "--blur",
        type=float,
        default=0.5,
        help="Gaussian blur radius in pixels (default: 0.5)",
    )

    parser.add_argument(
        "--align",
        type=int,
        default=0,
        help="try integer shifts up to N pixels (default: 0)",
    )

    parser.add_argument(
        "--ssim",
        type=float,
        default=0.995,
        help="minimum raw global SSIM (default: 0.995)",
    )

    parser.add_argument(
        "--ssim-blur",
        type=float,
        default=0.999,
        help="minimum blurred global SSIM (default: 0.999)",
    )

    parser.add_argument(
        "--local-p01",
        type=float,
        default=0.980,
        help="minimum 1st percentile tile SSIM (default: 0.980)",
    )

    parser.add_argument(
        "--local-threshold",
        type=float,
        default=0.980,
        help="suspicious tile SSIM threshold (default: 0.980)",
    )

    parser.add_argument(
        "--max-bad-fraction",
        type=float,
        default=0.005,
        help="maximum suspicious tile fraction (default: 0.005)",
    )

    parser.add_argument(
        "--json",
        type=Path,
        help="write full results as JSON"
    )

    parser.add_argument(
        "--save-failures",
        type=Path,
        help="save diagnostic images for failed pages"
    )

    parser.add_argument(
        "--keep-rendered",
        type=Path,
        help="copy rendered pages to this directory"
    )

    return parser.parse_args()


def options_from_args(args: Namespace) -> ComparisonOptions:
    """Build comparison options from parsed CLI arguments.

    Args:
        args (Namespace): The namespace object containing all CLI arguments.

    Returns:
        ComparisonOptions: A fully configured comparison options instance.
    """
    return ComparisonOptions(
        tile=args.tile,
        blur=args.blur,
        align=args.align,
        ssim=args.ssim,
        ssim_blur=args.ssim_blur,
        local_p01=args.local_p01,
        local_threshold=args.local_threshold,
        max_bad_fraction=args.max_bad_fraction,
    )


def write_json_report(
    args: Namespace,
    pages: int,
    results: list[PageResult]
) -> None:
    """Write an optional machine-readable comparison report.

    Args:
        args (Namespace): The parsed CLI arguments namespace.
        pages (int): Total number of pages compared.
        results (list[PageResult]): List of page comparison result objects.

    Returns:
        None: Writes the JSON report to the path specified by `args.json`.
    """
    if args.json is None:
        return

    failed = [
        result
        for result in results
        if not result.verdict.startswith("PASS")
    ]

    payload = {
        "original": str(args.original),
        "candidate": str(args.candidate),
        "pages": pages,
        "parameters": {
            "dpi": args.dpi,
            "jobs": args.jobs,
            **asdict(options_from_args(args)),
        },
        "summary": {
            "identical_pages": sum(result.identical for result in results),
            "passed_pages": pages - len(failed),
            "failed_pages": len(failed),
        },
        "results": [asdict(result) for result in results],
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)

    args.json.write_text(
        dumps(payload, indent=2) + "\n",
        encoding="utf-8"
    )


def compare_rendered_page(
    page: int,
    original_png: Path,
    candidate_png: Path,
    options: ComparisonOptions,
    failure_dir: Path | None,
    keep_rendered: Path | None,
) -> PageResult:
    """Load and compare one rendered page pair, optionally retaining its images.

    Args:
        page (int): The page number being compared.
        original_png (Path): Path to the original rendered PNG image.
        candidate_png (Path): Path to the candidate rendered PNG image.
        options (ComparisonOptions): Comparison configuration parameters.
        failure_dir (Path | None): Directory for saving diagnostic images on
            failure.
        keep_rendered (Path | None): Directory for keeping rendered page copies.

    Returns:
        PageResult: A result object containing comparison metrics and verdict.
    """
    if keep_rendered:
        copy2(original_png, keep_rendered / f"original-{page:04d}.png")
        copy2(candidate_png, keep_rendered / f"candidate-{page:04d}.png")

    return compare_page(
        load_rgb(original_png),
        load_rgb(candidate_png),
        page,
        options,
        failure_dir
    )


def compare_rendered_pages(
    rendered_pages: list[tuple[Path, Path]],
    options: ComparisonOptions,
    failure_dir: Path | None,
    keep_rendered: Path | None,
    jobs: int,
) -> Iterator[PageResult]:
    """Compare rendered pages concurrently, yielding results as they finish.

    Args:
        rendered_pages (list[tuple[Path, Path]]): List of page image pairs.
        options (ComparisonOptions): Comparison configuration parameters.
        failure_dir (Path | None): Directory for saving diagnostic images on
            failure.
        keep_rendered (Path | None): Directory for keeping rendered page copies.
        jobs (int): Maximum number of concurrent worker threads.

    Yields:
        PageResult: A result object containing comparison metrics and verdict,
            yielded as each page completes processing.
    """
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                compare_rendered_page,
                page,
                original_png,
                candidate_png,
                options,
                failure_dir,
                keep_rendered,
            )
            for page, (original_png, candidate_png)
            in enumerate(rendered_pages, start=1)
        ]

        for future in as_completed(futures):
            yield future.result()


def render_page_result(result: PageResult, use_color: bool) -> str:
    """Format one page result, highlighting failed comparisons in a terminal.

    Args:
        result (PageResult): The page comparison result to format.
        use_color (bool): Whether to apply ANSI color codes for failed results.

    Returns:
        str: A formatted string representation of the page result.
    """
    shift = (
        f" shift={result.shift_x:+d},{result.shift_y:+d}"
        if result.shift_x or result.shift_y
        else ""
    )

    line = (
        f"{result.page:4d}  "
        f"{result.verdict:10s} "
        f"SSIM={result.ssim:.6f}  "
        f"blur={result.ssim_blur:.6f}  "
        f"p01={result.local_p01:.6f}  "
        f"min={result.local_min:.6f}  "
        f"bad={result.local_bad_fraction:7.3%}{shift}"
    )

    if use_color and not result.verdict.startswith("PASS"):
        return f"{RED}{line}{RESET}"

    return line


def main() -> int:
    """Run the command-line application for perceptual PDF comparison."""
    args = parse_args()
    use_color = stdout.isatty()

    # Check that required external commands are available and that input files
    # exist.
    require_command("pdfinfo")
    require_command("pdftoppm")

    # Check that the input files exist.
    for pdf in (args.original, args.candidate):
        if not pdf.is_file():
            die(f"file not found: {pdf}")

    # Check that the input files have the same number of pages.
    pages_a, pages_b = page_count(args.original), page_count(args.candidate)
    if pages_a != pages_b:
        message = f"FAIL: different number of pages: {pages_a} != {pages_b}"
        print(f"{RED}{message}{RESET}" if use_color else message)

        return 1

    options = options_from_args(args)

    print(f"Pages: {pages_a}\nDPI: {args.dpi}")
    print(f"Rendering jobs: {args.jobs}")
    print(
        f"Thresholds: SSIM>={options.ssim}, "
        f"blur-SSIM>={options.ssim_blur}, "
        f"local-p01>={options.local_p01}, "
        f"bad-tiles<={options.max_bad_fraction:.3%}"
    )

    if options.align:
        print(f"Alignment: ±{options.align}px (integer shifts)")

    print()

    if args.keep_rendered:
        args.keep_rendered.mkdir(parents=True, exist_ok=True)

    results: list[PageResult] = []

    with TemporaryDirectory(prefix="pdf-perceptual-") as temporary_directory:
        temporary_path = Path(temporary_directory)

        print(f"Rendering {pages_a * 2} page images...", end="", flush=True)

        rendered_pages = render_page_pairs(
            args.original,
            args.candidate,
            pages_a,
            args.dpi,
            temporary_path,
            args.jobs,
            progress=lambda completed, total: print(
                f"\rRendering {completed}/{total} page images...",
                end="",
                flush=True,
            ),
        )

        print()

        for result in compare_rendered_pages(
            rendered_pages,
            options,
            args.save_failures,
            args.keep_rendered,
            args.jobs,
        ):
            results.append(result)
            print(render_page_result(result, use_color))

    failed = [
        result
        for result in results
        if not result.verdict.startswith("PASS")
    ]

    identical = sum(result.identical for result in results)
    print(
        "\n"
        f"Identical pages : {identical}/{pages_a}\n"
        f"Passed pages    : {pages_a - len(failed)}/{pages_a}\n"
        f"Failed pages    : {len(failed)}/{pages_a}"
    )

    if failed:
        failed_pages = " ".join(str(result.page) for result in failed)
        print("Failed page(s)  : " + failed_pages)

    write_json_report(args, pages_a, results)

    return int(bool(failed))
