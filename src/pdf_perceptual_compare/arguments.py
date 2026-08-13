"""Command-line interface for perceptual PDF comparison."""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from os import cpu_count
from pathlib import Path

from .comparison_options import ComparisonOptions


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
