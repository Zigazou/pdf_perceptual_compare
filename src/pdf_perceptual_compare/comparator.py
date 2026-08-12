"""Image comparison primitives used by the PDF comparison CLI."""

from __future__ import annotations

from math import inf
from pathlib import Path

from numpy import (
    ndarray, float32, uint8, asarray, mean, clip, array_equal, percentile
)
from PIL import Image, ImageChops, ImageFilter
from skimage.metrics import structural_similarity

from .comparison_options import ComparisonOptions
from .page_result import PageResult


def crop_for_shift(
    original: ndarray,
    candidate: ndarray,
    shift_x: int,
    shift_y: int
) -> tuple[ndarray, ndarray]:
    """Return overlapping areas after shifting ``candidate`` by
    ``(shift_x, shift_y)`` relative to ``original``.
    """
    height, width = original.shape[:2]

    ax0, ax1 = max(0, shift_x), min(width, width + shift_x)
    ay0, ay1 = max(0, shift_y), min(height, height + shift_y)
    bx0, bx1 = max(0, -shift_x), min(width, width - shift_x)
    by0, by1 = max(0, -shift_y), min(height, height - shift_y)

    return original[ay0:ay1, ax0:ax1], candidate[by0:by1, bx0:bx1]


def alignment_score(
    original: ndarray,
    candidate: ndarray,
    shift_x: int,
    shift_y: int
) -> float:
    """Calculate a fast, downsampled luminance error for a candidate shift."""
    shifted_original, shifted_candidate = crop_for_shift(
        original,
        candidate,
        shift_x,
        shift_y
    )

    luminance_original = shifted_original[::4, ::4].astype(
        float32).mean(axis=2)

    luminance_candidate = shifted_candidate[::4, ::4].astype(
        float32).mean(axis=2)

    return float(mean((luminance_original - luminance_candidate) ** 2))


def best_integer_shift(
    original: ndarray,
    candidate: ndarray,
    max_shift: int
) -> tuple[int, int]:
    """Find the best integer translation in the configured range."""
    if max_shift <= 0:
        return 0, 0

    best = (inf, 0, 0)
    for shift_y in range(-max_shift, max_shift + 1):
        for shift_x in range(-max_shift, max_shift + 1):
            score = alignment_score(original, candidate, shift_x, shift_y)
            if score < best[0]:
                best = (score, shift_x, shift_y)

    return best[1], best[2]


def blur_rgb(image: ndarray, radius: float) -> ndarray:
    """Apply a Gaussian blur, returning the original array when disabled."""
    if radius <= 0:
        return image

    pil_image = Image.fromarray(image, mode="RGB")

    return asarray(
        pil_image.filter(ImageFilter.GaussianBlur(radius)),
        dtype=uint8
    )


def ssim_with_map(
    original: ndarray,
    candidate: ndarray
) -> tuple[float, ndarray]:
    """Calculate global SSIM and its two-dimensional similarity map."""
    result = structural_similarity(
        original,
        candidate,
        data_range=255,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=True,
    )

    score = result[0]
    score_map = result[1]

    if score_map.ndim == 3:
        score_map = score_map.mean(axis=2)

    return float(score), score_map.astype(float32)


def tile_means(score_map: ndarray, tile: int) -> ndarray:
    """Calculate mean SSIM values for sufficiently large image tiles."""
    height, width = score_map.shape
    values: list[float] = []

    for row in range(0, height, tile):
        for column in range(0, width, tile):
            block = score_map[
                row: min(row + tile, height),
                column: min(column + tile, width)
            ]

            if block.size >= max(64, tile * tile // 4):
                values.append(float(block.mean()))

    if not values:
        values.append(float(score_map.mean()))

    return asarray(values, dtype=float32)


def save_diagnostic(
    original: ndarray,
    candidate: ndarray,
    score_map: ndarray,
    output: Path
) -> None:
    """Save amplified differences and an SSIM similarity map for a failed page."""
    def transform(value):
        return min(255, value * 8)

    output.mkdir(parents=True, exist_ok=True)
    difference = ImageChops.difference(
        Image.fromarray(original),
        Image.fromarray(candidate)
    )

    difference.point(transform).save(
        output / "diff-amplified.png"
    )

    similarity = clip(score_map, 0.0, 1.0)

    Image.fromarray((similarity * 255.0).astype(uint8), mode="L").save(
        output / "ssim-map.png"
    )


def compare_page(
    original: ndarray,
    candidate: ndarray,
    page: int,
    options: ComparisonOptions,
    failure_dir: Path | None = None,
) -> PageResult:
    """Compare two rendered pages and return their perceptual result."""

    # The two pages must be the same size to be considered identical.
    if original.shape != candidate.shape:
        return PageResult(
            page=page,
            identical=False,
            shift_x=0,
            shift_y=0,
            ssim=0.0,
            ssim_blur=0.0,
            local_p01=0.0,
            local_min=0.0,
            local_bad_fraction=1.0,
            verdict="FAIL(size)"
        )

    # If the two pages are pixel-identical, we can skip the rest of the
    # comparison.
    if array_equal(original, candidate):
        return PageResult(
            page=page,
            identical=True,
            shift_x=0,
            shift_y=0,
            ssim=1.0,
            ssim_blur=1.0,
            local_p01=1.0,
            local_min=1.0,
            local_bad_fraction=0.0,
            verdict="PASS"
        )

    # Calculate the best integer translation of the candidate relative to the
    # original.
    shift_x, shift_y = best_integer_shift(original, candidate, options.align)

    # Crop the images according to the calculated shift.
    shifted_original, shifted_candidate = crop_for_shift(
        original,
        candidate,
        shift_x,
        shift_y
    )

    # Calculate global SSIM.
    global_ssim, score_map = ssim_with_map(shifted_original, shifted_candidate)

    # Calculate local SSIM statistics on tiles of the similarity map.
    tiles = tile_means(score_map, options.tile)

    local_p01 = float(percentile(tiles, 1))
    local_min = float(tiles.min())
    local_bad_fraction = float(mean(tiles < options.local_threshold))

    # Calculate global SSIM after blurring both images.
    blur_ssim, _ = ssim_with_map(
        blur_rgb(shifted_original, options.blur),
        blur_rgb(shifted_candidate, options.blur)
    )

    # Determine whether the page passes or fails based on the configured
    # thresholds.
    raw_ok = (
        global_ssim >= options.ssim
        and local_p01 >= options.local_p01
        and local_bad_fraction <= options.max_bad_fraction
    )

    # Determine whether the page passes or fails based on the configured
    # thresholds after blurring both images.
    blur_ok = (
        blur_ssim >= options.ssim_blur
        and local_bad_fraction <= options.max_bad_fraction
    )

    # Determine the final verdict based on the raw and blurred SSIM results.
    verdict = "PASS" if raw_ok or blur_ok else "FAIL"

    # If the page fails and a failure directory is specified, save diagnostic
    # images.
    if verdict == "FAIL" and failure_dir is not None:
        save_diagnostic(
            shifted_original,
            shifted_candidate,
            score_map,
            failure_dir / f"page-{page:04d}"
        )

    # Return the page result with all relevant metrics and the final verdict.
    return PageResult(
        page=page,
        identical=False,
        shift_x=shift_x,
        shift_y=shift_y,
        ssim=global_ssim,
        ssim_blur=blur_ssim,
        local_p01=local_p01,
        local_min=local_min,
        local_bad_fraction=local_bad_fraction,
        verdict=verdict,
    )
