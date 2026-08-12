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

    Args:
        original (ndarray): The reference image array.
        candidate (ndarray): The shifted image array to align with the original.
        shift_x (int): Horizontal pixel offset of the candidate relative to the
            original.
        shift_y (int): Vertical pixel offset of the candidate relative to the
            original.

    Returns:
        tuple[ndarray, ndarray]: A tuple containing the cropped original and
            candidate arrays that overlap after applying the specified shifts.
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
    """Calculate a fast, downsampled luminance error for a candidate shift.

    Computes the mean squared difference between the luminance channels of two
    downsampled image regions after applying the specified pixel shifts.

    Args:
        original (ndarray): The reference image array.
        candidate (ndarray): The shifted image array to align with the original.
        shift_x (int): Horizontal pixel offset of the candidate relative to the
            original.
        shift_y (int): Vertical pixel offset of the candidate relative to the
            original.

    Returns:
        float: The mean squared luminance difference between the two
            downsampled regions.
    """
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
    """Find the best integer translation in the configured range.

    Searches for the optimal horizontal and vertical pixel shift that minimizes
    luminance difference between two images within the specified bounds.

    Args:
        original (ndarray): The reference image array.
        candidate (ndarray): The shifted image array to align with the original.
        max_shift (int): Maximum absolute shift in pixels to search for each
            axis.

    Returns:
        tuple[int, int]: A tuple ``(shift_x, shift_y)`` representing the best
            integer translation found within the specified range.
    """
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
    """Apply a Gaussian blur, returning the original array when disabled.

    Converts the input array to RGB via PIL if necessary, applies a Gaussian
    blur filter, and returns the result as an unsigned 8-bit integer array.

    Args:
        image (ndarray): The source image array.
        radius (float): The sigma value for the Gaussian blur kernel. A value of
            zero or negative disables blurring and returns the original array.

    Returns:
        ndarray: The blurred image as an unsigned 8-bit integer array, or the
            original array if ``radius <= 0``.
    """
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
    """Calculate global SSIM and its two-dimensional similarity map.

    Computes the Structural Similarity Index (SSIM) between two images using
    skimage's implementation with Gaussian weighting for improved robustness.

    Args:
        original (ndarray): The reference image array.
        candidate (ndarray): The comparison image array.

    Returns:
        tuple[float, ndarray]: A tuple containing the global SSIM score as a
        float and the 2D similarity map as an unsigned 8-bit integer array. If
        the map is three-dimensional, it is averaged across channels before
        returning.
    """
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
    """Calculate mean SSIM values for sufficiently large image tiles.

    Divides the similarity map into non-overlapping blocks and computes the
    average SSIM value within each block that meets a minimum size threshold.

    Args:
        score_map (ndarray): The 2D structural similarity index map.
        tile (int): The side length in pixels of each square tile to evaluate.

    Returns:
        ndarray: A one-dimensional array containing the mean SSIM value for each
        valid tile, sorted by their top-left position. If no tiles meet the size
        threshold, returns a single-element array with the overall map mean.
    """
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
    """Save amplified differences and an SSIM similarity map for a failed page.

    Generates two diagnostic images in the specified output directory:
        1. An amplified difference image highlighting pixel-level discrepancies.
        2. A normalized SSIM similarity map visualizing regional agreement.

    Args:
        original (ndarray): The reference image array.
        candidate (ndarray): The comparison image array.
        score_map (ndarray): The structural similarity index map to save as an
            image.
        output (Path): The directory path where diagnostic images will be
            written.

    Raises:
        FileNotFoundError: If the output directory cannot be created due to
            insufficient permissions or a missing parent directory.
    """
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
    """Compare two rendered pages and return their perceptual result.

    Performs a multi-stage comparison between two image arrays to determine
    whether they are identical or similar within configured thresholds. The
    process includes pixel-identical checks, integer shift alignment, global
    SSIM computation, local tile-based statistics, and blurred SSIM evaluation.

    Args:
        original (ndarray): The reference page image array.
        candidate (ndarray): The comparison page image array.
        page (int): The page number for diagnostic file naming.
        options (ComparisonOptions): Configuration object containing thresholds
            for alignment, tile size, SSIM, local statistics, and bad fraction.
        failure_dir (Path | None, optional): Directory path to save diagnostic
            images if the comparison fails. Defaults to ``None``.

    Returns:
        PageResult: An instance containing all computed metrics including shift
            values, global and blurred SSIM scores, local statistics, and a
            final verdict string.
    """
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
