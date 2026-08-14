"""Per-page visual-comparison result."""

from dataclasses import dataclass

from .verdict import Verdict


@dataclass
# pylint: disable=too-many-instance-attributes
class PageResult:
    """A dataclass representing the per-page visual comparison result between
    two images.

    Attributes:
        page (int): The page number for which this result applies.
        identical (bool): Whether the two pages are perceptually identical
            according to the configured thresholds.
        shift_x (int): Horizontal pixel shift between the two pages.
        shift_y (int): Vertical pixel shift between the two pages.
        ssim (float): Structural Similarity Index Measure value, closer to 1
            indicates higher similarity.
        ssim_blur (float): Blurred SSIM value for robustness against minor
            misalignments.
        local_p01 (float): Local percentile-01 metric indicating outlier pixel
            differences.
        local_min (float): Minimum pixel difference between the two pages in
            this region.
        local_bad_fraction (float): Fraction of pixels that differ
            significantly from expected values.
        verdict (Verdict): A textual summary of the comparison result,
            Verdict.PASS, Verdict.FAIL or Verdict.FAIL_SIZE.
    """

    page: int
    identical: bool
    shift_x: int
    shift_y: int
    ssim: float
    ssim_blur: float
    local_p01: float
    local_min: float
    local_bad_fraction: float
    verdict: Verdict
