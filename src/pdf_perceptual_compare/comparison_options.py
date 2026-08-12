"""Thresholds and rendering-independent options for a page comparison."""

from dataclasses import dataclass


@dataclass
# pylint: disable=too-many-instance-attributes
class ComparisonOptions:
    """Thresholds and rendering-independent options for a page comparison.

    Attributes:
        tile (int): The number of tiles to use in the comparison. Defaults to
            128.
        blur (float): The blur factor applied during comparison.
            Defaults to 0.5.
        align (int): Alignment offset used for comparison. Defaults to 0.
        ssim (float): SSIM threshold value between 0.0 and 1.0. Defaults to
            0.995.
        ssim_blur (float): Blur factor applied during SSIM calculation.
            Defaults to 0.999.
        local_p01 (float): Local percentile threshold for pixel comparison.
            Defaults to 0.980.
        local_threshold (float): Local threshold value between 0.0 and 1.0.
            Defaults to 0.980.
        max_bad_fraction (float): Maximum fraction of bad pixels allowed.
            Defaults to 0.005.
    """

    tile: int = 128
    blur: float = 0.5
    align: int = 0
    ssim: float = 0.995
    ssim_blur: float = 0.999
    local_p01: float = 0.980
    local_threshold: float = 0.980
    max_bad_fraction: float = 0.005
