"""Unit tests for perceptual page comparison."""

from numpy import uint8, zeros

from pdf_perceptual_compare.comparator import (
    ComparisonOptions,
    compare_page,
    crop_for_shift
)
from pdf_perceptual_compare.verdict import Verdict


def test_compare_page_accepts_identical_images() -> None:
    """Verify that compare_page returns PASS for identical images."""
    image = zeros((16, 16, 3), dtype=uint8)

    result = compare_page(
        image,
        image.copy(),
        page=1,
        options=ComparisonOptions()
    )

    assert result.verdict.is_pass
    assert result.identical is True
    assert result.ssim == 1.0


def test_compare_page_rejects_different_dimensions() -> None:
    """Verify that compare_page returns FAIL for mismatched dimensions."""
    result = compare_page(
        zeros((16, 16, 3), dtype=uint8),
        zeros((15, 16, 3), dtype=uint8),
        page=1,
        options=ComparisonOptions(),
    )

    assert result.verdict == Verdict.FAIL_SIZE


def test_crop_for_shift_returns_matching_overlap() -> None:
    """Verify that crop_for_shift returns images with matching overlap."""
    image = zeros((10, 10, 3), dtype=uint8)

    shifted_a, shifted_b = crop_for_shift(image, image, shift_x=2, shift_y=-1)

    assert shifted_a.shape == shifted_b.shape == (9, 8, 3)
