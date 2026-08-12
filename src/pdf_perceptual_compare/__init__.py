"""Perceptual, page-by-page PDF comparison."""

from .comparator import compare_page
from .page_result import PageResult

__all__ = ["PageResult", "compare_page"]
