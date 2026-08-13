"""Formalized verdict states for perceptual PDF comparison."""
from enum import Enum, auto


class Verdict(Enum):
    """Formalized verdict states for perceptual PDF comparison."""

    # Comparison is OK (equal or within thresholds).
    PASS = auto()

    # Comparison is not OK (exceeds thresholds).
    FAIL = auto()

    # Comparison could not be performed due to a page with a different size.
    FAIL_SIZE = auto()

    # Comparison could not be performed because the PDFs have different page
    # counts.
    DIFFERENT_PAGE_NUMBER = auto()  # Used when page counts don't match

    def __str__(self) -> str:
        """Return a string representation of the verdict.

        Returns:
            str: A string representation of the verdict, such as "PASS", "FAIL",
                "FAIL(size)" or "FAIL(page count)" for different page counts.
        """
        if self == Verdict.PASS:
            return "PASS"
        elif self == Verdict.FAIL:
            return "FAIL"
        elif self == Verdict.DIFFERENT_PAGE_NUMBER:
            return "FAIL(page count)"
        elif self == Verdict.FAIL_SIZE:
            return "FAIL(size)"

        return self.name.upper()

    @property
    def is_pass(self) -> bool:
        """Check if the verdict indicates a successful comparison."""
        return self == Verdict.PASS

    @property
    def is_fail(self) -> bool:
        """Check if the verdict indicates failure."""
        return self != Verdict.PASS
