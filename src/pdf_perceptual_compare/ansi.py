"""ANSI escape codes for terminal colorization."""

from dataclasses import dataclass


@dataclass
class Ansi:
    """A dataclass representing ANSI escape codes for terminal colorization.

    Attributes:
        RED (str): ANSI escape code for red text.
        RESET (str): ANSI escape code to reset text formatting.
    """

    RED: str = "\033[31m"
    RESET: str = "\033[0m"
