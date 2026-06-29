r"""Provide standard SEC form type labels."""

from __future__ import annotations

__all__ = ["SecForm"]

from enum import StrEnum


class SecForm(StrEnum):
    """Standard SEC form type labels.

    Each member is a string equal to the official SEC form name, so
    instances can be compared directly with strings:

    Example:
        ```pycon
        >>> from glyphik.data.sec import SecForm
        >>> SecForm.TEN_K == "10-K"
        True
        >>> "10-K" in SecForm._value2member_map_
        True

        ```
    """

    EIGHT_K = "8-K"
    TEN_K = "10-K"
    TEN_Q = "10-Q"
