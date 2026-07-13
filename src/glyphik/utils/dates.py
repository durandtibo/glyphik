r"""Contain utility functions for working with dates."""

from __future__ import annotations

__all__ = ["coerce_to_date"]

from datetime import date


def coerce_to_date(value: date | str) -> date:
    """Coerce a value to a :class:`~datetime.date` instance.

    If ``value`` is already a :class:`~datetime.date` instance, it is
    returned as-is. If it is a :class:`str`, it is parsed as an
    ISO 8601 formatted date (e.g. ``"2023-01-31"``) via
    :meth:`datetime.date.fromisoformat`.

    Args:
        value: Either a :class:`~datetime.date` instance, or a
            :class:`str` containing an ISO 8601 formatted date.

    Returns:
        The coerced :class:`~datetime.date` instance.

    Raises:
        ValueError: If ``value`` is a :class:`str` that is not a
            valid ISO 8601 formatted date.

    Example:
        ```pycon
        >>> from datetime import date
        >>> from glyphik.utils.dates import coerce_to_date
        >>> # From an existing date instance:
        >>> coerce_to_date(date(2023, 1, 31))
        datetime.date(2023, 1, 31)
        >>> # From an ISO-formatted string:
        >>> coerce_to_date("2023-01-31")
        datetime.date(2023, 1, 31)

        ```
    """
    return value if isinstance(value, date) else date.fromisoformat(value)
