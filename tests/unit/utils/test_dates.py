from __future__ import annotations

from datetime import date

import pytest

from glyphik.utils.dates import coerce_to_date

#####################################
#     Tests for coerce_to_date      #
#####################################


# --- Pass-through ---


def test_coerce_to_date_returns_date_instance_as_is() -> None:
    d = date(2023, 1, 31)
    assert coerce_to_date(d) is d


def test_coerce_to_date_date_instance_returns_date() -> None:
    assert coerce_to_date(date(2023, 1, 31)) == date(2023, 1, 31)


# --- From string ---


def test_coerce_to_date_parses_iso_string() -> None:
    assert coerce_to_date("2023-01-31") == date(2023, 1, 31)


def test_coerce_to_date_from_string_returns_date_instance() -> None:
    assert isinstance(coerce_to_date("2023-01-31"), date)


def test_coerce_to_date_parses_start_of_year() -> None:
    assert coerce_to_date("2023-01-01") == date(2023, 1, 1)


def test_coerce_to_date_parses_end_of_year() -> None:
    assert coerce_to_date("2023-12-31") == date(2023, 12, 31)


def test_coerce_to_date_parses_leap_day() -> None:
    assert coerce_to_date("2024-02-29") == date(2024, 2, 29)


# --- Invalid input ---


def test_coerce_to_date_invalid_format_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid isoformat string"):
        coerce_to_date("01/31/2023")


def test_coerce_to_date_invalid_date_raises_value_error() -> None:
    # The exact wording differs between CPython's C-accelerated ``_datetime``
    # module and the pure-Python fallback (e.g. PyPy, free-threaded builds),
    # so match on both known variants instead of one exact string.
    with pytest.raises(
        ValueError, match=r"day is out of range for month|day \d+ must be in range"
    ):
        coerce_to_date("2023-02-30")


def test_coerce_to_date_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid isoformat string"):
        coerce_to_date("")
