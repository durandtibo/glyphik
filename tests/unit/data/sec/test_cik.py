from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from glyphik.data.sec import fetch_ticker_from_cik
from glyphik.data.sec.cik import _fetch_ticker_from_cik
from glyphik.testing.fixtures import edgar_available
from glyphik.utils.imports import is_edgar_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_edgar_available():
    from edgar import Company
    from edgar.entity import CompanyNotFoundError

MODULE = "glyphik.data.sec.cik"


@pytest.fixture(autouse=True)
def clear_cache() -> Generator[None]:
    """Clear the LRU cache before each test to ensure isolation."""
    _fetch_ticker_from_cik.cache_clear()
    yield
    _fetch_ticker_from_cik.cache_clear()


def make_mock_company(ticker: str | None) -> MagicMock:
    company = MagicMock(spec=Company)
    company.get_ticker.return_value = ticker
    return company


##################################################
#   Tests for fetch_ticker_from_cik              #
##################################################

# --- return value ---


@edgar_available
def test_fetch_ticker_from_cik_returns_primary_ticker() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")):
        assert fetch_ticker_from_cik(320193) == "AAPL"


@edgar_available
def test_fetch_ticker_from_cik_returns_none_when_no_ticker() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(None)):
        assert fetch_ticker_from_cik(320193) is None


@edgar_available
def test_fetch_ticker_from_cik_returns_none_when_company_not_found() -> None:
    with patch(f"{MODULE}.Company", side_effect=CompanyNotFoundError(320193)):
        assert fetch_ticker_from_cik(320193) is None


# --- int and str input ---


@edgar_available
def test_fetch_ticker_from_cik_accepts_int() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")):
        assert fetch_ticker_from_cik(320193) == "AAPL"


@edgar_available
def test_fetch_ticker_from_cik_accepts_str() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")):
        assert fetch_ticker_from_cik("320193") == "AAPL"


@edgar_available
def test_fetch_ticker_from_cik_int_and_str_return_same_result() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")):
        assert fetch_ticker_from_cik(320193) == fetch_ticker_from_cik("320193")


# --- caching ---


@edgar_available
def test_fetch_ticker_from_cik_caches_result() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")) as mock_company:
        fetch_ticker_from_cik(320193)
        fetch_ticker_from_cik(320193)
    mock_company.assert_called_once()


@edgar_available
def test_fetch_ticker_from_cik_int_and_str_share_cache_entry() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")) as mock_company:
        fetch_ticker_from_cik(320193)
        fetch_ticker_from_cik("320193")
    mock_company.assert_called_once()


@edgar_available
def test_fetch_ticker_from_cik_different_ciks_cached_separately() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company("AAPL")) as mock_company:
        fetch_ticker_from_cik(320193)
        fetch_ticker_from_cik(789019)
    assert mock_company.call_count == 2


def test_fetch_ticker_from_cik_cache_size_is_256() -> None:
    assert _fetch_ticker_from_cik.cache_info().maxsize == 256
