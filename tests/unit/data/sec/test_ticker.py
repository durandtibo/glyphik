from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from glyphik.data.sec import fetch_cik_from_ticker
from glyphik.data.sec.ticker import _fetch_cik_from_ticker
from glyphik.testing.fixtures import edgar_available
from glyphik.utils.imports import is_edgar_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_edgar_available():
    from edgar import Company
    from edgar.entity import CompanyNotFoundError

MODULE = "glyphik.data.sec.ticker"


@pytest.fixture(autouse=True)
def clear_cache() -> Generator[None]:
    """Clear the LRU cache before each test to ensure isolation."""
    _fetch_cik_from_ticker.cache_clear()
    yield
    _fetch_cik_from_ticker.cache_clear()


def make_mock_company(cik: int | None) -> MagicMock:
    company = MagicMock(spec=Company)
    company.cik = cik
    return company


##################################################
#   Tests for fetch_cik_from_ticker              #
##################################################

# --- return value ---


@edgar_available
def test_fetch_cik_from_ticker_returns_cik() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)):
        assert fetch_cik_from_ticker("AAPL") == 320193


@edgar_available
def test_fetch_cik_from_ticker_returns_none_when_company_not_found() -> None:
    with patch(f"{MODULE}.Company", side_effect=CompanyNotFoundError("AAPL")):
        assert fetch_cik_from_ticker("AAPL") is None


# --- case insensitivity ---


@edgar_available
def test_fetch_cik_from_ticker_accepts_uppercase() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)):
        assert fetch_cik_from_ticker("AAPL") == 320193


@edgar_available
def test_fetch_cik_from_ticker_accepts_lowercase() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)):
        assert fetch_cik_from_ticker("aapl") == 320193


@edgar_available
def test_fetch_cik_from_ticker_uppercase_and_lowercase_return_same_result() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)):
        assert fetch_cik_from_ticker("AAPL") == fetch_cik_from_ticker("aapl")


@edgar_available
def test_fetch_cik_from_ticker_passes_uppercased_ticker_to_company() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)) as mock_company:
        fetch_cik_from_ticker("aapl")
    mock_company.assert_called_once_with("AAPL")


# --- caching ---


@edgar_available
def test_fetch_cik_from_ticker_caches_result() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)) as mock_company:
        fetch_cik_from_ticker("AAPL")
        fetch_cik_from_ticker("AAPL")
    mock_company.assert_called_once()


@edgar_available
def test_fetch_cik_from_ticker_uppercase_and_lowercase_share_cache_entry() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)) as mock_company:
        fetch_cik_from_ticker("AAPL")
        fetch_cik_from_ticker("aapl")
    mock_company.assert_called_once()


@edgar_available
def test_fetch_cik_from_ticker_different_tickers_cached_separately() -> None:
    with patch(f"{MODULE}.Company", return_value=make_mock_company(320193)) as mock_company:
        fetch_cik_from_ticker("AAPL")
        fetch_cik_from_ticker("MSFT")
    assert mock_company.call_count == 2


def test_fetch_cik_from_ticker_cache_size_is_256() -> None:
    assert _fetch_cik_from_ticker.cache_info().maxsize == 256
