from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from glyphik.data.sec import CompanyIdentifier

MODULE = "glyphik.data.sec.company"


#######################################
#     Tests for CompanyIdentifier     #
#######################################


def test_company_identifier_is_frozen() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'ticker'"):
        identifier.ticker = "MSFT"


def test_company_identifier_stores_fields() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert identifier.cik == 320193
    assert identifier.ticker == "AAPL"


def test_company_identifier_equality() -> None:
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert identifier_a == identifier_b


def test_company_identifier_from_cik() -> None:
    with patch(f"{MODULE}.fetch_ticker_from_cik", return_value="AAPL") as mock_fetch:
        identifier = CompanyIdentifier.from_cik(320193)
    mock_fetch.assert_called_once_with(320193)
    assert identifier == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_company_identifier_from_cik_raises_when_ticker_not_found() -> None:
    with (
        patch(f"{MODULE}.fetch_ticker_from_cik", return_value=None),
        pytest.raises(ValueError, match=r"Cannot find ticker for CIK 320193"),
    ):
        CompanyIdentifier.from_cik(320193)


def test_company_identifier_from_ticker() -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=320193) as mock_fetch:
        identifier = CompanyIdentifier.from_ticker("AAPL")
    mock_fetch.assert_called_once_with("AAPL")
    assert identifier == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_company_identifier_from_ticker_raises_when_cik_not_found() -> None:
    with (
        patch(f"{MODULE}.fetch_cik_from_ticker", return_value=None),
        pytest.raises(ValueError, match=r"Cannot find CIK for ticker 'AAPL'"),
    ):
        CompanyIdentifier.from_ticker("AAPL")
