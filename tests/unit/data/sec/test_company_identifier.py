from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest

from glyphik.data.sec import CompanyIdentifier

MODULE = "glyphik.data.sec.company_identifier"


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


def test_company_identifier_from_edgar_company_with_ticker() -> None:
    mock_company = Mock(cik=320193)
    mock_company.get_ticker.return_value = "AAPL"
    identifier = CompanyIdentifier.from_edgar_company(mock_company)
    mock_company.get_ticker.assert_called_once_with()
    assert identifier == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_company_identifier_from_edgar_company_without_ticker() -> None:
    mock_company = Mock(cik=320193)
    mock_company.get_ticker.return_value = None
    with patch(f"{MODULE}.fetch_ticker_from_cik", return_value="AAPL") as mock_fetch:
        identifier = CompanyIdentifier.from_edgar_company(mock_company)
    mock_fetch.assert_called_once_with(320193)
    assert identifier == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_company_identifier_from_edgar_company_without_ticker_raises_when_cik_not_found() -> None:
    mock_company = Mock(cik=320193)
    mock_company.get_ticker.return_value = None
    with (
        patch(f"{MODULE}.fetch_ticker_from_cik", return_value=None),
        pytest.raises(ValueError, match=r"Cannot find ticker for CIK 320193"),
    ):
        CompanyIdentifier.from_edgar_company(mock_company)


###################################################
#     Tests for CompanyIdentifier.content_hash     #
###################################################


def test_company_identifier_content_hash_returns_str() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert isinstance(identifier.content_hash(), str)


def test_company_identifier_content_hash_default_length() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert len(identifier.content_hash()) == 64


def test_company_identifier_content_hash_custom_length() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert len(identifier.content_hash(length=16)) == 16


def test_company_identifier_content_hash_is_deterministic() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert identifier.content_hash() == identifier.content_hash()


def test_company_identifier_content_hash_equal_for_equal_identifiers() -> None:
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert identifier_a.content_hash() == identifier_b.content_hash()


def test_company_identifier_content_hash_differs_for_different_cik() -> None:
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=789019, ticker="AAPL")
    assert identifier_a.content_hash() != identifier_b.content_hash()


def test_company_identifier_content_hash_differs_for_different_ticker() -> None:
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=320193, ticker="MSFT")
    assert identifier_a.content_hash() != identifier_b.content_hash()


def test_company_identifier_content_hash_matches_known_value() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert (
        identifier.content_hash()
        == "0865aae7780371d4287740b7f7659ebf1619e61b7667b8aa2571b1488c9894d2"
    )


def test_company_identifier_content_hash_delegates_to_hash_string() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    with patch(f"{MODULE}.hash_string", return_value="mock-hash") as mock_hash_string:
        result = identifier.content_hash(length=16)
    mock_hash_string.assert_called_once_with('{"cik": 320193, "ticker": "AAPL"}', length=16)
    assert result == "mock-hash"


def test_company_identifier_content_hash_invalid_length_raises() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    with pytest.raises(ValueError, match=r"length must be an even number between 2 and 128"):
        identifier.content_hash(length=1)


def test_company_identifier_content_hash_differs_from_builtin_hash() -> None:
    # content_hash is a deterministic content digest, distinct from the
    # dataclass-generated __hash__ (which is an int, not a hex string,
    # and isn't guaranteed to be stable across processes).
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert identifier.content_hash() != hash(identifier)
