from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

import pytest
from coola.hashing import HasherRegistry, hash_object

from glyphik.data.sec import CompanyIdentifier
from glyphik.data.sec.company_identifier import (
    CompanyIdentifierHasher,
    get_company_identifiers_from_tickers,
)

MODULE = "glyphik.data.sec.company_identifier"


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry()


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


def test_company_identifier_strips_ticker() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker=" AAPL")
    assert identifier.ticker == "AAPL"


def test_company_identifier_upper_ticker() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="aapl")
    assert identifier.ticker == "AAPL"


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


##################################################
#     Tests for get_company_identifiers_from_tickers  #
##################################################


# --- Default (fail-fast) behavior ---


def test_get_company_identifiers_from_tickers_returns_resolved_identifiers() -> None:
    identifiers = [
        CompanyIdentifier(cik=320193, ticker="AAPL"),
        CompanyIdentifier(cik=789019, ticker="MSFT"),
    ]
    with patch(f"{MODULE}.CompanyIdentifier.from_ticker", side_effect=identifiers) as mock_from:
        result = get_company_identifiers_from_tickers(["AAPL", "MSFT"])
        assert result == identifiers
        assert mock_from.call_count == 2


def test_get_company_identifiers_from_tickers_preserves_order() -> None:
    identifiers = [
        CompanyIdentifier(cik=1, ticker="A"),
        CompanyIdentifier(cik=2, ticker="B"),
        CompanyIdentifier(cik=3, ticker="C"),
    ]
    with patch(f"{MODULE}.CompanyIdentifier.from_ticker", side_effect=identifiers):
        result = get_company_identifiers_from_tickers(["A", "B", "C"])
        assert [ci.ticker for ci in result] == ["A", "B", "C"]


def test_get_company_identifiers_from_tickers_empty_list_returns_empty_list() -> None:
    with patch(f"{MODULE}.CompanyIdentifier.from_ticker") as mock_from:
        result = get_company_identifiers_from_tickers([])
        assert result == []
        mock_from.assert_not_called()


def test_get_company_identifiers_from_tickers_raises_on_unresolved_ticker() -> None:
    with (
        patch(
            f"{MODULE}.CompanyIdentifier.from_ticker",
            side_effect=ValueError("Cannot find CIK for ticker 'BAD'"),
        ),
        pytest.raises(ValueError, match=r"Cannot find CIK for ticker"),
    ):
        get_company_identifiers_from_tickers(["BAD"])


def test_get_company_identifiers_from_tickers_default_does_not_skip_unresolved() -> None:
    good = CompanyIdentifier(cik=320193, ticker="AAPL")
    with (
        patch(
            f"{MODULE}.CompanyIdentifier.from_ticker",
            side_effect=[good, ValueError("Cannot find CIK for ticker 'BAD'")],
        ),
        pytest.raises(ValueError, match=r"Cannot find CIK for ticker"),
    ):
        get_company_identifiers_from_tickers(["AAPL", "BAD"])


# --- skip_unresolved=True behavior ---


def test_get_company_identifiers_from_tickers_skip_unresolved_omits_bad_tickers() -> None:
    good = CompanyIdentifier(cik=320193, ticker="AAPL")
    with patch(
        f"{MODULE}.CompanyIdentifier.from_ticker",
        side_effect=[good, ValueError("Cannot find CIK for ticker 'BAD'")],
    ):
        result = get_company_identifiers_from_tickers(["AAPL", "BAD"], skip_unresolved=True)
        assert result == [good]


def test_get_company_identifiers_from_tickers_skip_unresolved_all_bad_returns_empty() -> None:
    with patch(
        f"{MODULE}.CompanyIdentifier.from_ticker",
        side_effect=ValueError("Cannot find CIK for ticker"),
    ):
        result = get_company_identifiers_from_tickers(["BAD1", "BAD2"], skip_unresolved=True)
        assert result == []


def test_get_company_identifiers_from_tickers_skip_unresolved_preserves_relative_order() -> None:
    good1 = CompanyIdentifier(cik=1, ticker="A")
    good2 = CompanyIdentifier(cik=2, ticker="C")
    with patch(
        f"{MODULE}.CompanyIdentifier.from_ticker",
        side_effect=[good1, ValueError("bad"), good2],
    ):
        result = get_company_identifiers_from_tickers(["A", "B", "C"], skip_unresolved=True)
        assert result == [good1, good2]


def test_get_company_identifiers_from_tickers_skip_unresolved_calls_from_ticker_for_all() -> None:
    with patch(
        f"{MODULE}.CompanyIdentifier.from_ticker",
        side_effect=[ValueError("bad"), CompanyIdentifier(cik=2, ticker="B")],
    ) as mock_from:
        get_company_identifiers_from_tickers(["A", "B"], skip_unresolved=True)
        assert mock_from.call_count == 2


#############################################
#     Tests for CompanyIdentifierHasher     #
#############################################


def test_company_identifier_hasher_repr() -> None:
    assert repr(CompanyIdentifierHasher()) == "CompanyIdentifierHasher()"


def test_company_identifier_hasher_str() -> None:
    assert str(CompanyIdentifierHasher()) == "CompanyIdentifierHasher()"


def test_company_identifier_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert isinstance(hasher.hash(identifier, registry=registry), str)


def test_company_identifier_hasher_hash_default_length(registry: HasherRegistry) -> None:
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert len(hasher.hash(identifier, registry=registry)) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min-valid"),
        pytest.param(32, id="middle"),
        pytest.param(64, id="default"),
        pytest.param(128, id="max-valid"),
    ],
)
def test_company_identifier_hasher_hash_length(registry: HasherRegistry, length: int) -> None:
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert len(hasher.hash(identifier, registry=registry, length=length)) == length


def test_company_identifier_hasher_hash_same_identifier_same_hash(
    registry: HasherRegistry,
) -> None:
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hasher.hash(identifier, registry=registry) == hasher.hash(identifier, registry=registry)


def test_company_identifier_hasher_hash_equal_identifiers_same_hash(
    registry: HasherRegistry,
) -> None:
    hasher = CompanyIdentifierHasher()
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hasher.hash(identifier_a, registry=registry) == hasher.hash(
        identifier_b, registry=registry
    )


def test_company_identifier_hasher_hash_different_cik_different_hash(
    registry: HasherRegistry,
) -> None:
    hasher = CompanyIdentifierHasher()
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=789019, ticker="AAPL")
    assert hasher.hash(identifier_a, registry=registry) != hasher.hash(
        identifier_b, registry=registry
    )


def test_company_identifier_hasher_hash_different_ticker_different_hash(
    registry: HasherRegistry,
) -> None:
    hasher = CompanyIdentifierHasher()
    identifier_a = CompanyIdentifier(cik=320193, ticker="AAPL")
    identifier_b = CompanyIdentifier(cik=320193, ticker="MSFT")
    assert hasher.hash(identifier_a, registry=registry) != hasher.hash(
        identifier_b, registry=registry
    )


def test_company_identifier_hasher_hash_matches_content_hash(registry: HasherRegistry) -> None:
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hasher.hash(identifier, registry=registry) == identifier.content_hash()


def test_company_identifier_hasher_hash_matches_content_hash_with_length(
    registry: HasherRegistry,
) -> None:
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hasher.hash(identifier, registry=registry, length=32) == identifier.content_hash(
        length=32
    )


def test_company_identifier_hasher_hash_ignores_registry(registry: HasherRegistry) -> None:
    """The registry is accepted for interface compatibility but should
    not affect the resulting hash, since CompanyIdentifierHasher
    delegates entirely to CompanyIdentifier.content_hash."""
    hasher = CompanyIdentifierHasher()
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hasher.hash(identifier, registry=HasherRegistry()) == hasher.hash(
        identifier, registry=registry
    )


#################################
#     Tests for hash_object     #
#################################


def test_hash_object_company_identifier() -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hash_object(identifier) == identifier.content_hash()


@pytest.mark.parametrize("length", [16, 32])
def test_hash_object_company_identifier_length(length: int) -> None:
    identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
    assert hash_object(identifier, length=length) == identifier.content_hash(length=length)
