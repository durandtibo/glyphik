from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from glyphik.data.sec import CompanyIdentifier
from glyphik.data.sp1500 import Company, get_sp1500_company_identifiers

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.data.sp1500.company_identifier"


@pytest.fixture
def company_with_cik() -> Company:
    return Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware, Storage & Peripherals",
        index="S&P 500",
    )


@pytest.fixture
def company_without_cik() -> Company:
    return Company(
        ticker="XYZ",
        cik=None,
        security="Example Mid Corp",
        gics_sector="Industrials",
        gics_sub_industry="Industrial Machinery",
        index="S&P MidCap 400",
    )


################################################
#     Tests for get_sp1500_company_identifiers #
################################################


# --- path / find_missing_ciks pass-through ---


def test_get_sp1500_company_identifiers_default_path_is_none() -> None:
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]) as mock_load:
        get_sp1500_company_identifiers()

    mock_load.assert_called_once_with(path=None, find_missing_ciks=True)


def test_get_sp1500_company_identifiers_passes_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "sp1500.json"

    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]) as mock_load:
        get_sp1500_company_identifiers(path=path)

    mock_load.assert_called_once_with(path=path, find_missing_ciks=True)


def test_get_sp1500_company_identifiers_passes_str_path(tmp_path: Path) -> None:
    path_str = str(tmp_path / "sp1500.json")

    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]) as mock_load:
        get_sp1500_company_identifiers(path=path_str)

    mock_load.assert_called_once_with(path=path_str, find_missing_ciks=True)


def test_get_sp1500_company_identifiers_always_passes_find_missing_ciks_true() -> None:
    # Regression test: find_missing_ciks must be passed explicitly as
    # True, not left to load_or_fetch_sp1500_companies's own default --
    # this function's "every identifier has a real CIK" guarantee must
    # not silently depend on that default never changing.
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]) as mock_load:
        get_sp1500_company_identifiers()

    assert mock_load.call_args.kwargs["find_missing_ciks"] is True


# --- basic shape ---


def test_get_sp1500_company_identifiers_returns_list() -> None:
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]):
        result = get_sp1500_company_identifiers()
    assert isinstance(result, list)


def test_get_sp1500_company_identifiers_empty_result() -> None:
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]):
        result = get_sp1500_company_identifiers()
    assert result == []


# --- company with a cik: direct construction, no lookup ---


def test_get_sp1500_company_identifiers_company_with_cik_builds_identifier_directly(
    company_with_cik: Company,
) -> None:
    with (
        patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[company_with_cik]),
        patch.object(CompanyIdentifier, "from_ticker") as mock_from_ticker,
    ):
        result = get_sp1500_company_identifiers()

    assert result == [CompanyIdentifier(cik=320193, ticker="AAPL")]
    mock_from_ticker.assert_not_called()


def test_get_sp1500_company_identifiers_company_with_cik_does_not_log_warning(
    company_with_cik: Company, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[company_with_cik]),
        caplog.at_level("WARNING"),
    ):
        get_sp1500_company_identifiers()

    assert caplog.records == []


# --- company without a cik: from_ticker fallback ---


def test_get_sp1500_company_identifiers_company_without_cik_uses_from_ticker_fallback(
    company_without_cik: Company,
) -> None:
    fallback_identifier = CompanyIdentifier(cik=999999, ticker="XYZ")
    with (
        patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[company_without_cik]),
        patch.object(
            CompanyIdentifier, "from_ticker", return_value=fallback_identifier
        ) as mock_from_ticker,
    ):
        result = get_sp1500_company_identifiers()

    mock_from_ticker.assert_called_once_with("XYZ")
    assert result == [fallback_identifier]


def test_get_sp1500_company_identifiers_company_without_cik_logs_warning(
    company_without_cik: Company, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[company_without_cik]),
        patch.object(
            CompanyIdentifier, "from_ticker", return_value=CompanyIdentifier(cik=1, ticker="XYZ")
        ),
        caplog.at_level("INFO"),
    ):
        get_sp1500_company_identifiers()

    assert any("XYZ" in message for message in caplog.messages)


def test_get_sp1500_company_identifiers_propagates_from_ticker_value_error(
    company_without_cik: Company,
) -> None:
    with (
        patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[company_without_cik]),
        patch.object(
            CompanyIdentifier, "from_ticker", side_effect=ValueError("Cannot find CIK for ticker")
        ),
        pytest.raises(ValueError, match="Cannot find CIK for ticker"),
    ):
        get_sp1500_company_identifiers()


# --- mixed companies: order and correctness ---


def test_get_sp1500_company_identifiers_preserves_order_with_mixed_companies(
    company_with_cik: Company, company_without_cik: Company
) -> None:
    fallback_identifier = CompanyIdentifier(cik=999999, ticker="XYZ")
    with (
        patch(
            f"{MODULE}.load_or_fetch_sp1500_companies",
            return_value=[company_without_cik, company_with_cik],
        ),
        patch.object(CompanyIdentifier, "from_ticker", return_value=fallback_identifier),
    ):
        result = get_sp1500_company_identifiers()

    assert result == [
        fallback_identifier,
        CompanyIdentifier(cik=320193, ticker="AAPL"),
    ]
