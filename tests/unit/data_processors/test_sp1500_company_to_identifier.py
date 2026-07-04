from __future__ import annotations

from unittest.mock import patch

import pytest

from glyphik.data.sec import CompanyIdentifier
from glyphik.data.sp1500 import Company
from glyphik.data_processors import Sp1500CompanyToIdentifierProcessor

MODULE = "glyphik.data_processors.sp1500_company_to_identifier"


@pytest.fixture
def company() -> Company:
    return Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
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


######################################################
#   Tests for Sp1500CompanyToIdentifierProcessor     #
######################################################


# --- Constructor ---


def test_sp1500_company_to_identifier_processor_repr_contains_class_name() -> None:
    assert "Sp1500CompanyToIdentifierProcessor" in repr(Sp1500CompanyToIdentifierProcessor())


def test_sp1500_company_to_identifier_processor_str_contains_class_name() -> None:
    assert "Sp1500CompanyToIdentifierProcessor" in str(Sp1500CompanyToIdentifierProcessor())


# --- process ---


def test_sp1500_company_to_identifier_processor_process_returns_company_identifier(
    company: Company,
) -> None:
    result = Sp1500CompanyToIdentifierProcessor().process(company)
    assert isinstance(result, CompanyIdentifier)


def test_sp1500_company_to_identifier_processor_process_with_cik(company: Company) -> None:
    result = Sp1500CompanyToIdentifierProcessor().process(company)
    assert result == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_sp1500_company_to_identifier_processor_process_with_cik_does_not_call_from_ticker(
    company: Company,
) -> None:
    with patch(f"{MODULE}.CompanyIdentifier.from_ticker") as mock_from_ticker:
        Sp1500CompanyToIdentifierProcessor().process(company)
    mock_from_ticker.assert_not_called()


def test_sp1500_company_to_identifier_processor_process_without_cik_calls_from_ticker(
    company_without_cik: Company,
) -> None:
    with patch(
        f"{MODULE}.CompanyIdentifier.from_ticker",
        return_value=CompanyIdentifier(cik=999999, ticker="XYZ"),
    ) as mock_from_ticker:
        result = Sp1500CompanyToIdentifierProcessor().process(company_without_cik)
    mock_from_ticker.assert_called_once_with("XYZ")
    assert result == CompanyIdentifier(cik=999999, ticker="XYZ")


def test_sp1500_company_to_identifier_processor_process_without_cik_raises_when_ticker_not_found(
    company_without_cik: Company,
) -> None:
    with (
        patch(f"{MODULE}.CompanyIdentifier.from_ticker", side_effect=ValueError("not found")),
        pytest.raises(ValueError, match=r"not found"),
    ):
        Sp1500CompanyToIdentifierProcessor().process(company_without_cik)
