from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from glyphik.data.sec import CompanyIdentifier
from glyphik.data_processors import EdgarCompanyToIdentifierProcessor

MODULE = "glyphik.data_processors.edgar_company_to_identifier"


@pytest.fixture
def company() -> Mock:
    mock_company = Mock(cik=320193)
    mock_company.get_ticker.return_value = "AAPL"
    return mock_company


######################################################
#   Tests for EdgarCompanyToIdentifierProcessor      #
######################################################


# --- Constructor ---


def test_edgar_company_to_identifier_processor_repr_contains_class_name() -> None:
    assert "EdgarCompanyToIdentifierProcessor" in repr(EdgarCompanyToIdentifierProcessor())


def test_edgar_company_to_identifier_processor_str_contains_class_name() -> None:
    assert "EdgarCompanyToIdentifierProcessor" in str(EdgarCompanyToIdentifierProcessor())


# --- process ---


def test_edgar_company_to_identifier_processor_process_returns_company_identifier(
    company: Mock,
) -> None:
    result = EdgarCompanyToIdentifierProcessor().process(company)
    assert isinstance(result, CompanyIdentifier)


def test_edgar_company_to_identifier_processor_process_with_ticker(company: Mock) -> None:
    result = EdgarCompanyToIdentifierProcessor().process(company)
    company.get_ticker.assert_called_once_with()
    assert result == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_edgar_company_to_identifier_processor_process_without_ticker(company: Mock) -> None:
    company.get_ticker.return_value = None
    with patch(
        "glyphik.data.sec.company_identifier.fetch_ticker_from_cik", return_value="AAPL"
    ) as mock_fetch:
        result = EdgarCompanyToIdentifierProcessor().process(company)
    mock_fetch.assert_called_once_with(320193)
    assert result == CompanyIdentifier(cik=320193, ticker="AAPL")


def test_edgar_company_to_identifier_processor_process_calls_from_edgar_company(
    company: Mock,
) -> None:
    with patch(
        f"{MODULE}.CompanyIdentifier.from_edgar_company",
        return_value=CompanyIdentifier(cik=320193, ticker="AAPL"),
    ) as mock_from_edgar:
        result = EdgarCompanyToIdentifierProcessor().process(company)
    mock_from_edgar.assert_called_once_with(company)
    assert result == CompanyIdentifier(cik=320193, ticker="AAPL")
