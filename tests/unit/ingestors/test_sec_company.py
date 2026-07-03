"""Unit tests for SecCompanyIngestor."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from zenpyre.ingestors import InMemoryIngestor

from glyphik.ingestors import SecCompanyIngestor

MODULE = "glyphik.ingestors.sec_company"


@pytest.fixture
def ciks_or_tickers() -> list[str | int]:
    return ["AAPL", 789019]


def _make_ingestor(company_ingestor: InMemoryIngestor) -> SecCompanyIngestor:
    return SecCompanyIngestor(ingestor=company_ingestor)


######################################################
#     Tests for SecCompanyIngestor                   #
######################################################


# --- Constructor ---


def test_sec_company_ingestor_stores_ingestor() -> None:
    company_ingestor = InMemoryIngestor(data=[], copy=False)
    ingestor = _make_ingestor(company_ingestor)
    assert ingestor._ingestor is company_ingestor


def test_sec_company_ingestor_repr_contains_class_name() -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]))
    assert "SecCompanyIngestor" in repr(ingestor)


def test_sec_company_ingestor_str_contains_class_name() -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]))
    assert "SecCompanyIngestor" in str(ingestor)


# --- ingest ---


def test_sec_company_ingestor_ingest_returns_list() -> None:
    with patch(f"{MODULE}.Company"):
        result = _make_ingestor(InMemoryIngestor(data=[])).ingest()
    assert isinstance(result, list)


def test_sec_company_ingestor_ingest_returns_empty_list_for_empty_input() -> None:
    with patch(f"{MODULE}.Company"):
        result = _make_ingestor(InMemoryIngestor(data=[])).ingest()
    assert result == []


def test_sec_company_ingestor_ingest_builds_one_company_per_input(
    ciks_or_tickers: list[str | int],
) -> None:
    with patch(f"{MODULE}.Company") as mock_company:
        result = _make_ingestor(InMemoryIngestor(data=ciks_or_tickers, copy=False)).ingest()
    assert len(result) == len(ciks_or_tickers)
    assert mock_company.call_count == len(ciks_or_tickers)


def test_sec_company_ingestor_ingest_passes_cik_or_ticker_to_company(
    ciks_or_tickers: list[str | int],
) -> None:
    with patch(f"{MODULE}.Company") as mock_company:
        _make_ingestor(InMemoryIngestor(data=ciks_or_tickers, copy=False)).ingest()
    mock_company.assert_any_call("AAPL")
    mock_company.assert_any_call(789019)


def test_sec_company_ingestor_ingest_returns_company_instances(
    ciks_or_tickers: list[str | int],
) -> None:
    with patch(f"{MODULE}.Company") as mock_company:
        mock_company.side_effect = lambda cik_or_ticker: cik_or_ticker
        result = _make_ingestor(InMemoryIngestor(data=ciks_or_tickers, copy=False)).ingest()
    assert result == ciks_or_tickers


def test_sec_company_ingestor_ingest_can_be_called_multiple_times(
    ciks_or_tickers: list[str | int],
) -> None:
    with patch(f"{MODULE}.Company"):
        ingestor = _make_ingestor(InMemoryIngestor(data=ciks_or_tickers, copy=False))
        assert len(ingestor.ingest()) == len(ingestor.ingest())
