"""Unit tests for Sp1500FilingIngestor."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from zenpyre.ingestors import InMemoryIngestor

from glyphik.data.sec import SecFilingRecord
from glyphik.data.sp1500 import Company
from glyphik.ingestors import Sp1500FilingIngestor

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.ingestors.sp1500_filing"


@pytest.fixture
def companies() -> list[Company]:
    return [
        Company(
            ticker="AAPL",
            cik=320193,
            security="Apple Inc.",
            gics_sector="",
            gics_sub_industry="",
            index="S&P 500",
        ),
        Company(
            ticker="MSFT",
            cik=789019,
            security="Microsoft Corp.",
            gics_sector="",
            gics_sub_industry="",
            index="S&P 500",
        ),
    ]


@pytest.fixture
def records() -> list[SecFilingRecord]:
    return [
        SecFilingRecord.from_metadata({"ticker": "AAPL", "form": "10-K", "filepath": "tmp/a.pkl"}),
        SecFilingRecord.from_metadata({"ticker": "MSFT", "form": "10-K", "filepath": "tmp/b.pkl"}),
    ]


def _make_ingestor(company_ingestor: InMemoryIngestor, tmp_path: Path) -> Sp1500FilingIngestor:
    return Sp1500FilingIngestor(
        company_ingestor=company_ingestor,
        output_dir=tmp_path / "sec",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=["10-K", "10-Q"],
    )


######################################################
#     Tests for Sp1500FilingIngestor                 #
######################################################


# --- Constructor ---


def test_sp1500_filing_ingestor_stores_company_ingestor(tmp_path: Path) -> None:
    company_ingestor = InMemoryIngestor(data=[], copy=False)
    ingestor = _make_ingestor(company_ingestor, tmp_path)
    assert ingestor._company_ingestor is company_ingestor


def test_sp1500_filing_ingestor_stores_output_dir(tmp_path: Path) -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
    assert ingestor._output_dir == tmp_path / "sec"


def test_sp1500_filing_ingestor_accepts_str_output_dir(tmp_path: Path) -> None:
    ingestor = Sp1500FilingIngestor(
        company_ingestor=InMemoryIngestor(data=[]),
        output_dir=str(tmp_path / "sec"),
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=["10-K"],
    )
    assert ingestor._output_dir == tmp_path / "sec"


def test_sp1500_filing_ingestor_stores_start_date(tmp_path: Path) -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
    assert ingestor._start_date == date(2025, 1, 1)


def test_sp1500_filing_ingestor_stores_end_date(tmp_path: Path) -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
    assert ingestor._end_date == date(2026, 6, 1)


def test_sp1500_filing_ingestor_stores_forms(tmp_path: Path) -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
    assert ingestor._forms == ["10-K", "10-Q"]


def test_sp1500_filing_ingestor_repr_contains_class_name(tmp_path: Path) -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
    assert "Sp1500FilingIngestor" in repr(ingestor)


def test_sp1500_filing_ingestor_str_contains_class_name(tmp_path: Path) -> None:
    ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
    assert "Sp1500FilingIngestor" in str(ingestor)


# --- ingest ---


def test_sp1500_filing_ingestor_ingest_returns_list(tmp_path: Path) -> None:
    with patch(f"{MODULE}.load_or_fetch_filings", return_value=[]):
        result = _make_ingestor(InMemoryIngestor(data=[]), tmp_path).ingest()
    assert isinstance(result, list)


def test_sp1500_filing_ingestor_ingest_returns_records(
    tmp_path: Path, records: list[SecFilingRecord]
) -> None:
    with patch(f"{MODULE}.load_or_fetch_filings", return_value=records):
        result = _make_ingestor(InMemoryIngestor(data=[]), tmp_path).ingest()
    assert result == records


def test_sp1500_filing_ingestor_ingest_passes_companies_to_load_or_fetch(
    tmp_path: Path, companies: list[Company]
) -> None:
    with patch(f"{MODULE}.load_or_fetch_filings", return_value=[]) as mock_load:
        _make_ingestor(InMemoryIngestor(data=companies, copy=False), tmp_path).ingest()
    mock_load.assert_called_once_with(
        companies=companies,
        output_dir=tmp_path / "sec",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=["10-K", "10-Q"],
    )


def test_sp1500_filing_ingestor_ingest_can_be_called_multiple_times(
    tmp_path: Path, records: list[SecFilingRecord]
) -> None:
    with patch(f"{MODULE}.load_or_fetch_filings", return_value=records):
        ingestor = _make_ingestor(InMemoryIngestor(data=[]), tmp_path)
        assert ingestor.ingest() == ingestor.ingest()
