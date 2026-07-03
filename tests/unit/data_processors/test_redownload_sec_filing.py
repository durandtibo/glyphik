from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from glyphik.data.sec import SecFilingRecord
from glyphik.data_processors import RedownloadSecFilingProcessor
from glyphik.testing.fixtures import edgar_available

MODULE = "glyphik.data_processors.redownload_sec_filing"


@pytest.fixture
def record() -> SecFilingRecord:
    return SecFilingRecord.from_metadata(
        {
            "filepath": "tmp/test.pkl",
            "cik": 320193,
            "form": "10-K",
            "accession_no": "0000320193-24-000123",
        }
    )


######################################################
#     Tests for RedownloadSecFilingProcessor         #
######################################################


# --- repr / str ---


def test_redownload_sec_filing_processor_repr_contains_class_name() -> None:
    processor = RedownloadSecFilingProcessor()
    assert "RedownloadSecFilingProcessor" in repr(processor)


def test_redownload_sec_filing_processor_str_contains_class_name() -> None:
    processor = RedownloadSecFilingProcessor()
    assert "RedownloadSecFilingProcessor" in str(processor)


# --- process: filing already loadable ---


@edgar_available
def test_redownload_sec_filing_processor_process_returns_same_record_when_loadable(
    record: SecFilingRecord,
) -> None:
    with (
        patch.object(SecFilingRecord, "load_filing", return_value=MagicMock()),
        patch(f"{MODULE}.Company") as mock_company,
    ):
        result = RedownloadSecFilingProcessor().process(record)
    assert result is record
    mock_company.assert_not_called()


# --- process: filing missing, unique match found ---


@edgar_available
def test_redownload_sec_filing_processor_process_redownloads_when_not_loadable(
    record: SecFilingRecord,
) -> None:
    mock_filing = MagicMock()
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 1
    mock_filings.__getitem__.return_value = mock_filing
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.Company", return_value=mock_company) as mock_company_cls,
    ):
        result = RedownloadSecFilingProcessor().process(record)

    mock_company_cls.assert_called_once_with(320193)
    mock_filing.save.assert_called_once_with("tmp/test.pkl")
    assert result is record


@edgar_available
def test_redownload_sec_filing_processor_process_filters_by_form_and_accession_no(
    record: SecFilingRecord,
) -> None:
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 1
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        RedownloadSecFilingProcessor().process(record)

    mock_company.get_filings.assert_called_once_with(form="10-K")
    mock_company.get_filings.return_value.filter.assert_called_once_with(
        accession_number="0000320193-24-000123"
    )


# --- process: filing missing, no unique match ---


@edgar_available
def test_redownload_sec_filing_processor_process_no_match_does_not_save(
    record: SecFilingRecord,
) -> None:
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 0
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        result = RedownloadSecFilingProcessor().process(record)

    assert result is record


@edgar_available
def test_redownload_sec_filing_processor_process_multiple_matches_does_not_save(
    record: SecFilingRecord,
) -> None:
    mock_filing_a = MagicMock()
    mock_filing_b = MagicMock()
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 2
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        RedownloadSecFilingProcessor().process(record)

    mock_filing_a.save.assert_not_called()
    mock_filing_b.save.assert_not_called()


@edgar_available
def test_redownload_sec_filing_processor_process_returns_record_unchanged_on_no_match(
    record: SecFilingRecord,
) -> None:
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 0
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        result = RedownloadSecFilingProcessor().process(record)

    assert result.metadata == record.metadata
