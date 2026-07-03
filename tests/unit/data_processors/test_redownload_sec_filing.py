"""Unit tests for RedownloadSecFilingProcessor."""

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


def _mock_company_with_match(mock_filing: MagicMock) -> MagicMock:
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 1
    mock_filings.__getitem__.return_value = mock_filing
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings
    return mock_company


######################################################
#     Tests for RedownloadSecFilingProcessor         #
######################################################


# --- Constructor ---


@edgar_available
def test_redownload_sec_filing_processor_stores_check_sgml_default() -> None:
    processor = RedownloadSecFilingProcessor()
    assert processor._check_sgml is False


@edgar_available
def test_redownload_sec_filing_processor_stores_check_sgml_true() -> None:
    processor = RedownloadSecFilingProcessor(check_sgml=True)
    assert processor._check_sgml is True


@edgar_available
def test_redownload_sec_filing_processor_repr() -> None:
    assert repr(RedownloadSecFilingProcessor()).startswith("RedownloadSecFilingProcessor(")


@edgar_available
def test_redownload_sec_filing_processor_str() -> None:
    assert str(RedownloadSecFilingProcessor()).startswith("RedownloadSecFilingProcessor(")


# --- process: check_sgml=False (default) ---


@edgar_available
def test_redownload_sec_filing_processor_default_does_not_check_sgml(
    record: SecFilingRecord,
) -> None:
    mock_filing = MagicMock()
    with (
        patch.object(SecFilingRecord, "load_filing", return_value=mock_filing),
        patch(f"{MODULE}.has_valid_sgml") as mock_has_valid_sgml,
        patch(f"{MODULE}.Company") as mock_company,
    ):
        result = RedownloadSecFilingProcessor(check_sgml=False).process(record)

    mock_has_valid_sgml.assert_not_called()
    mock_company.assert_not_called()
    assert result is record


# --- process: check_sgml=True, filing loadable ---


@edgar_available
def test_redownload_sec_filing_processor_check_sgml_valid_does_not_redownload(
    record: SecFilingRecord,
) -> None:
    mock_filing = MagicMock()
    with (
        patch.object(SecFilingRecord, "load_filing", return_value=mock_filing),
        patch(f"{MODULE}.has_valid_sgml", return_value=True) as mock_has_valid_sgml,
        patch(f"{MODULE}.Company") as mock_company,
    ):
        result = RedownloadSecFilingProcessor(check_sgml=True).process(record)

    mock_has_valid_sgml.assert_called_once_with(mock_filing)
    mock_company.assert_not_called()
    assert result is record


@edgar_available
def test_redownload_sec_filing_processor_check_sgml_invalid_redownloads(
    record: SecFilingRecord,
) -> None:
    mock_filing = MagicMock()
    mock_new_filing = MagicMock()
    mock_company = _mock_company_with_match(mock_new_filing)

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=mock_filing),
        patch(f"{MODULE}.has_valid_sgml", return_value=False),
        patch(f"{MODULE}.Company", return_value=mock_company) as mock_company_cls,
    ):
        result = RedownloadSecFilingProcessor(check_sgml=True).process(record)

    mock_company_cls.assert_called_once_with(320193)
    mock_new_filing.save.assert_called_once_with("tmp/test.pkl")
    assert result is record


@edgar_available
def test_redownload_sec_filing_processor_check_sgml_invalid_no_match_does_not_save(
    record: SecFilingRecord,
) -> None:
    mock_filing = MagicMock()
    mock_filings = MagicMock()
    mock_filings.__len__.return_value = 0
    mock_company = MagicMock()
    mock_company.get_filings.return_value.filter.return_value = mock_filings

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=mock_filing),
        patch(f"{MODULE}.has_valid_sgml", return_value=False),
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        result = RedownloadSecFilingProcessor(check_sgml=True).process(record)

    assert result is record


# --- process: check_sgml=True, filing not loadable ---


@edgar_available
def test_redownload_sec_filing_processor_check_sgml_not_loadable_skips_sgml_check(
    record: SecFilingRecord,
) -> None:
    mock_new_filing = MagicMock()
    mock_company = _mock_company_with_match(mock_new_filing)

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.has_valid_sgml") as mock_has_valid_sgml,
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        RedownloadSecFilingProcessor(check_sgml=True).process(record)

    mock_has_valid_sgml.assert_not_called()


@edgar_available
def test_redownload_sec_filing_processor_check_sgml_not_loadable_still_redownloads(
    record: SecFilingRecord,
) -> None:
    mock_new_filing = MagicMock()
    mock_company = _mock_company_with_match(mock_new_filing)

    with (
        patch.object(SecFilingRecord, "load_filing", return_value=None),
        patch(f"{MODULE}.has_valid_sgml", return_value=True),
        patch(f"{MODULE}.Company", return_value=mock_company),
    ):
        RedownloadSecFilingProcessor(check_sgml=True).process(record)

    mock_new_filing.save.assert_called_once_with("tmp/test.pkl")
