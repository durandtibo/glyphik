"""Unit tests for SecFilingRecord."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from edgar import Company, Filing

from glyphik.data.sec import SecFilingRecord, fetch_filings, fetch_form_filings

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.data.sec.filing"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_filing(
    accession: str = "0000320193-24-000001",
    filing_date: date = date(2024, 1, 15),
) -> MagicMock:
    filing = MagicMock(spec=Filing)
    filing.accession_number = accession
    filing.filing_date = filing_date
    return filing


def _make_mock_company(
    cik: int = 320193,
    name: str = "Test Corp",
    filings: list | None = None,
) -> MagicMock:
    company = MagicMock(spec=Company)
    company.cik = cik
    company.name = name
    collection = MagicMock()
    collection.filter.return_value = filings or []
    company.get_filings.return_value = collection
    return company


#######################################
#     Tests for SecFilingRecord       #
#######################################


# --- Construction ---


def test_sec_filing_record_is_frozen() -> None:
    record = SecFilingRecord(id="abc", metadata={"key": "value"})
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'id'"):
        record.id = "other"


def test_sec_filing_record_default_metadata() -> None:
    record = SecFilingRecord(id="abc")
    assert record.metadata == {}


def test_sec_filing_record_stores_id() -> None:
    record = SecFilingRecord(id="abc", metadata={"key": "value"})
    assert record.id == "abc"


def test_sec_filing_record_stores_metadata() -> None:
    metadata = {"filepath": "tmp/test.pkl", "cik": 123}
    record = SecFilingRecord(id="abc", metadata=metadata)
    assert record.metadata == metadata


# --- from_metadata ---


def test_sec_filing_record_from_metadata_returns_sec_filing_record() -> None:
    assert isinstance(SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"}), SecFilingRecord)


def test_sec_filing_record_from_metadata_sets_metadata() -> None:
    metadata = {"filepath": "tmp/test.pkl", "cik": 123}
    record = SecFilingRecord.from_metadata(metadata)
    assert record.metadata == metadata


def test_sec_filing_record_from_metadata_sets_id() -> None:
    record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    assert record.id is not None


def test_sec_filing_record_from_metadata_id_is_str() -> None:
    record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    assert isinstance(record.id, str)


def test_sec_filing_record_from_metadata_id_is_stable() -> None:
    metadata = {"filepath": "tmp/test.pkl", "cik": 123}
    assert SecFilingRecord.from_metadata(metadata).id == SecFilingRecord.from_metadata(metadata).id


def test_sec_filing_record_from_metadata_different_metadata_different_id() -> None:
    record_a = SecFilingRecord.from_metadata({"filepath": "tmp/a.pkl"})
    record_b = SecFilingRecord.from_metadata({"filepath": "tmp/b.pkl"})
    assert record_a.id != record_b.id


def test_sec_filing_record_from_metadata_key_order_independent() -> None:
    record_a = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl", "cik": 123})
    record_b = SecFilingRecord.from_metadata({"cik": 123, "filepath": "tmp/test.pkl"})
    assert record_a.id == record_b.id


# --- load_filing ---


def test_sec_filing_record_load_filing_missing_filepath_raises() -> None:
    record = SecFilingRecord(id="abc", metadata={})
    with pytest.raises(ValueError, match="filepath"):
        record.load_filing()


def test_sec_filing_record_load_filing_calls_filing_load() -> None:
    record = SecFilingRecord(id="abc", metadata={"filepath": "tmp/test.pkl"})
    with patch(f"{MODULE}.Filing.load") as mock_load:
        record.load_filing()
    mock_load.assert_called_once_with("tmp/test.pkl")


def test_sec_filing_record_load_filing_returns_filing() -> None:
    mock_filing = MagicMock()
    record = SecFilingRecord(id="abc", metadata={"filepath": "tmp/test.pkl"})
    with patch(f"{MODULE}.Filing.load", return_value=mock_filing):
        result = record.load_filing()
    assert result is mock_filing


##############################################
#     Tests for fetch_form_filings           #
##############################################


def test_fetch_form_filings_returns_list(tmp_path: Path) -> None:
    company = _make_mock_company()
    assert isinstance(
        fetch_form_filings(
            company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
        ),
        list,
    )


def test_fetch_form_filings_empty_filings_returns_empty(tmp_path: Path) -> None:
    company = _make_mock_company(filings=[])
    assert (
        fetch_form_filings(
            company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
        )
        == []
    )


def test_fetch_form_filings_returns_sec_filing_records(tmp_path: Path) -> None:
    company = _make_mock_company(filings=[_make_mock_filing()])
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert all(isinstance(r, SecFilingRecord) for r in result)


def test_fetch_form_filings_one_record_per_filing(tmp_path: Path) -> None:
    company = _make_mock_company(
        filings=[_make_mock_filing("acc-001"), _make_mock_filing("acc-002")]
    )
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert len(result) == 2


def test_fetch_form_filings_record_metadata_keys(tmp_path: Path) -> None:
    company = _make_mock_company(filings=[_make_mock_filing()])
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert set(result[0].metadata.keys()) == {
        "accession_no",
        "cik",
        "company_name",
        "filepath",
        "form",
        "source",
        "ticker",
    }


def test_fetch_form_filings_record_metadata_values(tmp_path: Path) -> None:
    company = _make_mock_company(
        cik=320193, name="Test Corp", filings=[_make_mock_filing("acc-001")]
    )
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert result[0].metadata["accession_no"] == "acc-001"
    assert result[0].metadata["cik"] == 320193
    assert result[0].metadata["company_name"] == "Test Corp"
    assert result[0].metadata["form"] == "10-K"
    assert result[0].metadata["source"] == "SEC EDGAR"


def test_fetch_form_filings_saves_file_when_not_exists(tmp_path: Path) -> None:
    filing = _make_mock_filing("acc-001")
    company = _make_mock_company(filings=[filing])
    fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    filing.save.assert_called_once()


def test_fetch_form_filings_skips_existing_file(tmp_path: Path) -> None:
    filing = _make_mock_filing("acc-001")
    (tmp_path / "acc-001.pkl").touch()
    company = _make_mock_company(filings=[filing])
    fetch_form_filings(
        company=company,
        output_dir=tmp_path,
        form="10-K",
        date_range="2024-01-01:2024-12-31",
        force_download=False,
    )
    filing.save.assert_not_called()


def test_fetch_form_filings_force_download_overwrites_existing(tmp_path: Path) -> None:
    filing = _make_mock_filing("acc-001")
    (tmp_path / "acc-001.pkl").touch()
    company = _make_mock_company(filings=[filing])
    fetch_form_filings(
        company=company,
        output_dir=tmp_path,
        form="10-K",
        date_range="2024-01-01:2024-12-31",
        force_download=True,
    )
    filing.save.assert_called_once()


##############################################
#     Tests for fetch_filings                #
##############################################


def test_fetch_filings_returns_list(tmp_path: Path) -> None:
    with patch(f"{MODULE}.Company", return_value=_make_mock_company()):
        result = fetch_filings(cik=320193, start_date=date(2024, 1, 1), output_dir=tmp_path)
    assert isinstance(result, list)


def test_fetch_filings_empty_returns_empty(tmp_path: Path) -> None:
    with patch(f"{MODULE}.Company", return_value=_make_mock_company(filings=[])):
        result = fetch_filings(cik=320193, start_date=date(2024, 1, 1), output_dir=tmp_path)
    assert result == []


def test_fetch_filings_default_forms_are_ten_k_and_ten_q(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.Company", return_value=_make_mock_company()),
        patch(f"{MODULE}.fetch_form_filings", return_value=[]) as mock_fetch,
    ):
        fetch_filings(cik=320193, start_date=date(2024, 1, 1), output_dir=tmp_path)
    forms = [c.kwargs["form"] for c in mock_fetch.call_args_list]
    assert set(forms) == {"10-K", "10-Q"}


def test_fetch_filings_custom_forms(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.Company", return_value=_make_mock_company()),
        patch(f"{MODULE}.fetch_form_filings", return_value=[]) as mock_fetch,
    ):
        fetch_filings(cik=320193, start_date=date(2024, 1, 1), output_dir=tmp_path, forms=["10-K"])
    forms = [c.kwargs["form"] for c in mock_fetch.call_args_list]
    assert forms == ["10-K"]


def test_fetch_filings_returns_sec_filing_records(tmp_path: Path) -> None:
    record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    with (
        patch(f"{MODULE}.Company", return_value=_make_mock_company()),
        patch(f"{MODULE}.fetch_form_filings", return_value=[record]),
    ):
        result = fetch_filings(cik=320193, start_date=date(2024, 1, 1), output_dir=tmp_path)
    assert all(isinstance(r, SecFilingRecord) for r in result)
