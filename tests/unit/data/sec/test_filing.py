"""Unit tests for SecFilingRecord."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from glyphik.data.sec import SecFilingRecord, fetch_filings, fetch_form_filings
from glyphik.data.sec.filing import has_valid_sgml
from glyphik.testing.fixtures import edgar_available
from glyphik.utils.imports import is_edgar_available

if TYPE_CHECKING:
    from pathlib import Path

if is_edgar_available():
    from edgar import Company, Filing

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


##############################################
#     Tests for fetch_form_filings           #
##############################################


@edgar_available
def test_fetch_form_filings_returns_list(tmp_path: Path) -> None:
    company = _make_mock_company()
    assert isinstance(
        fetch_form_filings(
            company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
        ),
        list,
    )


@edgar_available
def test_fetch_form_filings_empty_filings_returns_empty(tmp_path: Path) -> None:
    company = _make_mock_company(filings=[])
    assert (
        fetch_form_filings(
            company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
        )
        == []
    )


@edgar_available
def test_fetch_form_filings_returns_sec_filing_records(tmp_path: Path) -> None:
    company = _make_mock_company(filings=[_make_mock_filing()])
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert all(isinstance(r, SecFilingRecord) for r in result)


@edgar_available
def test_fetch_form_filings_one_record_per_filing(tmp_path: Path) -> None:
    company = _make_mock_company(
        filings=[_make_mock_filing("acc-001"), _make_mock_filing("acc-002")]
    )
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert len(result) == 2


@edgar_available
def test_fetch_form_filings_record_metadata_keys(tmp_path: Path) -> None:
    company = _make_mock_company(filings=[_make_mock_filing()])
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert set(result[0].metadata.keys()) == {
        "accession_number",
        "cik",
        "company_name",
        "filepath",
        "filing_date",
        "form",
        "source",
        "ticker",
    }


@edgar_available
def test_fetch_form_filings_record_metadata_values(tmp_path: Path) -> None:
    company = _make_mock_company(
        cik=320193, name="Test Corp", filings=[_make_mock_filing("acc-001")]
    )
    result = fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    assert result[0].metadata["accession_number"] == "acc-001"
    assert result[0].metadata["cik"] == 320193
    assert result[0].metadata["company_name"] == "Test Corp"
    assert result[0].metadata["filepath"] == (tmp_path / "acc-001.pkl").as_posix()
    assert result[0].metadata["filing_date"] == "2024-01-15"
    assert result[0].metadata["form"] == "10-K"
    assert result[0].metadata["source"] == "SEC EDGAR"


@edgar_available
def test_fetch_form_filings_saves_file_when_not_exists(tmp_path: Path) -> None:
    filing = _make_mock_filing("acc-001")
    company = _make_mock_company(filings=[filing])
    fetch_form_filings(
        company=company, output_dir=tmp_path, form="10-K", date_range="2024-01-01:2024-12-31"
    )
    filing.save.assert_called_once()


@edgar_available
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


@edgar_available
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


@edgar_available
def test_fetch_filings_returns_list(tmp_path: Path) -> None:
    with patch(f"{MODULE}.Company", return_value=_make_mock_company()):
        result = fetch_filings(
            cik_or_ticker=320193, start_date=date(2024, 1, 1), output_dir=tmp_path
        )
    assert isinstance(result, list)


@edgar_available
def test_fetch_filings_empty_returns_empty(tmp_path: Path) -> None:
    with patch(f"{MODULE}.Company", return_value=_make_mock_company(filings=[])):
        result = fetch_filings(
            cik_or_ticker=320193, start_date=date(2024, 1, 1), output_dir=tmp_path
        )
    assert result == []


@edgar_available
def test_fetch_filings_default_forms_are_ten_k_and_ten_q(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.Company", return_value=_make_mock_company()),
        patch(f"{MODULE}.fetch_form_filings", return_value=[]) as mock_fetch,
    ):
        fetch_filings(cik_or_ticker=320193, start_date=date(2024, 1, 1), output_dir=tmp_path)
    forms = [c.kwargs["form"] for c in mock_fetch.call_args_list]
    assert set(forms) == {"10-K", "10-Q"}


@edgar_available
def test_fetch_filings_custom_forms(tmp_path: Path) -> None:
    with (
        patch(f"{MODULE}.Company", return_value=_make_mock_company()),
        patch(f"{MODULE}.fetch_form_filings", return_value=[]) as mock_fetch,
    ):
        fetch_filings(
            cik_or_ticker=320193, start_date=date(2024, 1, 1), output_dir=tmp_path, forms=["10-K"]
        )
    forms = [c.kwargs["form"] for c in mock_fetch.call_args_list]
    assert forms == ["10-K"]


@edgar_available
def test_fetch_filings_returns_sec_filing_records(tmp_path: Path) -> None:
    record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    with (
        patch(f"{MODULE}.Company", return_value=_make_mock_company()),
        patch(f"{MODULE}.fetch_form_filings", return_value=[record]),
    ):
        result = fetch_filings(
            cik_or_ticker=320193, start_date=date(2024, 1, 1), output_dir=tmp_path
        )
    assert all(isinstance(r, SecFilingRecord) for r in result)


@edgar_available
def test_fetch_filings_with_cik(tmp_path: Path) -> None:
    with patch(f"{MODULE}.Company", return_value=_make_mock_company()) as mock_company:
        result = fetch_filings(
            cik_or_ticker=320193, start_date=date(2024, 1, 1), output_dir=tmp_path
        )
    assert isinstance(result, list)
    mock_company.assert_called_once_with(320193)


@edgar_available
def test_fetch_filings_with_ticker(tmp_path: Path) -> None:
    with patch(f"{MODULE}.Company", return_value=_make_mock_company()) as mock_company:
        result = fetch_filings(
            cik_or_ticker="AAPL", start_date=date(2024, 1, 1), output_dir=tmp_path
        )
    assert isinstance(result, list)
    mock_company.assert_called_once_with("AAPL")


#########################################
#   Tests for has_valid_sgml            #
#########################################


def _make_sgml(n_attachments: int) -> MagicMock:
    sgml_data = MagicMock()
    sgml_data.attachments = [MagicMock() for _ in range(n_attachments)]
    return sgml_data


# --- valid cases ---


@edgar_available
def test_has_valid_sgml_returns_true_with_one_attachment() -> None:
    filing = _make_mock_filing()
    filing.sgml.return_value = _make_sgml(1)
    assert has_valid_sgml(filing) is True


@edgar_available
def test_has_valid_sgml_returns_true_with_multiple_attachments() -> None:
    filing = _make_mock_filing()
    filing.sgml.return_value = _make_sgml(5)
    assert has_valid_sgml(filing) is True


@edgar_available
def test_has_valid_sgml_calls_sgml_once() -> None:
    filing = _make_mock_filing()
    filing.sgml.return_value = _make_sgml(1)
    has_valid_sgml(filing)
    filing.sgml.assert_called_once()


# --- invalid cases ---


@edgar_available
def test_has_valid_sgml_returns_false_when_sgml_is_none() -> None:
    filing = _make_mock_filing()
    filing.sgml.return_value = None
    assert has_valid_sgml(filing) is False


@edgar_available
def test_has_valid_sgml_returns_false_with_zero_attachments() -> None:
    filing = _make_mock_filing()
    filing.sgml.return_value = _make_sgml(0)
    assert has_valid_sgml(filing) is False


@edgar_available
def test_has_valid_sgml_returns_false_when_sgml_raises() -> None:
    filing = _make_mock_filing()
    filing.sgml.side_effect = ValueError("malformed SGML")
    assert has_valid_sgml(filing) is False


@edgar_available
def test_has_valid_sgml_returns_false_on_network_error() -> None:
    filing = _make_mock_filing()
    filing.sgml.side_effect = ConnectionError("network unreachable")
    assert has_valid_sgml(filing) is False


@edgar_available
def test_has_valid_sgml_returns_false_on_generic_exception() -> None:
    filing = _make_mock_filing()
    filing.sgml.side_effect = Exception("unexpected failure")
    assert has_valid_sgml(filing) is False


# --- return type ---


@edgar_available
def test_has_valid_sgml_returns_bool_on_success() -> None:
    filing = _make_mock_filing()
    filing.sgml.return_value = _make_sgml(1)
    assert isinstance(has_valid_sgml(filing), bool)


@edgar_available
def test_has_valid_sgml_returns_bool_on_failure() -> None:
    filing = _make_mock_filing()
    filing.sgml.side_effect = ValueError("boom")
    assert isinstance(has_valid_sgml(filing), bool)
