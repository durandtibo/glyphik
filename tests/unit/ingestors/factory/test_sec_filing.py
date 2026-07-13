from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from coola.equality import objects_are_equal
from zenpyre.ingestors.factory import BaseIngestorFactory

from glyphik.ingestors.factory import SecFilingIngestorFactory

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.ingestors.factory.sec_filing"


def _make_factory(tmp_path: Path, **overrides: Any) -> SecFilingIngestorFactory:
    """Return a SecFilingIngestorFactory instance for testing."""
    kwargs = {
        "companies": ["AAPL", "MSFT"],
        "base_dir": tmp_path,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "forms": ["10-K"],
    }
    kwargs.update(overrides)
    return SecFilingIngestorFactory(**kwargs)


##################################################
#     Tests for SecFilingIngestorFactory         #
##################################################


# --- Inheritance ---


def test_sec_filing_ingestor_factory_is_base_ingestor_factory(tmp_path: Path) -> None:
    assert isinstance(_make_factory(tmp_path), BaseIngestorFactory)


# --- __init__ argument normalization ---


def test_sec_filing_ingestor_factory_stores_companies_as_list(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, companies=("AAPL", "MSFT"))
    assert factory._companies == ["AAPL", "MSFT"]


def test_sec_filing_ingestor_factory_sanitizes_base_dir_from_str(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, base_dir=str(tmp_path))
    assert factory._base_dir == tmp_path


def test_sec_filing_ingestor_factory_parses_start_date_from_str(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, start_date="2023-01-01")
    assert factory._start_date == date(2023, 1, 1)


def test_sec_filing_ingestor_factory_accepts_start_date_as_date(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, start_date=date(2023, 1, 1))
    assert factory._start_date == date(2023, 1, 1)


def test_sec_filing_ingestor_factory_parses_end_date_from_str(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, end_date="2023-12-31")
    assert factory._end_date == date(2023, 12, 31)


def test_sec_filing_ingestor_factory_accepts_end_date_as_date(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, end_date=date(2023, 12, 31))
    assert factory._end_date == date(2023, 12, 31)


def test_sec_filing_ingestor_factory_stores_forms(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path, forms=["10-K", "10-Q"])
    assert factory._forms == ["10-K", "10-Q"]


# --- make_ingestor wiring ---


def test_sec_filing_ingestor_factory_make_ingestor_builds_document_store_from_base_dir(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory") as mock_store_factory_cls,
        patch(f"{MODULE}.SecFilingIngestor"),
        patch(f"{MODULE}.SecFilingDocumentStoreIngestor"),
    ):
        factory.make_ingestor()
        mock_store_factory_cls.assert_called_once_with(base_dir=tmp_path)
        mock_store_factory_cls.return_value.make_document_store.assert_called_once_with()


def test_sec_filing_ingestor_factory_make_ingestor_builds_filing_ingestor_with_expected_dates(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path, start_date="2023-01-01", end_date="2023-12-31")
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory"),
        patch(f"{MODULE}.SecFilingIngestor") as mock_filing_ingestor_cls,
        patch(f"{MODULE}.SecFilingDocumentStoreIngestor"),
    ):
        factory.make_ingestor()
        _, call_kwargs = mock_filing_ingestor_cls.call_args
        assert call_kwargs["start_date"] == date(2023, 1, 1)
        assert call_kwargs["end_date"] == date(2023, 12, 31)
        assert call_kwargs["forms"] == ["10-K"]
        assert call_kwargs["output_dir"] == tmp_path / "sec" / "filing"


def test_sec_filing_ingestor_factory_make_ingestor_returns_store_ingestor(
    tmp_path: Path,
) -> None:
    factory = _make_factory(tmp_path)
    with (
        patch(f"{MODULE}.SecFilingDocumentStoreFactory"),
        patch(f"{MODULE}.SecFilingIngestor"),
        patch(f"{MODULE}.SecFilingDocumentStoreIngestor") as mock_store_ingestor_cls,
    ):
        result = factory.make_ingestor()
        assert result is mock_store_ingestor_cls.return_value


# --- _get_repr_kwargs ---


def test_sec_filing_ingestor_factory_get_repr_kwargs(tmp_path: Path) -> None:
    factory = _make_factory(
        tmp_path,
        companies=["AAPL", "MSFT", "GOOG"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        forms=["10-K"],
    )
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "companies": 3,
            "base_dir": tmp_path,
            "start_date": date(2023, 1, 1),
            "end_date": date(2023, 12, 31),
            "forms": ["10-K"],
        },
    )


# --- __repr__ and __str__ ---


def test_sec_filing_ingestor_factory_repr_starts_with_class_name(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path)
    assert repr(factory).startswith("SecFilingIngestorFactory(")


def test_sec_filing_ingestor_factory_str_starts_with_class_name(tmp_path: Path) -> None:
    factory = _make_factory(tmp_path)
    assert str(factory).startswith("SecFilingIngestorFactory(")
