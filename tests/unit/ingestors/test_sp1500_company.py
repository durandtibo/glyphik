from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from glyphik.data.sp1500 import Company
from glyphik.ingestors import Sp1500CompanyIngestor

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.ingestors.sp1500_company"


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


######################################################
#     Tests for Sp1500CompanyIngestor                #
######################################################


# --- Constructor ---


def test_sp1500_company_ingestor_stores_path(tmp_path: Path) -> None:
    ingestor = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json")
    assert ingestor._path == tmp_path / "sp1500.json"


def test_sp1500_company_ingestor_stores_find_missing_ciks(tmp_path: Path) -> None:
    ingestor = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json", find_missing_ciks=False)
    assert ingestor._find_missing_ciks is False


def test_sp1500_company_ingestor_find_missing_ciks_default_is_true(tmp_path: Path) -> None:
    ingestor = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json")
    assert ingestor._find_missing_ciks is True


def test_sp1500_company_ingestor_accepts_str_path(tmp_path: Path) -> None:
    ingestor = Sp1500CompanyIngestor(path=str(tmp_path / "sp1500.json"))
    assert ingestor._path == tmp_path / "sp1500.json"


def test_sp1500_company_ingestor_repr_contains_class_name(tmp_path: Path) -> None:
    ingestor = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json")
    assert "Sp1500CompanyIngestor" in repr(ingestor)


def test_sp1500_company_ingestor_str_contains_class_name(tmp_path: Path) -> None:
    ingestor = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json")
    assert "Sp1500CompanyIngestor" in str(ingestor)


def test_sp1500_company_ingestor_repr_contains_path(tmp_path: Path) -> None:
    path = tmp_path / "sp1500.json"
    assert str(path) in repr(Sp1500CompanyIngestor(path=path))


# --- ingest ---


def test_sp1500_company_ingestor_ingest_returns_list(tmp_path: Path) -> None:
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]):
        result = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json").ingest()
    assert isinstance(result, list)


def test_sp1500_company_ingestor_ingest_returns_companies(
    tmp_path: Path, companies: list[Company]
) -> None:
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=companies):
        result = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json").ingest()
    assert result == companies


def test_sp1500_company_ingestor_ingest_calls_load_or_fetch(tmp_path: Path) -> None:
    path = tmp_path / "sp1500.json"
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]) as mock_load:
        Sp1500CompanyIngestor(path=path).ingest()
    mock_load.assert_called_once_with(path=path, find_missing_ciks=True)


def test_sp1500_company_ingestor_ingest_passes_find_missing_ciks_false(tmp_path: Path) -> None:
    path = tmp_path / "sp1500.json"
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=[]) as mock_load:
        Sp1500CompanyIngestor(path=path, find_missing_ciks=False).ingest()
    mock_load.assert_called_once_with(path=path, find_missing_ciks=False)


def test_sp1500_company_ingestor_ingest_can_be_called_multiple_times(
    tmp_path: Path, companies: list[Company]
) -> None:
    with patch(f"{MODULE}.load_or_fetch_sp1500_companies", return_value=companies):
        ingestor = Sp1500CompanyIngestor(path=tmp_path / "sp1500.json")
        assert ingestor.ingest() == ingestor.ingest()
