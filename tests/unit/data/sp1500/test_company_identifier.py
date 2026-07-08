from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from glyphik.data.sp1500 import Company, get_sp1500_company_identifiers

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.data.sp1500.company_identifier"


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


####################################################
#     Tests for get_sp1500_company_identifiers     #
####################################################


def test_get_sp1500_company_identifiers_default_path_is_none() -> None:
    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = []
        get_sp1500_company_identifiers()

    mock_ingestor_cls.assert_called_once_with(path=None)


def test_get_sp1500_company_identifiers_passes_custom_path(tmp_path: Path) -> None:
    path = tmp_path / "sp1500.json"

    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = []
        get_sp1500_company_identifiers(path=path)

    mock_ingestor_cls.assert_called_once_with(path=path)


def test_get_sp1500_company_identifiers_passes_str_path(tmp_path: Path) -> None:
    path_str = str(tmp_path / "sp1500.json")

    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = []
        get_sp1500_company_identifiers(path=path_str)

    mock_ingestor_cls.assert_called_once_with(path=path_str)


def test_get_sp1500_company_identifiers_returns_list() -> None:
    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = []
        result = get_sp1500_company_identifiers()

    assert isinstance(result, list)


def test_get_sp1500_company_identifiers_empty_result() -> None:
    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = []
        result = get_sp1500_company_identifiers()

    assert result == []


def test_get_sp1500_company_identifiers_converts_companies_to_identifiers(
    companies: list[Company],
) -> None:
    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = companies
        result = get_sp1500_company_identifiers()

    assert len(result) == len(companies)
    assert [r.ticker for r in result] == [c.ticker for c in companies]
    assert [r.cik for r in result] == [c.cik for c in companies]


def test_get_sp1500_company_identifiers_preserves_order(companies: list[Company]) -> None:
    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.return_value = list(reversed(companies))
        result = get_sp1500_company_identifiers()

    assert [r.ticker for r in result] == [c.ticker for c in reversed(companies)]


def test_get_sp1500_company_identifiers_propagates_ingest_exception() -> None:
    with patch(f"{MODULE}.Sp1500CompanyIngestor") as mock_ingestor_cls:
        mock_ingestor_cls.return_value.ingest.side_effect = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            get_sp1500_company_identifiers()
