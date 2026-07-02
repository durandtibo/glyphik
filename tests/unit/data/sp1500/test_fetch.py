"""Unit tests for Sp1500Company and the S&P 1500 fetching utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from coola.testing.fixtures import pandas_available
from coola.utils.imports import is_pandas_available
from zenpyre.utils.dataclass_io import save_dataclasses

from glyphik.data.sp1500 import (
    Company,
    fetch_sp1500_companies,
    load_or_fetch_sp1500_companies,
)
from glyphik.data.sp1500.fetch import (
    _find_column,
    _find_constituent_table,
    _find_optional_column,
    _parse_table,
)

if TYPE_CHECKING:
    from pathlib import Path

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    from coola.utils.fallback.pandas import pandas as pd

MODULE = "glyphik.data.sp1500.fetch"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sp500_table() -> pd.DataFrame:
    """A minimal table mimicking the S&P 500 Wikipedia constituent
    table."""
    return pd.DataFrame(
        {
            "Symbol": ["AAPL", "MSFT"],
            "Security": ["Apple Inc.", "Microsoft Corporation"],
            "GICS Sector": ["Information Technology", "Information Technology"],
            "GICS Sub-Industry": ["Technology Hardware", "Systems Software"],
            "CIK": [320193, 789019],
        }
    )


@pytest.fixture
def midcap_table_without_cik() -> pd.DataFrame:
    """A minimal table mimicking the S&P MidCap 400 table (no CIK
    column)."""
    return pd.DataFrame(
        {
            "Ticker symbol": ["XYZ"],
            "Company": ["Example Mid Corp"],
            "GICS Sector": ["Industrials"],
            "GICS Sub Industry": ["Industrial Machinery"],
        }
    )


@pytest.fixture
def irrelevant_table() -> pd.DataFrame:
    """A table that does not contain a ticker column."""
    return pd.DataFrame({"Foo": [1, 2], "Bar": [3, 4]})


@pytest.fixture
def companies() -> list[Company]:
    return [
        Company(
            ticker="AAPL",
            cik=320193,
            security="Apple Inc.",
            gics_sector="Information Technology",
            gics_sub_industry="Technology Hardware",
            index="S&P 500",
        ),
        Company(
            ticker="XYZ",
            cik=None,
            security="Example Mid Corp",
            gics_sector="Industrials",
            gics_sub_industry="Industrial Machinery",
            index="S&P MidCap 400",
        ),
    ]


###############################################
#     Tests for _find_constituent_table       #
###############################################


@pandas_available
def test_find_constituent_table_returns_matching_table(
    sp500_table: pd.DataFrame, irrelevant_table: pd.DataFrame
) -> None:
    result = _find_constituent_table([irrelevant_table, sp500_table], "S&P 500")
    assert result is sp500_table


@pandas_available
def test_find_constituent_table_returns_first_match(sp500_table: pd.DataFrame) -> None:
    other_match = sp500_table.copy()
    result = _find_constituent_table([sp500_table, other_match], "S&P 500")
    assert result is sp500_table


@pandas_available
def test_find_constituent_table_no_match_raises(irrelevant_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"Could not find a constituent table"):
        _find_constituent_table([irrelevant_table], "S&P 500")


@pandas_available
def test_find_constituent_table_empty_list_raises() -> None:
    with pytest.raises(ValueError, match=r"Could not find a constituent table"):
        _find_constituent_table([], "S&P 500")


@pandas_available
def test_find_constituent_table_recognises_alternate_ticker_column(
    midcap_table_without_cik: pd.DataFrame,
) -> None:
    result = _find_constituent_table([midcap_table_without_cik], "S&P MidCap 400")
    assert result is midcap_table_without_cik


###########################################
#     Tests for _find_optional_column     #
###########################################


@pandas_available
def test_find_optional_column_returns_match(sp500_table: pd.DataFrame) -> None:
    assert _find_optional_column(sp500_table, ("Symbol", "Ticker symbol")) == "Symbol"


@pandas_available
def test_find_optional_column_returns_first_match_in_priority_order() -> None:
    table = pd.DataFrame({"Ticker": ["AAPL"], "Symbol": ["AAPL"]})
    assert _find_optional_column(table, ("Symbol", "Ticker")) == "Symbol"


@pandas_available
def test_find_optional_column_no_match_returns_none(irrelevant_table: pd.DataFrame) -> None:
    assert _find_optional_column(irrelevant_table, ("Symbol", "Ticker symbol")) is None


@pandas_available
def test_find_optional_column_empty_candidates_returns_none(sp500_table: pd.DataFrame) -> None:
    assert _find_optional_column(sp500_table, ()) is None


##################################
#     Tests for _find_column    #
##################################


@pandas_available
def test_find_column_returns_match(sp500_table: pd.DataFrame) -> None:
    assert _find_column(sp500_table, ("Symbol", "Ticker"), "ticker", "S&P 500") == "Symbol"


@pandas_available
def test_find_column_no_match_raises(irrelevant_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"Could not find a 'ticker' column for S&P 500"):
        _find_column(irrelevant_table, ("Symbol", "Ticker"), "ticker", "S&P 500")


@pandas_available
def test_find_column_error_message_includes_field_and_index(irrelevant_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"GICS Sector"):
        _find_column(irrelevant_table, ("GICS Sector",), "GICS Sector", "S&P MidCap 400")


##################################
#     Tests for _parse_table    #
##################################


@pandas_available
def test_parse_table_returns_list_of_company(sp500_table: pd.DataFrame) -> None:
    result = _parse_table(sp500_table, "S&P 500")
    assert all(isinstance(c, Company) for c in result)


@pandas_available
def test_parse_table_one_company_per_row(sp500_table: pd.DataFrame) -> None:
    result = _parse_table(sp500_table, "S&P 500")
    assert len(result) == 2


@pandas_available
def test_parse_table_field_values(sp500_table: pd.DataFrame) -> None:
    result = _parse_table(sp500_table, "S&P 500")
    assert result[0] == Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
        index="S&P 500",
    )


@pandas_available
def test_parse_table_sets_index_name(sp500_table: pd.DataFrame) -> None:
    result = _parse_table(sp500_table, "S&P 500")
    assert all(c.index == "S&P 500" for c in result)


@pandas_available
def test_parse_table_missing_cik_column_sets_none(midcap_table_without_cik: pd.DataFrame) -> None:
    result = _parse_table(midcap_table_without_cik, "S&P MidCap 400")
    assert result[0].cik is None


@pandas_available
def test_parse_table_missing_cik_other_fields_still_set(
    midcap_table_without_cik: pd.DataFrame,
) -> None:
    result = _parse_table(midcap_table_without_cik, "S&P MidCap 400")
    assert result[0].ticker == "XYZ"
    assert result[0].security == "Example Mid Corp"
    assert result[0].gics_sector == "Industrials"
    assert result[0].gics_sub_industry == "Industrial Machinery"


@pandas_available
def test_parse_table_nan_cik_value_sets_none() -> None:
    table = pd.DataFrame(
        {
            "Symbol": ["AAPL"],
            "Security": ["Apple Inc."],
            "GICS Sector": ["Information Technology"],
            "GICS Sub-Industry": ["Technology Hardware"],
            "CIK": [float("nan")],
        }
    )
    result = _parse_table(table, "S&P 500")
    assert result[0].cik is None


@pandas_available
def test_parse_table_strips_whitespace() -> None:
    table = pd.DataFrame(
        {
            "Symbol": ["  AAPL  "],
            "Security": [" Apple Inc. "],
            "GICS Sector": [" Information Technology "],
            "GICS Sub-Industry": [" Technology Hardware "],
            "CIK": [320193],
        }
    )
    result = _parse_table(table, "S&P 500")
    assert result[0].ticker == "AAPL"
    assert result[0].security == "Apple Inc."


@pandas_available
def test_parse_table_missing_required_column_raises(irrelevant_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"Could not find a 'ticker' column for S&P 500"):
        _parse_table(irrelevant_table, "S&P 500")


@pandas_available
def test_parse_table_empty_table_returns_empty_list() -> None:
    table = pd.DataFrame(
        {"Symbol": [], "Security": [], "GICS Sector": [], "GICS Sub-Industry": [], "CIK": []}
    )
    assert _parse_table(table, "S&P 500") == []


##########################################
#     Tests for fetch_sp1500_companies   #
##########################################


@pandas_available
def test_fetch_sp1500_companies_returns_list(sp500_table: pd.DataFrame) -> None:
    with patch(f"{MODULE}.pd.read_html", return_value=[sp500_table]):
        result = fetch_sp1500_companies()
    assert isinstance(result, list)


@pandas_available
def test_fetch_sp1500_companies_combines_all_three_indices(
    sp500_table: pd.DataFrame, midcap_table_without_cik: pd.DataFrame
) -> None:
    with patch(
        f"{MODULE}.pd.read_html",
        side_effect=[[sp500_table], [midcap_table_without_cik], [sp500_table]],
    ):
        result = fetch_sp1500_companies()
    assert len(result) == 2 + 1 + 2


@pandas_available
def test_fetch_sp1500_companies_sets_correct_index_per_call(
    sp500_table: pd.DataFrame, midcap_table_without_cik: pd.DataFrame
) -> None:
    with patch(
        f"{MODULE}.pd.read_html",
        side_effect=[[sp500_table], [midcap_table_without_cik], [sp500_table]],
    ):
        result = fetch_sp1500_companies()
    indices = {c.index for c in result}
    assert indices == {"S&P 500", "S&P MidCap 400", "S&P SmallCap 600"}


@pandas_available
def test_fetch_sp1500_companies_calls_read_html_three_times(sp500_table: pd.DataFrame) -> None:
    with patch(f"{MODULE}.pd.read_html", return_value=[sp500_table]) as mock_read_html:
        fetch_sp1500_companies()
    assert mock_read_html.call_count == 3


@pandas_available
def test_fetch_sp1500_companies_propagates_parse_error(irrelevant_table: pd.DataFrame) -> None:
    with (
        patch(f"{MODULE}.pd.read_html", return_value=[irrelevant_table]),
        pytest.raises(
            ValueError, match=r"Could not find a constituent table with a ticker column for S&P 500"
        ),
    ):
        fetch_sp1500_companies()


#################################################
#     Tests for load_or_fetch_sp1500_companies  #
#################################################


#################################################
#     Tests for load_or_fetch_sp1500_companies  #
#################################################


# --- Cache exists ---


def test_load_or_fetch_sp1500_companies_loads_existing_cache(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"
    save_dataclasses(companies, path)

    with patch(f"{MODULE}.fetch_sp1500_companies") as mock_fetch:
        result = load_or_fetch_sp1500_companies(path)

    assert result == companies
    mock_fetch.assert_not_called()


def test_load_or_fetch_sp1500_companies_does_not_overwrite_existing_cache(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"
    save_dataclasses(companies, path)
    original_mtime = path.stat().st_mtime_ns

    with patch(f"{MODULE}.fetch_sp1500_companies"):
        load_or_fetch_sp1500_companies(path)

    assert path.stat().st_mtime_ns == original_mtime


def test_load_or_fetch_sp1500_companies_accepts_str_path(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"
    save_dataclasses(companies, path)

    result = load_or_fetch_sp1500_companies(str(path))
    assert result == companies


def test_load_or_fetch_sp1500_companies_cache_hit_does_not_call_fill_missing_ciks(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"
    save_dataclasses(companies, path)

    with patch(f"{MODULE}.fill_missing_ciks") as mock_fill:
        load_or_fetch_sp1500_companies(path)

    mock_fill.assert_not_called()


# --- Cache does not exist ---


def test_load_or_fetch_sp1500_companies_fetches_when_no_cache(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies) as mock_fetch,
        patch(f"{MODULE}.fill_missing_ciks", return_value=companies),
    ):
        result = load_or_fetch_sp1500_companies(path)

    mock_fetch.assert_called_once()
    assert result == companies


def test_load_or_fetch_sp1500_companies_saves_after_fetching(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies),
        patch(f"{MODULE}.fill_missing_ciks", return_value=companies),
    ):
        load_or_fetch_sp1500_companies(path)

    assert path.exists()


def test_load_or_fetch_sp1500_companies_saved_file_is_loadable(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies),
        patch(f"{MODULE}.fill_missing_ciks", return_value=companies),
    ):
        load_or_fetch_sp1500_companies(path)

    assert load_or_fetch_sp1500_companies(path) == companies


def test_load_or_fetch_sp1500_companies_second_call_uses_cache(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies) as mock_fetch,
        patch(f"{MODULE}.fill_missing_ciks", return_value=companies),
    ):
        load_or_fetch_sp1500_companies(path)
        load_or_fetch_sp1500_companies(path)

    mock_fetch.assert_called_once()


def test_load_or_fetch_sp1500_companies_empty_fetch_result(tmp_path: Path) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=[]),
        patch(f"{MODULE}.fill_missing_ciks", return_value=[]),
    ):
        result = load_or_fetch_sp1500_companies(path)

    assert result == []
    assert path.exists()


# --- find_missing_ciks parameter ---


def test_load_or_fetch_sp1500_companies_calls_fill_missing_ciks_by_default(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies),
        patch(f"{MODULE}.fill_missing_ciks", return_value=companies) as mock_fill,
    ):
        load_or_fetch_sp1500_companies(path)

    mock_fill.assert_called_once_with(companies)


def test_load_or_fetch_sp1500_companies_skips_fill_when_find_missing_ciks_false(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies),
        patch(f"{MODULE}.fill_missing_ciks") as mock_fill,
    ):
        load_or_fetch_sp1500_companies(path, find_missing_ciks=False)

    mock_fill.assert_not_called()


def test_load_or_fetch_sp1500_companies_find_missing_ciks_true_result(
    tmp_path: Path, companies: list[Company]
) -> None:
    path = tmp_path / "sp1500.json"
    enriched = [
        Company(
            ticker="XYZ",
            cik=999999,
            security="Example Mid Corp",
            gics_sector="Industrials",
            gics_sub_industry="Industrial Machinery",
            index="S&P MidCap 400",
        )
    ]

    with (
        patch(f"{MODULE}.fetch_sp1500_companies", return_value=companies),
        patch(f"{MODULE}.fill_missing_ciks", return_value=enriched),
    ):
        result = load_or_fetch_sp1500_companies(path, find_missing_ciks=True)

    assert result == enriched
