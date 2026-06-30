"""Unit tests for load_or_fetch_filings, load_or_fetch_company_filings,
and Config."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from glyphik.data.sec import SecFilingRecord
from glyphik.data.sp1500 import (
    Sp1500Company,
    load_or_fetch_company_filings,
    load_or_fetch_filings,
)
from glyphik.data.sp1500.filing import Config
from glyphik.testing.fixtures import edgar_available

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "glyphik.data.sp1500.filing"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=("10-K", "10-Q"),
    )


@pytest.fixture
def company() -> Sp1500Company:
    return Sp1500Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
        index="S&P 500",
    )


@pytest.fixture
def company_without_cik() -> Sp1500Company:
    return Sp1500Company(
        ticker="XYZ",
        cik=None,
        security="Example Corp",
        gics_sector="Industrials",
        gics_sub_industry="Industrial Machinery",
        index="S&P MidCap 400",
    )


###################################
#     Tests for Config           #
###################################


def test_config_cache_key_returns_str(config: Config) -> None:
    assert isinstance(config.cache_key(), str)


def test_config_cache_key_is_stable(config: Config) -> None:
    assert config.cache_key() == config.cache_key()


def test_config_cache_key_same_config_same_key() -> None:
    config_a = Config(start_date=date(2025, 1, 1), end_date=date(2026, 6, 1), forms=("10-K",))
    config_b = Config(start_date=date(2025, 1, 1), end_date=date(2026, 6, 1), forms=("10-K",))
    assert config_a.cache_key() == config_b.cache_key()


def test_config_cache_key_different_dates_different_key() -> None:
    config_a = Config(start_date=date(2025, 1, 1), end_date=date(2026, 6, 1), forms=("10-K",))
    config_b = Config(start_date=date(2024, 1, 1), end_date=date(2026, 6, 1), forms=("10-K",))
    assert config_a.cache_key() != config_b.cache_key()


def test_config_cache_key_different_forms_different_key() -> None:
    config_a = Config(start_date=date(2025, 1, 1), end_date=date(2026, 6, 1), forms=("10-K",))
    config_b = Config(start_date=date(2025, 1, 1), end_date=date(2026, 6, 1), forms=("10-Q",))
    assert config_a.cache_key() != config_b.cache_key()


###################################################
#     Tests for load_or_fetch_company_filings     #
###################################################


# --- no CIK ---


@edgar_available
def test_load_or_fetch_company_filings_no_cik_returns_empty(
    company_without_cik: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    result = load_or_fetch_company_filings(company_without_cik, config, tmp_path)
    assert result == []


@edgar_available
def test_load_or_fetch_company_filings_no_cik_does_not_call_fetch_filings(
    company_without_cik: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    with patch(f"{MODULE}.fetch_filings") as mock_fetch:
        load_or_fetch_company_filings(company_without_cik, config, tmp_path)
    mock_fetch.assert_not_called()


# --- cache hit ---


@edgar_available
def test_load_or_fetch_company_filings_cache_hit_loads_from_disk(
    company: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    cached_records = [SecFilingRecord(id="abc", metadata={"ticker": "AAPL"})]
    with patch(f"{MODULE}.format_cik", return_value="0000320193"):
        cache_path = tmp_path / "0000320193" / ".cache" / f"{config.cache_key()}.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.touch()
        with (
            patch(f"{MODULE}.load_dataclasses", return_value=cached_records) as mock_load,
            patch(f"{MODULE}.fetch_filings") as mock_fetch,
        ):
            result = load_or_fetch_company_filings(company, config, tmp_path)
    assert result == cached_records
    mock_load.assert_called_once_with(cache_path, SecFilingRecord)
    mock_fetch.assert_not_called()


# --- cache miss: success ---


@edgar_available
def test_load_or_fetch_company_filings_cache_miss_calls_fetch_filings(
    company: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    with (
        patch(f"{MODULE}.format_cik", return_value="0000320193"),
        patch(f"{MODULE}.fetch_filings", return_value=[]) as mock_fetch,
        patch(f"{MODULE}.save_dataclasses"),
    ):
        load_or_fetch_company_filings(company, config, tmp_path)
    mock_fetch.assert_called_once_with(
        cik_or_ticker=320193,
        start_date=config.start_date,
        end_date=config.end_date,
        output_dir=tmp_path,
        forms=list(config.forms),
    )


@edgar_available
def test_load_or_fetch_company_filings_cache_miss_returns_fetched_filings(
    company: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    filings = [SecFilingRecord(id="abc", metadata={"ticker": "AAPL"})]
    with (
        patch(f"{MODULE}.format_cik", return_value="0000320193"),
        patch(f"{MODULE}.fetch_filings", return_value=filings),
        patch(f"{MODULE}.save_dataclasses"),
    ):
        result = load_or_fetch_company_filings(company, config, tmp_path)
    assert result == filings


@edgar_available
def test_load_or_fetch_company_filings_cache_miss_saves_to_cache(
    company: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    filings = [SecFilingRecord(id="abc", metadata={"ticker": "AAPL"})]
    with (
        patch(f"{MODULE}.format_cik", return_value="0000320193"),
        patch(f"{MODULE}.fetch_filings", return_value=filings),
        patch(f"{MODULE}.save_dataclasses") as mock_save,
    ):
        load_or_fetch_company_filings(company, config, tmp_path)
    expected_path = tmp_path / "0000320193" / ".cache" / f"{config.cache_key()}.json"
    mock_save.assert_called_once_with(filings, expected_path)


# --- cache miss: failure ---


@edgar_available
def test_load_or_fetch_company_filings_fetch_failure_returns_empty(
    company: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    with (
        patch(f"{MODULE}.format_cik", return_value="0000320193"),
        patch(f"{MODULE}.fetch_filings", side_effect=ConnectionError("network down")),
    ):
        result = load_or_fetch_company_filings(company, config, tmp_path)
    assert result == []


@edgar_available
def test_load_or_fetch_company_filings_fetch_failure_does_not_save(
    company: Sp1500Company, config: Config, tmp_path: Path
) -> None:
    with (
        patch(f"{MODULE}.format_cik", return_value="0000320193"),
        patch(f"{MODULE}.fetch_filings", side_effect=ConnectionError("network down")),
        patch(f"{MODULE}.save_dataclasses") as mock_save,
    ):
        load_or_fetch_company_filings(company, config, tmp_path)
    mock_save.assert_not_called()


########################################
#     Tests for load_or_fetch_filings   #
########################################


@edgar_available
def test_load_or_fetch_filings_returns_list(company: Sp1500Company, tmp_path: Path) -> None:
    with patch(f"{MODULE}.load_or_fetch_company_filings", return_value=[]):
        result = load_or_fetch_filings(
            companies=[company],
            output_dir=tmp_path,
            start_date=date(2025, 1, 1),
            end_date=date(2026, 6, 1),
            forms=["10-K", "10-Q"],
        )
    assert isinstance(result, list)


@edgar_available
def test_load_or_fetch_filings_combines_all_company_results(
    company: Sp1500Company, company_without_cik: Sp1500Company, tmp_path: Path
) -> None:
    filings_a = [SecFilingRecord(id="a", metadata={})]
    filings_b = [SecFilingRecord(id="b", metadata={})]
    with patch(f"{MODULE}.load_or_fetch_company_filings", side_effect=[filings_a, filings_b]):
        result = load_or_fetch_filings(
            companies=[company, company_without_cik],
            output_dir=tmp_path,
            start_date=date(2025, 1, 1),
            end_date=date(2026, 6, 1),
            forms=["10-K"],
        )
    assert result == filings_a + filings_b


@edgar_available
def test_load_or_fetch_filings_calls_company_filings_once_per_company(
    company: Sp1500Company, tmp_path: Path
) -> None:
    with patch(f"{MODULE}.load_or_fetch_company_filings", return_value=[]) as mock_load:
        load_or_fetch_filings(
            companies=[company, company, company],
            output_dir=tmp_path,
            start_date=date(2025, 1, 1),
            end_date=date(2026, 6, 1),
            forms=["10-K"],
        )
    assert mock_load.call_count == 3


@edgar_available
def test_load_or_fetch_filings_passes_correct_config(
    company: Sp1500Company, tmp_path: Path
) -> None:
    with patch(f"{MODULE}.load_or_fetch_company_filings", return_value=[]) as mock_load:
        load_or_fetch_filings(
            companies=[company],
            output_dir=tmp_path,
            start_date=date(2025, 1, 1),
            end_date=date(2026, 6, 1),
            forms=["10-K", "10-Q"],
        )
    _, kwargs = mock_load.call_args
    assert kwargs["config"] == Config(
        start_date=date(2025, 1, 1), end_date=date(2026, 6, 1), forms=("10-K", "10-Q")
    )


@edgar_available
def test_load_or_fetch_filings_empty_companies_returns_empty_list(tmp_path: Path) -> None:
    result = load_or_fetch_filings(
        companies=[],
        output_dir=tmp_path,
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=["10-K"],
    )
    assert result == []


@edgar_available
def test_load_or_fetch_filings_accepts_str_output_dir(
    company: Sp1500Company, tmp_path: Path
) -> None:
    with patch(f"{MODULE}.load_or_fetch_company_filings", return_value=[]) as mock_load:
        load_or_fetch_filings(
            companies=[company],
            output_dir=str(tmp_path),
            start_date=date(2025, 1, 1),
            end_date=date(2026, 6, 1),
            forms=["10-K"],
        )
    _, kwargs = mock_load.call_args
    assert kwargs["output_dir"] == tmp_path
