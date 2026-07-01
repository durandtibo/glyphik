"""Unit tests for fill_missing_ciks."""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from glyphik.data.sp1500 import Company, fill_missing_ciks

MODULE = "glyphik.data.sp1500.cik"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
            ticker="XYZ",
            cik=None,
            security="Example Mid Corp",
            gics_sector="",
            gics_sub_industry="",
            index="S&P MidCap 400",
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
def companies_all_missing() -> list[Company]:
    return [
        Company(
            ticker="XYZ",
            cik=None,
            security="XYZ Inc",
            gics_sector="",
            gics_sub_industry="",
            index="S&P MidCap 400",
        ),
        Company(
            ticker="ABC",
            cik=None,
            security="ABC Inc",
            gics_sector="",
            gics_sub_industry="",
            index="S&P SmallCap 600",
        ),
    ]


@pytest.fixture
def companies_none_missing() -> list[Company]:
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


#######################################
#     Tests for fill_missing_ciks     #
#######################################


# --- return type and length ---


def test_fill_missing_ciks_returns_list(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        result = fill_missing_ciks(companies)
    assert isinstance(result, list)


def test_fill_missing_ciks_returns_same_length(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        result = fill_missing_ciks(companies)
    assert len(result) == len(companies)


# --- companies with existing CIK ---


def test_fill_missing_ciks_preserves_existing_cik(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        result = fill_missing_ciks(companies)
    assert result[0].cik == 320193
    assert result[2].cik == 789019


def test_fill_missing_ciks_does_not_call_fetch_for_existing_cik(
    companies_none_missing: list[Company],
) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker") as mock_fetch:
        fill_missing_ciks(companies_none_missing)
    mock_fetch.assert_not_called()


def test_fill_missing_ciks_returns_same_instance_for_existing_cik(
    companies_none_missing: list[Company],
) -> None:
    result = fill_missing_ciks(companies_none_missing)
    assert result[0] is companies_none_missing[0]
    assert result[1] is companies_none_missing[1]


# --- companies with missing CIK ---


def test_fill_missing_ciks_fills_missing_cik(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        result = fill_missing_ciks(companies)
    assert result[1].cik == 111111


def test_fill_missing_ciks_calls_fetch_for_missing_cik(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111) as mock_fetch:
        fill_missing_ciks(companies)
    mock_fetch.assert_called_once_with("XYZ")


def test_fill_missing_ciks_preserves_other_fields_when_filling(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        result = fill_missing_ciks(companies)
    assert result[1].ticker == "XYZ"
    assert result[1].security == "Example Mid Corp"
    assert result[1].index == "S&P MidCap 400"


def test_fill_missing_ciks_returns_new_instance_for_missing_cik(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        result = fill_missing_ciks(companies)
    assert result[1] is not companies[1]


def test_fill_missing_ciks_lookup_returns_none_sets_none(companies: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=None):
        result = fill_missing_ciks(companies)
    assert result[1].cik is None


# --- all missing / none missing ---


def test_fill_missing_ciks_all_missing(companies_all_missing: list[Company]) -> None:
    with patch(f"{MODULE}.fetch_cik_from_ticker", side_effect=[111111, 222222]):
        result = fill_missing_ciks(companies_all_missing)
    assert result[0].cik == 111111
    assert result[1].cik == 222222


def test_fill_missing_ciks_none_missing(companies_none_missing: list[Company]) -> None:
    result = fill_missing_ciks(companies_none_missing)
    assert result[0].cik == 320193
    assert result[1].cik == 789019


# --- edge cases ---


def test_fill_missing_ciks_empty_list() -> None:
    assert fill_missing_ciks([]) == []


def test_fill_missing_ciks_does_not_mutate_input(companies: list[Company]) -> None:
    original = [dataclasses.replace(c) for c in companies]
    with patch(f"{MODULE}.fetch_cik_from_ticker", return_value=111111):
        fill_missing_ciks(companies)
    assert companies == original
