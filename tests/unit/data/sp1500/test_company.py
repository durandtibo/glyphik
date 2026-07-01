"""Unit tests for Sp1500Company and the S&P 1500 fetching utilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from glyphik.data.sp1500 import (
    Company,
)

MODULE = "glyphik.data.sp1500.company"


#############################
#     Tests for Company     #
#############################


def test_company_is_frozen() -> None:
    company = Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
        index="S&P 500",
    )
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'ticker'"):
        company.ticker = "MSFT"


def test_company_stores_fields() -> None:
    company = Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
        index="S&P 500",
    )
    assert company.ticker == "AAPL"
    assert company.cik == 320193
    assert company.security == "Apple Inc."
    assert company.gics_sector == "Information Technology"
    assert company.gics_sub_industry == "Technology Hardware"
    assert company.index == "S&P 500"


def test_company_cik_can_be_none() -> None:
    company = Company(
        ticker="XYZ",
        cik=None,
        security="Example Mid Corp",
        gics_sector="Industrials",
        gics_sub_industry="Industrial Machinery",
        index="S&P MidCap 400",
    )
    assert company.cik is None


def test_company_equality() -> None:
    company_a = Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
        index="S&P 500",
    )
    company_b = Company(
        ticker="AAPL",
        cik=320193,
        security="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
        index="S&P 500",
    )
    assert company_a == company_b
