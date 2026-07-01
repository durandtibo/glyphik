r"""Provide a utility to fetch the current S&P 1500 constituents from
Wikipedia."""

from __future__ import annotations

__all__ = ["Company"]

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    """A single S&P 1500 constituent company.

    Args:
        ticker: The stock ticker symbol (e.g. ``"AAPL"``).
        cik: The SEC Central Index Key (CIK), or ``None`` if not
            available on the source page (e.g. the S&P MidCap 400
            Wikipedia table does not include a CIK column).
        security: The official company/security name.
        gics_sector: The GICS sector classification.
        gics_sub_industry: The GICS sub-industry classification.
        index: The S&P sub-index the company belongs to — one of
            ``"S&P 500"``, ``"S&P MidCap 400"``, or
            ``"S&P SmallCap 600"``.
    """

    ticker: str
    cik: int | None
    security: str
    gics_sector: str
    gics_sub_industry: str
    index: str
