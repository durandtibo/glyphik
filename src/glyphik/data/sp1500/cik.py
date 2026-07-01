r"""Contain utilities for CIKs."""

from __future__ import annotations

__all__ = ["fill_missing_ciks"]

import dataclasses
from typing import TYPE_CHECKING

from zenpyre.utils.rich import make_progressbar

from glyphik.data.sec import fetch_cik_from_ticker

if TYPE_CHECKING:
    from glyphik.data.sp1500.company import Company


def fill_missing_ciks(companies: list[Company]) -> list[Company]:
    """Fill in missing CIK numbers for a list of S&P 1500 companies.

    For each company whose ``cik`` is ``None``, looks up its CIK via
    :func:`~glyphik.data.sec.fetch_cik_from_ticker` using its ticker
    symbol, and returns a new company instance with ``cik`` filled in.
    Companies that already have a CIK are returned unchanged.

    Args:
        companies: The list of :class:`~glyphik.data.sp1500.Company`
            instances to process.

    Returns:
        A new list of :class:`~glyphik.data.sp1500.Company`
        instances, with ``cik`` filled in wherever it was previously
        ``None`` and a matching ticker lookup succeeded.
    """
    with make_progressbar() as progress:
        task = progress.add_task("Filling missing CIKs...", total=len(companies))
        filled_companies = []
        for company in companies:
            filled_companies.append(
                dataclasses.replace(company, cik=fetch_cik_from_ticker(company.ticker))
                if company.cik is None
                else company
            )
            progress.advance(task)

    return filled_companies
