r"""Provide a convenience function to fetch S&P 1500 company
identifiers."""

from __future__ import annotations

__all__ = ["get_sp1500_company_identifiers"]

import logging
from typing import TYPE_CHECKING

from glyphik.data.sec import CompanyIdentifier
from glyphik.data.sp1500.fetch import load_or_fetch_sp1500_companies

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def get_sp1500_company_identifiers(path: Path | str | None = None) -> list[CompanyIdentifier]:
    r"""Fetch the identifiers (ticker + CIK) of every S&P 1500 company.

    Loads the S&P 1500 company list via
    :func:`~glyphik.data.sp1500.fetch.load_or_fetch_sp1500_companies`
    (from a cached JSON file if ``path`` is given and exists, or freshly
    fetched from Wikipedia otherwise), then converts each
    :class:`~glyphik.data.sp1500.Company` into a
    :class:`~glyphik.data.sec.CompanyIdentifier`.

    Unlike ``Company``, whose ``cik`` field may be ``None`` (Wikipedia
    does not list CIKs for S&P MidCap 400 constituents),
    ``CompanyIdentifier``'s ``cik`` field is required. This function
    always fetches with ``find_missing_ciks=True``, so a freshly
    fetched company list will already have every CIK filled in. As a
    second line of defense -- since ``find_missing_ciks`` has no effect
    when loading from an existing cache, so a stale cache written
    before CIKs were filled in (or written by a caller who explicitly
    disabled that step) can still contain a ``None`` CIK -- any company
    still missing a CIK at this point is resolved individually via
    :meth:`~glyphik.data.sec.CompanyIdentifier.from_ticker`, which
    performs a live ticker-to-CIK lookup.

    Args:
        path: The path to the JSON cache file used by
            :func:`~glyphik.data.sp1500.fetch.load_or_fetch_sp1500_companies`.
            If ``None`` (the default), caching is disabled entirely:
            the company list is freshly fetched from Wikipedia on every
            call, and nothing is loaded from or saved to disk.

    Returns:
        A list of :class:`~glyphik.data.sec.CompanyIdentifier`
            instances, one per S&P 1500 company, in the same order as
            returned by ``load_or_fetch_sp1500_companies``.

    Raises:
        ValueError: If a company's CIK is missing from the loaded data
            and no CIK can be found for its ticker via
            :meth:`~glyphik.data.sec.CompanyIdentifier.from_ticker`
            (e.g. a stale or delisted ticker).

    Example:
        ```pycon
        >>> from glyphik.data.sp1500 import get_sp1500_company_identifiers
        >>> identifiers = get_sp1500_company_identifiers()  # doctest: +SKIP

        ```
    """
    companies = load_or_fetch_sp1500_companies(path=path, find_missing_ciks=True)

    identifiers: list[CompanyIdentifier] = []
    for company in companies:
        if company.cik is not None:
            identifier = CompanyIdentifier(cik=company.cik, ticker=company.ticker)
        else:
            logger.info(
                "Company %r has no CIK in the loaded data; looking it up by ticker...",
                company.ticker,
            )
            identifier = CompanyIdentifier.from_ticker(company.ticker)
        identifiers.append(identifier)
    return identifiers
