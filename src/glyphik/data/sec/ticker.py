r"""Provide a utility to fetch a ticker symbol from a CIK number."""

from __future__ import annotations

__all__ = ["fetch_cik_from_ticker"]

import logging
from functools import lru_cache

from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    from edgar import Company
    from edgar.entity import CompanyNotFoundError
else:  # pragma: no cover
    from glyphik.utils.fallback.edgar import Company, CompanyNotFoundError

logger: logging.Logger = logging.getLogger(__name__)


def fetch_cik_from_ticker(ticker: str) -> int | None:
    """Return the SEC CIK number for a given ticker symbol.

    Results are cached with an LRU cache of size 256 to avoid repeated
    network calls for the same ticker within a session.

    Looks up the company via edgar and returns its CIK.  Returns
    ``None`` if the company cannot be found.

    Args:
        ticker: The stock ticker symbol to look up (e.g. ``"AAPL"``).
            Matching is case-insensitive — the ticker is upper-cased
            before lookup so that ``"aapl"`` and ``"AAPL"`` hit the
            same cache entry.

    Returns:
        The SEC CIK number (e.g. ``320193``), or ``None`` if it cannot
        be determined.

    Example:
        ```pycon
        >>> from glyphik.data.sec import fetch_cik_from_ticker
        >>> cik = fetch_cik_from_ticker("AAPL")  # doctest: +SKIP
        >>> cik  # doctest: +SKIP
        320193

        ```
    """
    return _fetch_cik_from_ticker(ticker.upper())


@lru_cache(maxsize=256)
def _fetch_cik_from_ticker(ticker: str) -> int | None:
    """Fetch the CIK for a normalised, upper-cased ticker.

    Takes a normalised upper-case ticker so that ``"aapl"`` and
    ``"AAPL"`` hit the same cache entry.

    Args:
        ticker: The upper-cased stock ticker symbol to look up
            (e.g. ``"AAPL"``).

    Returns:
        The SEC CIK number (e.g. ``320193``), or ``None`` if it cannot
        be determined.
    """
    try:
        return Company(ticker).cik
    except CompanyNotFoundError:
        logger.debug("Company not found for ticker %s", ticker)
        return None
