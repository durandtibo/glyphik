r"""Provide a utility to fetch a ticker symbol from a CIK number."""

from __future__ import annotations

__all__ = ["fetch_ticker_from_cik"]

import logging
from functools import lru_cache

from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    from edgar import Company
    from edgar.entity import CompanyNotFoundError
else:  # pragma: no cover
    from glyphik.utils.fallback.edgar import Company, CompanyNotFoundError

logger: logging.Logger = logging.getLogger(__name__)


def fetch_ticker_from_cik(cik: int | str) -> str | None:
    """Return the primary ticker symbol for a given CIK number.

    Accepts the CIK as either an integer or a string. Results are
    cached with an LRU cache of size 256 to avoid repeated network
    calls for the same CIK within a session.

    Looks up the company via edgar and returns its primary ticker via
    :meth:`~edgar.Company.get_ticker`. Returns ``None`` if the company
    has no ticker or is not found.

    Args:
        cik: The SEC Central Index Key (CIK) number to look up, as an
            integer or string (e.g. ``320193`` or ``"320193"``).

    Returns:
        The primary ticker symbol (e.g. ``"AAPL"``), or ``None`` if it
        cannot be determined.

    Example:
        ```pycon
        >>> from glyphik.data.sec import fetch_ticker_from_cik
        >>> ticker = fetch_ticker_from_cik(320193)  # doctest: +SKIP
        >>> ticker  # doctest: +SKIP
        'AAPL'

        ```
    """
    return _fetch_ticker_from_cik(int(cik))


@lru_cache(maxsize=256)
def _fetch_ticker_from_cik(cik: int) -> str | None:
    """Fetch the primary ticker for a normalised integer CIK.

    Takes a normalised integer CIK so that ``320193`` and ``"320193"``
    hit the same cache entry.

    Args:
        cik: The SEC Central Index Key (CIK) number to look up, as a
            normalised integer.

    Returns:
        The primary ticker symbol (e.g. ``"AAPL"``), or ``None`` if it
        cannot be determined.
    """
    try:
        ticker = Company(cik).get_ticker()
    except CompanyNotFoundError:
        logger.debug("Company not found for CIK %s", cik)
        return None

    if not ticker:
        logger.debug("No ticker found for CIK %s", cik)
        return None

    return ticker
