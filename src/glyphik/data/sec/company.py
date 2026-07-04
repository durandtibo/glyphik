r"""Provide a utility to fetch the current S&P 1500 constituents from
Wikipedia."""

from __future__ import annotations

__all__ = ["CompanyIdentifier"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

from glyphik.data.sec.cik import fetch_ticker_from_cik
from glyphik.data.sec.ticker import fetch_cik_from_ticker

if TYPE_CHECKING:
    import edgar


@dataclass(frozen=True)
class CompanyIdentifier:
    """A single company identifier.

    Args:
        ticker: The stock ticker symbol (e.g. ``"AAPL"``).
        cik: The SEC Central Index Key (CIK).
    """

    cik: int
    ticker: str

    @classmethod
    def from_cik(cls, cik: int) -> CompanyIdentifier:
        """Build a :class:`CompanyIdentifier` by looking up the ticker
        for a CIK.

        Args:
            cik: The SEC Central Index Key (CIK) to look up.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If no ticker can be found for ``cik``.
        """
        ticker = fetch_ticker_from_cik(cik)
        if ticker is None:
            msg = f"Cannot find ticker for CIK {cik}"
            raise ValueError(msg)
        return cls(cik=cik, ticker=ticker)

    @classmethod
    def from_ticker(cls, ticker: str) -> CompanyIdentifier:
        """Build a :class:`CompanyIdentifier` by looking up the CIK for
        a ticker.

        Args:
            ticker: The stock ticker symbol to look up.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If no CIK can be found for ``ticker``.
        """
        cik = fetch_cik_from_ticker(ticker)
        if cik is None:
            msg = f"Cannot find CIK for ticker {ticker!r}"
            raise ValueError(msg)
        return cls(cik=cik, ticker=ticker)

    @classmethod
    def from_edgar_company(cls, company: edgar.Company) -> CompanyIdentifier:
        """Build a :class:`CompanyIdentifier` from an ``edgar.Company``
        object.

        If the ``edgar.Company`` does not expose a ticker directly, it is
        resolved from its CIK via :meth:`from_cik`.

        Args:
            company: The ``edgar.Company`` to build an identifier from.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If ``company`` has no ticker and no ticker can be
                found for its CIK.
        """
        ticker = company.get_ticker()
        if ticker is None:
            return cls.from_cik(company.cik)
        return cls(cik=company.cik, ticker=ticker)
