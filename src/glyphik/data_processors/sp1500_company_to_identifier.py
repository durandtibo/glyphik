r"""Define a processor that converts an Sp1500 Company object into a
CompanyIdentifier."""

from __future__ import annotations

__all__ = ["Sp1500CompanyToIdentifierProcessor"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from zenpyre.data_processors import BaseProcessor

from glyphik.data.sec import CompanyIdentifier

if TYPE_CHECKING:
    from glyphik.data.sp1500 import Company


class Sp1500CompanyToIdentifierProcessor(
    BaseProcessor["Company", CompanyIdentifier], InlineDisplayMixin
):
    """Convert an :class:`~glyphik.data.sp1500.Company` object into a
    :class:`~glyphik.data.sec.CompanyIdentifier`.

    If ``data.cik`` is not available (e.g. the S&P MidCap 400 source
    table does not include a CIK column), the CIK is resolved from the
    ticker via :meth:`~glyphik.data.sec.CompanyIdentifier.from_ticker`.

    Example:
        ```pycon
        >>> from glyphik.data.sp1500 import Company
        >>> from glyphik.data_processors import Sp1500CompanyToIdentifierProcessor
        >>> processor = Sp1500CompanyToIdentifierProcessor()
        >>> company = Company(
        ...     ticker="AAPL",
        ...     cik=320193,
        ...     security="Apple Inc.",
        ...     gics_sector="Information Technology",
        ...     gics_sub_industry="Technology Hardware",
        ...     index="S&P 500",
        ... )
        >>> processor.process(company)
        CompanyIdentifier(cik=320193, ticker='AAPL')

        ```
    """

    def process(self, data: Company) -> CompanyIdentifier:
        """Convert a :class:`~glyphik.data.sp1500.Company` object into
        a :class:`~glyphik.data.sec.CompanyIdentifier`.

        Args:
            data: The :class:`~glyphik.data.sp1500.Company` object to
                convert.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If ``data.cik`` is ``None`` and no CIK can be
                found for ``data.ticker``.
        """
        if data.cik is None:
            return CompanyIdentifier.from_ticker(data.ticker)
        return CompanyIdentifier(cik=data.cik, ticker=data.ticker)

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}
