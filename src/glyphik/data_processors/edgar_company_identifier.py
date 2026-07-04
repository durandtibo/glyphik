r"""Define a processor that converts an edgar.Company object into a
CompanyIdentifier."""

from __future__ import annotations

__all__ = ["EdgarCompanyToIdentifierProcessor"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from zenpyre.data_processors import BaseProcessor

from glyphik.data.sec import CompanyIdentifier

if TYPE_CHECKING:
    import edgar


class EdgarCompanyToIdentifierProcessor(
    BaseProcessor["edgar.Company", CompanyIdentifier], InlineDisplayMixin
):
    """Convert an ``edgar.Company`` object into a
        :class:`~glyphik.data.sec.CompanyIdentifier`.

    Example:
    ```pycon

    >>> import edgar
    >>> from glyphik.data_processors import EdgarCompanyToIdentifierProcessor
    >>> processor = EdgarCompanyToIdentifierProcessor()
    >>> processor.process(edgar.Company("AAPL"))  # doctest: +SKIP
    CompanyIdentifier(cik=320193, ticker='AAPL')

    ```
    """

    def process(self, data: edgar.Company) -> CompanyIdentifier:
        """Convert an ``edgar.Company`` object into a
        :class:`~glyphik.data.sec.CompanyIdentifier`.

        Args:
            data: The ``edgar.Company`` object to convert.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If ``data`` has no ticker and no ticker can be
                found for its CIK.
        """
        return CompanyIdentifier.from_edgar_company(data)

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}
