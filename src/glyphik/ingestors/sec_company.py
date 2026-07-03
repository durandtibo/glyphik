r"""Define an ingestor that builds SEC ``Company`` objects from CIKs or
tickers."""

from __future__ import annotations

__all__ = ["SecCompanyIngestor"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.utils.format import str_time_human
from zenpyre.ingestors.base import BaseIngestor

from glyphik.utils.imports import is_edgar_available

if TYPE_CHECKING or is_edgar_available():
    from edgar import Company
else:  # pragma: no cover
    from glyphik.utils.fallback.edgar import Company

logger: logging.Logger = logging.getLogger(__name__)


class SecCompanyIngestor(BaseIngestor[list[Any]], InlineDisplayMixin):
    """Ingestor that builds SEC ``Company`` objects.

    Wraps another ingestor that produces a list of CIK numbers and/or
    ticker symbols, and turns each one into an
    :class:`~edgar.Company` instance.

    Args:
        ingestor: The ingestor that supplies the list of CIKs and/or
            tickers used to build the ``Company`` instances.

    Example:
        ```pycon
        >>> from zenpyre.ingestors import InMemoryIngestor
        >>> from glyphik.ingestors import SecCompanyIngestor
        >>> ingestor = SecCompanyIngestor(ingestor=InMemoryIngestor(["AAPL", "NVDA"]))
        >>> companies = ingestor.ingest()  # doctest: +SKIP

        ```
    """

    def __init__(self, ingestor: BaseIngestor[list[str | int]]) -> None:
        self._ingestor = ingestor

    def ingest(self) -> list[Company]:
        """Build the list of SEC ``Company`` objects.

        Fetches the list of CIKs and/or tickers from the wrapped
        ingestor and constructs a :class:`~edgar.Company` for each one.

        Returns:
            A list of :class:`~edgar.Company` instances, one per CIK
            or ticker returned by the wrapped ingestor.
        """
        logger.info("Starting to ingest the list of SEC companies...")
        t_start = time.perf_counter()

        ciks_or_tickers = self._ingestor.ingest()
        companies = [Company(cik_or_ticker) for cik_or_ticker in ciks_or_tickers]

        logger.info(
            "%s SEC companies have been ingested in %s",
            f"{len(companies):,}",
            str_time_human(time.perf_counter() - t_start),
        )

        return companies

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"ingestor": self._ingestor}
