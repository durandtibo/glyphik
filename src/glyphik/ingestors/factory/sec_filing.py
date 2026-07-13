r"""Provide a concrete factory that ingests SEC filings for a set of
companies into a DuckDB-backed document store."""

from __future__ import annotations

__all__ = ["SecFilingIngestorFactory"]

from datetime import date
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.path import sanitize_path
from zenpyre.data_processors import SequenceProcessor
from zenpyre.ingestors import InMemoryIngestor
from zenpyre.ingestors.factory.base import BaseIngestorFactory

from glyphik.data_processors import SecFilingRecordToDocumentProcessor
from glyphik.document_stores.factory import SecFilingDocumentStoreFactory
from glyphik.ingestors import SecFilingDocumentStoreIngestor, SecFilingIngestor

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from zenpyre.ingestors.base import BaseIngestor

    from glyphik.data.sec import CompanyIdentifier


class SecFilingIngestorFactory(BaseIngestorFactory[Any], MultilineDisplayMixin):
    """A concrete BaseIngestor factory that builds an ingestor for
    downloading SEC filings for a set of companies and storing them
    in a DuckDB-backed :class:`~zenpyre.document_stores.BaseDocumentStore`.

    Each call to :meth:`make_ingestor` constructs a fresh
    :class:`~glyphik.ingestors.SecFilingDocumentStoreIngestor` that
    fetches filings of the given ``forms`` for ``companies`` between
    ``start_date`` and ``end_date``, converts each filing record to a
    document via
    :class:`~glyphik.data_processors.SecFilingRecordToDocumentProcessor`,
    and persists the results to a
    :class:`~glyphik.document_stores.factory.SecFilingDocumentStoreFactory`-backed
    document store rooted at ``base_dir``.

    Args:
        companies: The companies whose SEC filings should be ingested.
        base_dir: The base directory under which downloaded filings
            and the resulting document store are written. Accepts a
            :class:`~pathlib.Path` or a :class:`str`, which is
            sanitized via :func:`~coola.utils.path.sanitize_path`.
        start_date: The earliest filing date to include, as a
            :class:`~datetime.date` or an ISO-formatted :class:`str`.
        end_date: The latest filing date to include, as a
            :class:`~datetime.date` or an ISO-formatted :class:`str`.
        forms: The SEC form types to ingest (e.g. ``"10-K"``,
            ``"10-Q"``).

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from glyphik.data.sec import CompanyIdentifier
        >>> from glyphik.ingestors.factory import SecFilingIngestorFactory
        >>> factory = SecFilingIngestorFactory(  # doctest: +SKIP
        ...     companies=[CompanyIdentifier(ticker="AAPL")],
        ...     base_dir=Path("/tmp/my_app"),
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31",
        ...     forms=["10-K"],
        ... )
        >>> ingestor = factory.make_ingestor()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        companies: Sequence[CompanyIdentifier],
        base_dir: Path | str,
        start_date: date | str,
        end_date: date | str,
        forms: Sequence[str],
    ) -> None:
        self._companies = list(companies)
        self._base_dir = sanitize_path(base_dir)
        self._start_date = (
            start_date if isinstance(start_date, date) else date.fromisoformat(start_date)
        )
        self._end_date = end_date if isinstance(end_date, date) else date.fromisoformat(end_date)
        self._forms = forms

    def make_ingestor(self) -> BaseIngestor[Any]:
        document_store = SecFilingDocumentStoreFactory(
            base_dir=self._base_dir
        ).make_document_store()

        return SecFilingDocumentStoreIngestor(
            filing_ingestor=SecFilingIngestor(
                company_ingestor=InMemoryIngestor(self._companies),
                output_dir=self._base_dir / "sec" / "filing",
                start_date=self._start_date,
                end_date=self._end_date,
                forms=self._forms,
            ),
            document_store=document_store,
            processor=SequenceProcessor(SecFilingRecordToDocumentProcessor()),
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "companies": len(self._companies),
            "base_dir": self._base_dir,
            "start_date": self._start_date,
            "end_date": self._end_date,
            "forms": self._forms,
        }
