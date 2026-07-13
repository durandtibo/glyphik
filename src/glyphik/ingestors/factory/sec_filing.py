r"""Provide a concrete factory that ingests SEC filings for a set of
companies into a DuckDB-backed document store."""

from __future__ import annotations

__all__ = ["SecFilingIngestorFactory"]

from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.path import sanitize_path
from zenpyre.data_processors import SequenceProcessor
from zenpyre.document_stores import BaseDocumentStore
from zenpyre.ingestors import InMemoryIngestor
from zenpyre.ingestors.factory.base import BaseIngestorFactory

from glyphik.data.sp1500 import get_company_identifiers
from glyphik.data_processors import SecFilingRecordToDocumentProcessor
from glyphik.document_stores.factory import SecFilingDocumentStoreFactory
from glyphik.ingestors import SecFilingDocumentStoreIngestor, SecFilingIngestor
from glyphik.utils.dates import coerce_to_date

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path
    from typing import Self

    from zenpyre.ingestors.base import BaseIngestor

    from glyphik.data.sec import CompanyIdentifier


class SecFilingIngestorFactory(BaseIngestorFactory[BaseDocumentStore], MultilineDisplayMixin):
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
        batch_size: The number of filings to process and add to the
            document store in each batch, forwarded to
            :class:`~glyphik.ingestors.SecFilingDocumentStoreIngestor`.
            Defaults to ``32``.
        raise_on_error: If ``True`` (default), an exception raised
            while converting a filing record to a document is
            propagated and processing stops immediately. If
            ``False``, records that fail are logged and skipped.
            Forwarded to
            :class:`~zenpyre.data_processors.SequenceProcessor`.
        max_workers: The number of worker threads used to convert
            filing records to documents concurrently. Defaults to
            ``0``, which processes records sequentially. Forwarded
            to :class:`~zenpyre.data_processors.SequenceProcessor`.

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
        batch_size: int = 32,
        raise_on_error: bool = True,
        max_workers: int = 0,
    ) -> None:
        self._companies = list(companies)
        self._base_dir = sanitize_path(base_dir)
        self._start_date = coerce_to_date(start_date)
        self._end_date = coerce_to_date(end_date)
        self._forms = forms
        self._batch_size = batch_size
        self._raise_on_error = raise_on_error
        self._max_workers = max_workers

    def make_ingestor(self) -> BaseIngestor[BaseDocumentStore]:
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
            processor=SequenceProcessor(
                SecFilingRecordToDocumentProcessor(),
                raise_on_error=self._raise_on_error,
                max_workers=self._max_workers,
            ),
            batch_size=self._batch_size,
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "companies": len(self._companies),
            "base_dir": self._base_dir,
            "start_date": self._start_date,
            "end_date": self._end_date,
            "forms": self._forms,
            "batch_size": self._batch_size,
            "raise_on_error": self._raise_on_error,
            "max_workers": self._max_workers,
        }

    @classmethod
    def from_sp1500(
        cls,
        base_dir: Path | str,
        start_date: date | str,
        end_date: date | str,
        forms: Sequence[str],
        max_companies: int | None = None,
        batch_size: int = 32,
        raise_on_error: bool = True,
        max_workers: int = 0,
    ) -> Self:
        """Create a factory that ingests SEC filings for S&P 1500
        companies.

        The set of companies is resolved from an SP1500 company
        identifier file located at
        ``base_dir / "SP1500" / "company_identifier.json"`` (via
        :func:`~glyphik.data.sp1500.get_company_identifiers`),
        optionally truncated to the first ``max_companies`` entries.

        Args:
            base_dir: The base directory under which the SP1500
                company identifier file is read, and downloaded
                filings and the resulting document store are
                written. Accepts a :class:`~pathlib.Path` or a
                :class:`str`, which is sanitized via
                :func:`~coola.utils.path.sanitize_path`.
            start_date: The earliest filing date to include, as a
                :class:`~datetime.date` or an ISO-formatted
                :class:`str`.
            end_date: The latest filing date to include, as a
                :class:`~datetime.date` or an ISO-formatted
                :class:`str`.
            forms: The SEC form types to ingest (e.g. ``"10-K"``,
                ``"10-Q"``).
            max_companies: If set, only the first ``max_companies``
                companies from the resolved SP1500 list are
                ingested. If ``None``, all companies are ingested.
            batch_size: The number of filings to process and add to
                the document store in each batch. Forwarded to the
                constructor. Defaults to ``32``.
            raise_on_error: If ``True``, an exception raised while
                converting a filing record to a document propagates
                immediately. If ``False``, failing records are
                logged and skipped. Forwarded to the constructor.
            max_workers: The number of worker threads used to
                convert filing records to documents concurrently.
                Forwarded to the constructor. Defaults to ``0``.

        Returns:
            A configured ``SecFilingIngestorFactory`` for S&P 1500
            companies.

        Example:
            ```pycon
            >>> from pathlib import Path
            >>> from glyphik.ingestors.factory import SecFilingIngestorFactory
            >>> factory = SecFilingIngestorFactory.from_sp1500(  # doctest: +SKIP
            ...     base_dir=Path("/tmp/my_app"),
            ...     start_date="2023-01-01",
            ...     end_date="2023-12-31",
            ...     forms=["10-K"],
            ...     max_companies=50,
            ... )
            >>> ingestor = factory.make_ingestor()  # doctest: +SKIP

            ```
        """
        path = sanitize_path(base_dir)
        companies = get_company_identifiers(path / "SP1500" / "company_identifier.json")
        if max_companies is not None:
            companies = companies[:max_companies]
        return cls(
            companies=companies,
            base_dir=base_dir,
            start_date=start_date,
            end_date=end_date,
            forms=forms,
            batch_size=batch_size,
            raise_on_error=raise_on_error,
            max_workers=max_workers,
        )
