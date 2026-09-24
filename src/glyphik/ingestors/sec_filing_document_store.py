r"""Define an ingestor that converts SEC filings to documents and stores
them."""

from __future__ import annotations

__all__ = ["SecFilingDocumentStoreIngestor"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.batching import batchify
from coola.utils.format import str_time_human
from docculus.store import BaseDocumentStore
from zenpyre.ingestors.base import BaseIngestor

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from zenpyre.data_processors import BaseProcessor

    from glyphik.data.sec import SecFilingRecord

logger: logging.Logger = logging.getLogger(__name__)


class SecFilingDocumentStoreIngestor(BaseIngestor[BaseDocumentStore], MultilineDisplayMixin):
    """Ingestor that converts SEC filing records to LangChain documents
    and adds them to a document store.

    Retrieves filing records via ``filing_ingestor``, skips records
    already present in ``store`` (using
    :meth:`~docculus.store.BaseDocumentStore.contains_many` for
    deduplication), converts the remaining records to
    :class:`~langchain_core.documents.Document` instances via
    ``processor``, and adds them to ``store`` in batches.

    Args:
        filing_ingestor: An ingestor that provides the list of
            :class:`~glyphik.data.sec.SecFilingRecord` instances to
            process.
        processor: A processor that converts a list of
            :class:`~glyphik.data.sec.SecFilingRecord` instances to a
            list of :class:`~langchain_core.documents.Document`
            instances.
        document_store: The document store to add new documents to.
        batch_size: The number of filings to process and add to the
            store in each batch.  Defaults to ``32``.
    """

    def __init__(
        self,
        filing_ingestor: BaseIngestor[list[SecFilingRecord]],
        processor: BaseProcessor[list[SecFilingRecord], list[Document]],
        document_store: BaseDocumentStore,
        batch_size: int = 32,
    ) -> None:
        self._filing_ingestor = filing_ingestor
        self._processor = processor
        self._document_store = document_store
        self._batch_size = batch_size

    def ingest(self) -> BaseDocumentStore:
        """Ingest SEC filing records into the document store.

        Fetches records from ``filing_ingestor``, deduplicates against
        ``store``, converts new records to documents via ``processor``,
        and adds them to ``store`` in batches of ``batch_size``.

        Returns:
            The populated :class:`~docculus.store.BaseDocumentStore`.
        """
        logger.info("Starting to ingest filing documents to store...")
        t_start = time.perf_counter()

        filings = self._filing_ingestor.ingest()

        logger.info("Finding the missing documents in the store...")
        ids = [f.id for f in filings]
        present = self._document_store.contains_many(ids)
        new_filings = [f for f, is_present in zip(filings, present, strict=True) if not is_present]
        logger.info(
            "%s filings already in store, %s to add",
            f"{len(filings) - len(new_filings):,}",
            f"{len(new_filings):,}",
        )

        if new_filings:
            for batch in batchify(new_filings, size=self._batch_size):
                docs = self._processor.process(list(batch))
                self._document_store.set_many(docs)

        logger.info(
            "%s filing documents have been ingested to the store in %s",
            f"{len(filings):,}",
            str_time_human(time.perf_counter() - t_start),
        )

        return self._document_store

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "filing_ingestor": self._filing_ingestor,
            "processor": self._processor,
            "document_store": self._document_store,
            "batch_size": self._batch_size,
        }
