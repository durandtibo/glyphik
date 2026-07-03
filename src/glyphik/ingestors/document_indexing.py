r"""Define an ingestor that converts SEC filings to documents and stores
them."""

from __future__ import annotations

__all__ = ["DocumentIndexingIngestor"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.utils.format import str_time_human
from langchain_core.vectorstores import VectorStore
from zenpyre.document_stores import BaseDocumentStore
from zenpyre.ingestors.base import BaseIngestor

from glyphik.pipeline import DocumentIndexingPipeline

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

    from glyphik.data.sec import SecFilingRecord

logger: logging.Logger = logging.getLogger(__name__)


class DocumentIndexingIngestor(BaseIngestor[BaseDocumentStore], VectorStore):
    """Ingestor that converts SEC filing records to LangChain documents
    and adds them to a document store.

    Retrieves filing records via ``document_store_ingestor``, skips records
    already present in ``store`` (using
    :meth:`~zenpyre.document_stores.BaseDocumentStore.check_ids` for
    deduplication), converts the remaining records to
    :class:`~langchain_core.documents.Document` instances via
    ``processor``, and adds them to ``store`` in batches.

    Args:
        document_store_ingestor: An ingestor that provides the list of
            :class:`~glyphik.data.sec.SecFilingRecord` instances to
            process.
        processor: A processor that converts a list of
            :class:`~glyphik.data.sec.SecFilingRecord` instances to a
            list of :class:`~langchain_core.documents.Document`
            instances.
        store: The document store to add new documents to.
        batch_size: The number of filings to process and add to the
            store in each batch.  Defaults to ``32``.
    """

    def __init__(
        self,
        document_store_ingestor: BaseIngestor[list[SecFilingRecord]],
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self._document_store_ingestor = document_store_ingestor
        self._text_splitter = text_splitter
        self._vector_store = vector_store
        self._batch_size = batch_size

    def ingest(self) -> VectorStore:
        """Ingest SEC filing records into the document store.

        Fetches records from ``document_store_ingestor``, deduplicates against
        ``store``, converts new records to documents via ``processor``,
        and adds them to ``store`` in batches of ``batch_size``.

        Returns:
            The populated :class:`~zenpyre.document_stores.BaseDocumentStore`.
        """
        logger.info("Starting to ingest filing documents to store...")
        t_start = time.perf_counter()

        self._document_store_ingestor.ingest()

        DocumentIndexingPipeline()

        logger.info(
            "%s filing documents have been ingested to the store in %s",
            f"{len(filings):,}",
            str_time_human(time.perf_counter() - t_start),
        )

        return self._vector_store

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "document_store_ingestor": self._document_store_ingestor,
            "processor": self._processor,
            "store": self._store,
            "batch_size": self._batch_size,
        }
