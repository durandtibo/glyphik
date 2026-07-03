r"""Define an ingestor that indexes documents from a document store into
a vector store."""

from __future__ import annotations

__all__ = ["DocumentStoreIndexingIngestor"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.format import str_time_human
from langchain_core.vectorstores import VectorStore
from zenpyre.document_loaders import DocumentStoreLoader
from zenpyre.ingestors.base import BaseIngestor

from glyphik.pipeline import DocumentIndexingPipeline

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter
    from zenpyre.document_stores import BaseDocumentStore


logger: logging.Logger = logging.getLogger(__name__)


class DocumentStoreIndexingIngestor(BaseIngestor[VectorStore], MultilineDisplayMixin):
    """Ingestor that indexes documents from a document store into a
    vector store.

    Retrieves a populated :class:`~zenpyre.document_stores.BaseDocumentStore`
    via ``document_store_ingestor``, wraps it in a
    :class:`~zenpyre.document_loaders.DocumentStoreLoader`, and runs it
    through a :class:`~glyphik.pipeline.DocumentIndexingPipeline` to
    split and index the documents into ``vector_store`` in batches.

    Args:
        document_store_ingestor: An ingestor that provides the
            populated :class:`~zenpyre.document_stores.BaseDocumentStore`
            of documents to index.
        text_splitter: A :class:`~langchain_text_splitters.TextSplitter`
            used to split documents into chunks before indexing.
        vector_store: The :class:`~langchain_core.vectorstores.VectorStore`
            to index the chunks into.
        batch_size: The number of documents to process and index in
            each batch. Defaults to ``32``.
    """

    def __init__(
        self,
        document_store_ingestor: BaseIngestor[BaseDocumentStore],
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self._document_store_ingestor = document_store_ingestor
        self._text_splitter = text_splitter
        self._vector_store = vector_store
        self._batch_size = batch_size

    def ingest(self) -> VectorStore:
        """Ingest documents into the vector store.

        Fetches the populated document store from
        ``document_store_ingestor``, then splits and indexes its
        documents into ``vector_store`` in batches of ``batch_size``
        via :class:`~glyphik.pipeline.DocumentIndexingPipeline`.

        Returns:
            The populated :class:`~langchain_core.vectorstores.VectorStore`.
        """
        logger.info("Starting to index documents...")
        t_start = time.perf_counter()

        document_store = self._document_store_ingestor.ingest()

        pipeline = DocumentIndexingPipeline(
            document_loader=DocumentStoreLoader(document_store),
            text_splitter=self._text_splitter,
            vector_store=self._vector_store,
            batch_size=self._batch_size,
        )
        pipeline.execute()

        logger.info(
            "Ingestion complete. Indexed %s documents in %s",
            f"{document_store.count():,}",
            str_time_human(time.perf_counter() - t_start),
        )

        return self._vector_store

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "document_store_ingestor": self._document_store_ingestor,
            "text_splitter": self._text_splitter,
            "vector_store": self._vector_store,
            "batch_size": self._batch_size,
        }
