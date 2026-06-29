r"""Provide a batch pipeline for indexing large document collections
into a vector store."""

from __future__ import annotations

__all__ = ["BatchDocumentIndexingPipeline"]

import logging
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from langchain_core.vectorstores import VectorStore

from glyphik.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import TextSplitter

logger: logging.Logger = logging.getLogger(__name__)


class BatchDocumentIndexingPipeline(BasePipeline[VectorStore], MultilineDisplayMixin):
    """A pipeline that indexes large document collections into a vector
    store by processing documents in batches.

    Unlike :class:`~glyphik.pipeline.DocumentIndexingPipeline`, this
    pipeline uses :meth:`~langchain_core.document_loaders.BaseLoader.lazy_load`
    to iterate over documents lazily, avoiding loading the entire
    collection into memory at once. Documents are split and indexed in
    batches of ``batch_size``, making it suitable for large datasets
    that cannot fit in memory.

    Args:
        loader: A :class:`~langchain_core.document_loaders.BaseLoader`
            used to lazily load raw documents.
        text_splitter: A :class:`~langchain_text_splitters.TextSplitter`
            used to split documents into chunks.
        vector_store: A :class:`~langchain_core.vectorstores.VectorStore`
            to index the chunks into.
        batch_size: Number of documents to process per batch.
            Defaults to ``32``.

    Example:
        ```pycon
        >>> from glyphik.pipeline import BatchDocumentIndexingPipeline
        >>> pipeline = BatchDocumentIndexingPipeline(
        ...     loader=loader,
        ...     text_splitter=text_splitter,
        ...     vector_store=vector_store,
        ...     batch_size=64,
        ... )
        >>> vector_store = pipeline.execute()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        loader: BaseLoader,
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self._loader = loader
        self._text_splitter = text_splitter
        self._vector_store = vector_store
        self._batch_size = batch_size

    def execute(self) -> VectorStore:
        """Run the pipeline in batches and return the populated vector
        store.

        Lazily loads documents via the loader, processes them in batches
        of ``batch_size``, splits each batch into chunks, indexes the
        chunks into the vector store, and returns the vector store.

        Returns:
            The populated :class:`~langchain_core.vectorstores.VectorStore`
            instance.
        """
        logger.info("Starting BatchDocumentIndexingPipeline...")
        total_docs = 0
        total_chunks = 0

        batch: list[Document] = []
        for doc in self._loader.lazy_load():
            batch.append(doc)
            if len(batch) >= self._batch_size:
                chunks = self._split_and_index(batch)
                total_docs += len(batch)
                total_chunks += len(chunks)
                batch = []

        if batch:
            chunks = self._split_and_index(batch)
            total_docs += len(batch)
            total_chunks += len(chunks)

        logger.info(
            "BatchDocumentIndexingPipeline complete. Processed %s documents into %s chunks.",
            f"{total_docs:,}",
            f"{total_chunks:,}",
        )
        return self._vector_store

    def _split_and_index(self, documents: list[Document]) -> list[Document]:
        """Split a batch of documents and index the resulting chunks.

        Args:
            documents: The list of
                :class:`~langchain_core.documents.Document` instances
                to split and index.

        Returns:
            A list of chunked
            :class:`~langchain_core.documents.Document` instances that
            were added to the vector store.
        """
        logger.info("Processing batch of %s documents...", f"{len(documents):,}")
        chunks = self._text_splitter.split_documents(documents)
        self._vector_store.add_documents(chunks)
        logger.info("Indexed %s chunks.", f"{len(chunks):,}")
        return chunks

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "loader": self._loader,
            "text_splitter": self._text_splitter,
            "vector_store": self._vector_store,
            "batch_size": self._batch_size,
        }
