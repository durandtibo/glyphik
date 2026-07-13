r"""Provide a batch pipelines for indexing large document collections
into a vector store."""

from __future__ import annotations

__all__ = ["DocumentIndexingPipeline"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.format import str_time_human
from langchain_core.vectorstores import VectorStore
from zenpyre.documents import assign_ids, copy_ids_to_metadata
from zenpyre.utils.rich import make_spinner

from glyphik.pipelines.base import BasePipeline

if TYPE_CHECKING:
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import TextSplitter

logger: logging.Logger = logging.getLogger(__name__)


class DocumentIndexingPipeline(BasePipeline[VectorStore], MultilineDisplayMixin):
    """A pipelines that indexes large document collections into a vector
    store by processing documents in batches.

    Unlike :class:`~glyphik.pipelines.DocumentIndexingPipeline`, this
    pipelines uses :meth:`~langchain_core.document_loaders.BaseLoader.lazy_load`
    to iterate over documents lazily, avoiding loading the entire
    collection into memory at once. Documents are split and indexed in
    batches of ``batch_size``, making it suitable for large datasets
    that cannot fit in memory.

    Args:
        document_loader: A :class:`~langchain_core.document_loaders.BaseLoader`
            used to lazily load raw documents.
        text_splitter: A :class:`~langchain_text_splitters.TextSplitter`
            used to split documents into chunks.
        vector_store: A :class:`~langchain_core.vectorstores.VectorStore`
            to index the chunks into.
        batch_size: Number of documents to process per batch.
            Defaults to ``32``.

    Example:
        ```pycon
        >>> from glyphik.pipelines import DocumentIndexingPipeline
        >>> pipelines = DocumentIndexingPipeline(
        ...     document_loader=loader,
        ...     text_splitter=text_splitter,
        ...     vector_store=vector_store,
        ...     batch_size=64,
        ... )
        >>> vector_store = pipelines.run()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        document_loader: BaseLoader,
        text_splitter: TextSplitter,
        vector_store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self._document_loader = document_loader
        self._text_splitter = text_splitter
        self._vector_store = vector_store
        self._batch_size = batch_size

    def run(self) -> VectorStore:
        """Run the pipelines in batches and return the populated vector
        store.

        Lazily loads documents via the loader, processes them in batches
        of ``batch_size``, splits each batch into chunks, indexes the
        chunks into the vector store, and returns the vector store.

        Returns:
            The populated :class:`~langchain_core.vectorstores.VectorStore`
            instance.
        """
        logger.info("Starting document indexing pipelines...")
        t_start = time.perf_counter()
        total_docs = 0
        total_chunks = 0

        batch: list[Document] = []
        with make_spinner(transient=True) as progress:
            task = progress.add_task("Indexing documents...", total=None)
            for doc in self._document_loader.lazy_load():
                batch.append(doc)
                progress.advance(task)
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
            "Indexing complete. Processed %s documents into %s chunks in %s",
            f"{total_docs:,}",
            f"{total_chunks:,}",
            str_time_human(time.perf_counter() - t_start),
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
        logger.debug("Processing batch of %s documents...", f"{len(documents):,}")
        documents = copy_ids_to_metadata(documents, metadata_key="source_document_id")
        chunks = self._text_splitter.split_documents(documents)
        chunks = assign_ids(chunks)
        self._vector_store.add_documents(chunks)
        logger.debug("Indexed %s chunks", f"{len(chunks):,}")
        return chunks

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "document_loader": self._document_loader,
            "text_splitter": self._text_splitter,
            "vector_store": self._vector_store,
            "batch_size": self._batch_size,
        }
