r"""Provide a pipeline for indexing documents into a vector store."""

from __future__ import annotations

__all__ = ["DocumentIndexingPipeline"]

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


class DocumentIndexingPipeline(BasePipeline[VectorStore], MultilineDisplayMixin):
    """A pipeline that loads documents, splits them into chunks, and
    indexes them into a vector store.

    Args:
        loader: A :class:`~langchain_core.document_loaders.BaseLoader`
            used to load raw documents.
        text_splitter: A :class:`~langchain_text_splitters.TextSplitter`
            used to split documents into chunks.
        vector_store: A :class:`~langchain_core.vectorstores.VectorStore`
            to index the chunks into.

    Example:
        ```pycon
        >>> from glyphik.pipeline import DocumentIndexingPipeline
        >>> pipeline = DocumentIndexingPipeline(
        ...     loader=loader,
        ...     text_splitter=text_splitter,
        ...     vector_store=vector_store,
        ... )
        ... vector_store = pipeline.execute()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        loader: BaseLoader,
        text_splitter: TextSplitter,
        vector_store: VectorStore,
    ) -> None:
        self._loader = loader
        self._text_splitter = text_splitter
        self._vector_store = vector_store

    def execute(self) -> VectorStore:
        """Run the pipeline and return the populated vector store.

        Loads documents via the loader, splits them into chunks via the
        text splitter, indexes the chunks into the vector store, and
        returns the vector store.

        Returns:
            The populated :class:`~langchain_core.vectorstores.VectorStore`
            instance.
        """
        logger.info("Starting DocumentIndexingPipeline...")
        docs = self._loader.load()
        logger.info("Loaded %s documents.", f"{len(docs):,}")
        chunks = self._split_documents(docs)

        logger.info("Adding %s chunks into vector store...", f"{len(chunks):,}")
        self._vector_store.add_documents(chunks)
        logger.info("DocumentIndexingPipeline complete.")
        return self._vector_store

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks using the text splitter.

        Args:
            documents: The list of
                :class:`~langchain_core.documents.Document` instances
                to split.

        Returns:
            A list of chunked
            :class:`~langchain_core.documents.Document` instances.
        """
        logger.info("Splitting %s documents into chunks...", f"{len(documents):,}")
        chunks = self._text_splitter.split_documents(documents)
        logger.info("Created %s chunks.", f"{len(chunks):,}")
        return chunks

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "loader": self._loader,
            "text_splitter": self._text_splitter,
            "vector_store": self._vector_store,
        }
