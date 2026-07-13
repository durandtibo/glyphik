r"""Provide code to explore a document search pipelines."""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zenpyre.data_processors import SequenceProcessor
from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.ingestors import FirstNIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging, print_document

from glyphik.data.sec import SecForm
from glyphik.data_processors import (
    SecFilingRecordToDocumentProcessor,
    Sp1500CompanyToIdentifierProcessor,
)
from glyphik.ingestors import (
    DocumentStoreIndexingIngestor,
    SecFilingDocumentStoreIngestor,
    SecFilingIngestor,
    Sp1500CompanyIngestor,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
    from zenpyre.ingestors import BaseIngestor

logger: logging.Logger = logging.getLogger(__name__)

# Suppress HuggingFace/Chroma warnings for a cleaner console output
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the HuggingFace embedding model.

    Passes ``local_files_only=True`` to avoid redundant network
    validation requests when the model is already cached locally.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"local_files_only": True}
    )


def get_document_store(base_dir: Path, **kwargs: Any) -> DuckDBDocumentStore:
    """Return a persisted DuckDB document store."""
    return DuckDBDocumentStore(base_dir / "document_store" / "documents.duckdb", **kwargs)


def get_vector_store(base_dir: Path) -> Chroma:
    """Return a persisted Chroma vector store."""
    return Chroma(
        collection_name="documents",
        embedding_function=get_embedding_model(),
        persist_directory=base_dir.joinpath("vector_store").as_posix(),
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return a token-aware text splitter sized for the embedding
    model's max sequence length.

    Uses the embedding model's own tokenizer to count tokens, and caps
    ``chunk_size`` a few tokens below ``max_seq_length`` to leave room
    for special tokens (e.g. ``[CLS]``, ``[SEP]``) added during actual
    encoding.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def build_ingestor(base_dir: Path) -> BaseIngestor:
    """Build the S&P 1500 filing ingestor rooted at ``base_dir``."""
    return DocumentStoreIndexingIngestor(
        document_store_ingestor=SecFilingDocumentStoreIngestor(
            filing_ingestor=SecFilingIngestor(
                company_ingestor=ProcessorIngestor(
                    source=FirstNIngestor(
                        Sp1500CompanyIngestor(path=base_dir / "sp1500" / "companies.json"),
                        n=5,
                    ),
                    processor=SequenceProcessor(Sp1500CompanyToIdentifierProcessor()),
                ),
                output_dir=base_dir / "sec",
                start_date=date(2023, 1, 1),
                end_date=date(2026, 6, 1),
                forms=[SecForm.TEN_K, SecForm.TEN_Q],
            ),
            document_store=get_document_store(base_dir),
            processor=SequenceProcessor(SecFilingRecordToDocumentProcessor()),
        ),
        text_splitter=get_text_splitter(),
        vector_store=get_vector_store(base_dir),
        batch_size=4,
    )


def search(
    query: str, vector_store: VectorStore, search_kwargs: dict | None = None
) -> list[Document]:
    """Search the vector store and log the top matching documents."""
    logger.info("\n==================================================")
    logger.info("SEARCH QUERY: '%s'", query)

    # Provide a default configuration if nothing is passed
    if search_kwargs is None:
        search_kwargs = {"k": 2}

    logger.info("SEARCH KWARGS: %s", search_kwargs)
    logger.info("==================================================")

    # Create the retriever, passing your custom dictionary directly
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # Execute the search
    results = retriever.invoke(query)

    # Display the results
    if not results:
        logger.info("No relevant documents found")
        return []

    for doc in results:
        print_document(doc)

    return results


def main() -> None:
    """Run the document indexing pipelines and inspect the vector
    store."""
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"

    ingestor = build_ingestor(base_dir)
    logger.info("%s", ingestor)
    ingestor.ingest()
    vector_store = get_vector_store(base_dir)

    results = search(
        query="What is the income of MMM?",
        vector_store=vector_store,
        search_kwargs={"filter": {"ticker": {"$in": ["MMM"]}}},
    )

    document_store = get_document_store(base_dir, read_only=True)
    print_document(document_store.get(results[0].metadata["source_document_id"]))


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
