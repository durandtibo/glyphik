r"""Provide code to explore a document search pipelines."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zenpyre.document_loaders import DocumentListLoader
from zenpyre.embeddings.chroma import inspect_embeddings
from zenpyre.utils.rich import configure_rich_logging

from glyphik.pipelines import (
    BasePipeline,
    DocumentIndexingPipeline,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger: logging.Logger = logging.getLogger(__name__)

# Suppress HuggingFace/Chroma warnings for a cleaner console output
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_fake_documents() -> list[Document]:
    """Generate fake input documents with metadata."""
    logger.info("Creating fake documents...")
    return [
        Document(
            page_content=text,
            metadata={"source": source, "category": category},
        )
        for text, source, category in [
            (
                "The latest Mars rover, 'Perseverance', landed in Jezero Crater. Its primary mission is to seek signs of ancient life and collect samples of rock and regolith.",
                "space_news_01.txt",
                "Science",
            ),
            (
                "Python 3.12 introduces several performance improvements, including a more efficient garbage collector and faster comprehensions. It continues to be the dominant language for AI.",
                "tech_blog_v3.md",
                "Technology",
            ),
            (
                "To bake the perfect sourdough bread, you need a healthy starter, high-protein flour, water, and salt. The fermentation process requires precise temperature control and time.",
                "recipe_book.pdf",
                "Cooking",
            ),
            (
                "NASA's James Webb Space Telescope has captured stunning new images of the Orion Nebula, revealing previously hidden stars forming in dense clouds of gas and dust.",
                "space_news_02.txt",
                "Science",
            ),
        ]
    ]


def create_large_fake_documents(n: int = 1000) -> Iterator[Document]:
    """Generate a large number of fake documents with metadata.

    Yields documents lazily to avoid loading all ``n`` documents into
    memory at once, making it suitable for use with
    :class:`~glyphik.pipelines.BatchDocumentIndexingPipeline`.

    Args:
        n: Number of documents to generate. Defaults to ``1000``.

    Yields:
        A :class:`~langchain_core.documents.Document` with fake content
        and metadata.
    """
    templates = [
        (
            "The latest Mars rover, 'Perseverance', landed in Jezero Crater. Its primary mission is to seek signs of ancient life and collect samples of rock and regolith. Document number {i}.",
            "space_news_{i:05d}.txt",
            "Science",
        ),
        (
            "Python 3.12 introduces several performance improvements, including a more efficient garbage collector and faster comprehensions. It continues to be the dominant language for AI. This is entry {i} in the tech blog.",
            "tech_blog_{i:05d}.md",
            "Technology",
        ),
        (
            "To bake the perfect sourdough bread, you need a healthy starter, high-protein flour, water, and salt. The fermentation process requires precise temperature control and time.",
            "recipe_book_{i:05d}.pdf",
            "Cooking",
        ),
        (
            "NASA's James Webb Space Telescope has captured stunning new images of the Orion Nebula, revealing previously hidden stars forming in dense clouds of gas and dust.",
            "space_news_{i:05d}.txt",
            "Science",
        ),
    ]
    for i in range(n):
        text, source, category = templates[i % len(templates)]
        yield Document(
            page_content=text.format(i=i),
            metadata={"source": source.format(i=i), "category": category},
        )


def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2", **kwargs: Any
) -> HuggingFaceEmbeddings:
    """Return the HuggingFace embedding model.

    Passes ``local_files_only=True`` to avoid redundant network
    validation requests when the model is already cached locally.
    """
    model_kwargs = {"local_files_only": True, **kwargs.pop("model_kwargs", {})}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, **kwargs)


def get_vector_store(base_dir: Path) -> Chroma:
    """Return a persisted Chroma vector store."""
    return Chroma(
        collection_name="foo",
        embedding_function=get_embedding_model(
            cache_folder=base_dir.joinpath("cache/embeddings").as_posix()
        ),
        persist_directory=base_dir.joinpath("cache/chroma").as_posix(),
    )


def get_pipeline(vector_store: Chroma) -> BasePipeline:
    """Define the document indexing pipelines."""
    return DocumentIndexingPipeline(
        document_loader=DocumentListLoader(list(create_large_fake_documents())),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ".", " "]
        ),
        vector_store=vector_store,
    )


def main() -> None:
    """Run the document indexing pipelines and inspect the vector
    store."""
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"

    vector_store = get_vector_store(base_dir)
    pipeline = get_pipeline(vector_store)
    logger.info(pipeline)
    pipeline.execute()

    inspect_embeddings(vector_store)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
