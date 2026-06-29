r"""Contain code to explore a document search pipeline."""

from __future__ import annotations

import logging
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

from glyphik.pipeline import BasePipeline, DocumentIndexingPipeline

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


logger: logging.Logger = logging.getLogger(__name__)


def create_fake_documents() -> list[Document]:
    """Generate fake input documents with metadata."""
    logger.info("-> Creating fake documents...")

    raw_data = [
        {
            "text": "The latest Mars rover, 'Perseverance', landed in Jezero Crater. Its primary mission is to seek signs of ancient life and collect samples of rock and regolith.",
            "source": "space_news_01.txt",
            "category": "Science",
        },
        {
            "text": "Python 3.12 introduces several performance improvements, including a more efficient garbage collector and faster comprehensions. It continues to be the dominant language for AI.",
            "source": "tech_blog_v3.md",
            "category": "Technology",
        },
        {
            "text": "To bake the perfect sourdough bread, you need a healthy starter, high-protein flour, water, and salt. The fermentation process requires precise temperature control and time.",
            "source": "recipe_book.pdf",
            "category": "Cooking",
        },
        {
            "text": "NASA's James Webb Space Telescope has captured stunning new images of the Orion Nebula, revealing previously hidden stars forming in dense clouds of gas and dust.",
            "source": "space_news_02.txt",
            "category": "Science",
        },
    ]

    # Convert raw data into LangChain Document objects
    documents = []
    for item in raw_data:
        doc = Document(
            page_content=item["text"],
            metadata={"source": item["source"], "category": item["category"]},
        )
        documents.append(doc)

    return documents


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", **kwargs: Any) -> Embeddings:
    r"""Return the embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)


def get_pipeline(base_dir: Path) -> BasePipeline:
    r"""Define the pipeline."""
    return DocumentIndexingPipeline(
        loader=DocumentListLoader(create_fake_documents()),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ".", " "]
        ),
        vector_store=Chroma(
            collection_name="foo",
            embedding_function=get_embedding_model(
                cache_folder=base_dir.joinpath("cache/embeddings").as_posix()
            ),
        ),
    )


def main() -> None:
    r"""Define the main function."""
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"

    pipeline = get_pipeline(base_dir)
    logger.info(pipeline)
    vector_store = pipeline.execute()
    inspect_embeddings(vector_store)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO)
    main()
