r"""Contain code to explore a document search pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zenpyre.document_loaders import DocumentListLoader
from zenpyre.utils.rich import configure_rich_logging

from glyphik.pipeline import BasePipeline, DocumentIndexingPipeline

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

# Initialize the logger
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


def inspect_embeddings(vector_store: Chroma) -> None:
    """Retrieve and display the raw embeddings from the vector store."""
    logger.info("\n--- INSTRUCTING VECTOR STORE TO REVEAL EMBEDDINGS ---")

    # Use the .get() method and explicitly request 'embeddings'
    db_data = vector_store.get(include=["embeddings", "documents", "metadatas"])

    # Extract the lists from the dictionary
    embeddings = db_data["embeddings"]
    documents = db_data["documents"]
    metadatas = db_data["metadatas"]
    ids = db_data["ids"]

    # Check if we got anything back
    if embeddings is None:
        logger.info("No embeddings found in the database.")
        return

    logger.info(f"Successfully retrieved {len(embeddings)} embeddings.\n")

    # Loop through the first 2 chunks to see what the data looks like
    for i in range(min(2, len(embeddings))):
        logger.info(f"Chunk ID:   {ids[i]}")
        logger.info(f"Source:     {metadatas[i].get('source')}")
        logger.info(f"Text:       {documents[i]}")

        # An embedding is a massive list of floats. We will just print the first 5.
        vector = embeddings[i]
        logger.info(f"Dimensions: {len(vector)} numbers long")
        logger.info(
            f"Vector:     [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, {vector[3]:.4f}, {vector[4]:.4f}, ...]\n"
        )


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    r"""Return the embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def get_pipeline() -> BasePipeline:
    r"""Define the pipeline."""
    return DocumentIndexingPipeline(
        loader=DocumentListLoader(create_fake_documents()),
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ".", " "]
        ),
        vector_store=Chroma(collection_name="foo", embedding_function=get_embedding_model()),
    )


def main() -> None:
    r"""Define the main function."""
    pipeline = get_pipeline()
    logger.info(pipeline)
    vector_store = pipeline.execute()
    inspect_embeddings(vector_store)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO)
    main()
