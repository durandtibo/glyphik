r"""Contain code to explore a document search pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from coola.hashing import hash_string
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from glyphik.utils.logging import log_pretty

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

# Initialize the logger
logger: logging.Logger = logging.getLogger(__name__)

# Suppress HuggingFace/Chroma warnings for a cleaner console output
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_fake_documents() -> list[Document]:
    """Step 1: Generate fake input documents with metadata."""
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


def split_documents(documents: list[Document]) -> list[Document]:
    """Step 2: Split large documents into smaller chunks."""
    logger.info(f"-> Splitting {len(documents)} documents into chunks...")

    # In this example, the texts are short, so we use small chunk sizes.
    # For a real PDF, you might use chunk_size=500 and chunk_overlap=50.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"   Created {len(chunks)} chunks.")
    log_pretty(chunks)
    return chunks


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    r"""Return the embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(chunks: list[Document], base_dir: Path | None = None) -> Chroma:
    """Step 3 & 4: Generate embeddings and load them into a Vector
    Store."""
    logger.info("-> Initializing embedding model and vector database...")

    # We use a fast, free, local model from HuggingFace
    embeddings = get_embedding_model()

    # Initialize Chroma DB. Using persist_directory saves the DB to your hard drive.
    # If you leave it blank, it stays in memory temporarily.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="my_local_search_engine",
        persist_directory=base_dir.joinpath("vector_store").as_posix() if base_dir else None,
    )


def get_vector_store(base_dir: Path) -> Chroma:
    r"""Return the vector store."""
    # Initialize your embedding model
    embeddings = get_embedding_model()

    # 1. Instantiate the LangChain Chroma wrapper.
    # This automatically connects to the folder on your hard drive.
    return Chroma(
        collection_name="my_local_search_engine",
        persist_directory=base_dir.joinpath("vector_store").as_posix(),
        embedding_function=embeddings,
    )


def get_or_build_vector_store(base_dir: Path) -> Chroma:
    r"""Return the vector store."""
    model_name = "Qwen/Qwen3-Embedding-8B"
    # model_name = "bge-base"
    collection_name = hash_string(model_name)

    # Initialize your embedding model
    embeddings = get_embedding_model(model_name)
    # embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")

    # 1. Instantiate the LangChain Chroma wrapper.
    # This automatically connects to the folder on your hard drive.
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=base_dir.joinpath("vector_store2").as_posix(),
        embedding_function=embeddings,
    )

    # 2. Check the database for existing data using LangChain's .get() method
    # This returns a dictionary containing a list of document 'ids'
    existing_ids = vector_store.get()["ids"]

    if len(existing_ids) > 0:
        logger.info(
            f"Database loaded! Found {len(existing_ids)} existing documents. Skipping computation."
        )
    else:
        logger.info("Database is empty. Computing embeddings for the first time...")

        # Fetch and process your documents
        docs = create_fake_documents()
        chunks = split_documents(docs)

        # Add the chunks to the vector store.
        # Because 'persist_directory' was set above, this automatically saves to your hard drive!
        vector_store.add_documents(chunks)

        logger.info("Embeddings computed and saved to disk successfully!")

    return vector_store


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


def search(query: str, vector_store: Chroma, search_kwargs: dict | None = None) -> None:
    """Step 5: The Search Interface with dynamic search arguments."""
    logger.info("\n==================================================")
    logger.info(f"SEARCH QUERY: '{query}'")

    # Provide a default configuration if nothing is passed
    if search_kwargs is None:
        search_kwargs = {"k": 2}

    logger.info(f"SEARCH KWARGS: {search_kwargs}")
    logger.info("==================================================")

    # Create the retriever, passing your custom dictionary directly
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # Execute the search
    results = retriever.invoke(query)

    # Display the results
    if not results:
        logger.info("No relevant documents found.")
        return

    for i, doc in enumerate(results, 1):
        logger.info(f"\nRESULT {i}:")
        logger.info(f"Text:    {doc.page_content}")
        logger.info(f"Source:  {doc.metadata.get('source')}")
        logger.info(f"Category:{doc.metadata.get('category')}")
    logger.info("==================================================\n")


def build_and_search(base_dir: Path) -> None:  # noqa: D103
    # 1. Ingest
    docs = create_fake_documents()

    # 2. Process
    chunks = split_documents(docs)

    # 3 & 4. Store
    vector_store = create_vector_store(chunks, base_dir)
    inspect_embeddings(vector_store)

    # 5. Retrieve (Testing different semantic queries)

    # Test A: Asking a question (Semantic match)
    search("How do I make bread?", vector_store)

    # Test B: Searching for a concept (Semantic match for "astronomy/space")
    search("Tell me about exploring the universe.", vector_store)

    # Test C: Exact keyword match
    search("Python 3.12", vector_store)

    # 1. Search with NO filter (searches everything)
    search("Tell me about stars", vector_store)

    # 2. Search WITH a filter (searches ONLY within 'Cooking')
    search_kwargs = {
        "k": 2,
        # "filter": {
        #     "$and": [
        #         {"category": {"$in": ["Cooking", "Science"]}},
        #         {"year": {"$gt": 2020}},
        #     ]
        # },
        "filter": {"category": {"$in": ["Cooking"]}},
    }
    search("Tell me about stars", vector_store, search_kwargs)


def get_and_search(base_dir: Path) -> None:  # noqa: D103
    # 3 & 4. Store
    vector_store = get_or_build_vector_store(base_dir)
    # vector_store = get_vector_store(base_dir)
    inspect_embeddings(vector_store)

    # 5. Retrieve (Testing different semantic queries)

    # Test A: Asking a question (Semantic match)
    search("How do I make bread?", vector_store)

    # Test B: Searching for a concept (Semantic match for "astronomy/space")
    search("Tell me about exploring the universe.", vector_store)

    # Test C: Exact keyword match
    search("Python 3.12", vector_store)


def main() -> None:  # noqa: D103
    base_dir = Path(__file__).parent.parent / "tmp/search_demo/"

    # build_and_search(base_dir)
    get_and_search(base_dir)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
