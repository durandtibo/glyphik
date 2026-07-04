from __future__ import annotations

from unittest.mock import MagicMock

from coola.testing.fixtures import numpy_available
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from zenpyre.document_stores import BaseDocumentStore, InMemoryDocumentStore
from zenpyre.ingestors import InMemoryIngestor
from zenpyre.ingestors.base import BaseIngestor
from zenpyre.testing.fixtures import langchain_text_splitters_available
from zenpyre.utils.imports import is_langchain_text_splitters_available

from glyphik.ingestors import DocumentStoreIndexingIngestor

if is_langchain_text_splitters_available():
    from langchain_text_splitters import CharacterTextSplitter, TextSplitter
else:
    CharacterTextSplitter, TextSplitter = None, None

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_docs(n: int = 6) -> list[Document]:
    return [Document(id=str(i), page_content=f"Cats fact number {i}.") for i in range(n)]


def _make_document_store(n: int = 6) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    store.add_documents(_make_docs(n))
    return store


def _make_document_store_ingestor(n: int = 6) -> InMemoryIngestor:
    return InMemoryIngestor(_make_document_store(n))


def _make_text_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def _make_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(FakeEmbeddings(size=128))


def _make_ingestor(
    document_store_ingestor: BaseIngestor[BaseDocumentStore] | None = None,
    text_splitter: TextSplitter | None = None,
    vector_store: VectorStore | None = None,
    batch_size: int = 2,
) -> DocumentStoreIndexingIngestor:
    return DocumentStoreIndexingIngestor(
        document_store_ingestor=document_store_ingestor or _make_document_store_ingestor(),
        text_splitter=text_splitter or _make_text_splitter(),
        vector_store=vector_store or _make_vector_store(),
        batch_size=batch_size,
    )


####################################################
#     Tests for DocumentStoreIndexingIngestor      #
####################################################


# --- Inheritance ---


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_is_base_ingestor() -> None:
    assert isinstance(_make_ingestor(), BaseIngestor)


# --- ingest ---


@numpy_available
@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_returns_vector_store() -> None:
    assert isinstance(_make_ingestor().ingest(), VectorStore)


@numpy_available
@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_returns_same_vector_store_instance() -> None:
    vector_store = _make_vector_store()
    ingestor = _make_ingestor(vector_store=vector_store)
    assert ingestor.ingest() is vector_store


@numpy_available
@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_calls_document_store_ingestor() -> None:
    document_store_ingestor = MagicMock(spec=BaseIngestor)
    document_store_ingestor.ingest.return_value = _make_document_store()
    _make_ingestor(document_store_ingestor=document_store_ingestor).ingest()
    document_store_ingestor.ingest.assert_called_once()


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_calls_add_documents() -> None:
    vector_store = MagicMock(spec=VectorStore)
    _make_ingestor(vector_store=vector_store).ingest()
    vector_store.add_documents.assert_called()


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_with_empty_document_store() -> None:
    vector_store = MagicMock(spec=VectorStore)
    ingestor = _make_ingestor(
        document_store_ingestor=_make_document_store_ingestor(0), vector_store=vector_store
    )
    result = ingestor.ingest()
    assert result is vector_store
    vector_store.add_documents.assert_not_called()


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_processes_all_documents() -> None:
    n_docs = 7
    batch_size = 3
    vector_store = MagicMock(spec=VectorStore)
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.side_effect = lambda docs: docs
    ingestor = _make_ingestor(
        document_store_ingestor=_make_document_store_ingestor(n_docs),
        text_splitter=text_splitter,
        vector_store=vector_store,
        batch_size=batch_size,
    )
    ingestor.ingest()
    # 7 docs with batch_size=3: batches of 3, 3, 1 → 3 add_documents calls
    assert vector_store.add_documents.call_count == 3


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_ingest_docs_exactly_divisible_by_batch() -> None:
    n_docs = 6
    batch_size = 3
    vector_store = MagicMock(spec=VectorStore)
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.side_effect = lambda docs: docs
    ingestor = _make_ingestor(
        document_store_ingestor=_make_document_store_ingestor(n_docs),
        text_splitter=text_splitter,
        vector_store=vector_store,
        batch_size=batch_size,
    )
    ingestor.ingest()
    # 6 docs with batch_size=3: two full batches → 2 add_documents calls
    assert vector_store.add_documents.call_count == 2


# --- _get_repr_kwargs ---


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_get_repr_kwargs_keys() -> None:
    assert set(_make_ingestor()._get_repr_kwargs().keys()) == {
        "document_store_ingestor",
        "text_splitter",
        "vector_store",
        "batch_size",
    }


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_get_repr_kwargs_batch_size() -> None:
    assert _make_ingestor(batch_size=64)._get_repr_kwargs()["batch_size"] == 64


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_get_repr_kwargs_values() -> None:
    document_store_ingestor = _make_document_store_ingestor()
    text_splitter = _make_text_splitter()
    vector_store = _make_vector_store()
    ingestor = DocumentStoreIndexingIngestor(
        document_store_ingestor=document_store_ingestor,
        text_splitter=text_splitter,
        vector_store=vector_store,
        batch_size=32,
    )
    kwargs = ingestor._get_repr_kwargs()
    assert kwargs["document_store_ingestor"] is document_store_ingestor
    assert kwargs["text_splitter"] is text_splitter
    assert kwargs["vector_store"] is vector_store
    assert kwargs["batch_size"] == 32


# --- __repr__ and __str__ ---


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_repr_starts_with_class_name() -> None:
    assert repr(_make_ingestor()).startswith("DocumentStoreIndexingIngestor(")


@langchain_text_splitters_available
def test_document_store_indexing_ingestor_str_starts_with_class_name() -> None:
    assert str(_make_ingestor()).startswith("DocumentStoreIndexingIngestor(")
