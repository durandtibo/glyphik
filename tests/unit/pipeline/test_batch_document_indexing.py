"""Unit tests for BatchDocumentIndexingPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
from zenpyre.document_loaders import DocumentListLoader

from glyphik.pipeline import BasePipeline, BatchDocumentIndexingPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_docs(n: int = 6) -> list[Document]:
    return [Document(page_content=f"Cats fact number {i}.") for i in range(n)]


def _make_loader(n: int = 6) -> DocumentListLoader:
    return DocumentListLoader(_make_docs(n))


def _make_text_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def _make_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(FakeEmbeddings(size=128))


def _make_pipeline(
    loader: BaseLoader | None = None,
    text_splitter: TextSplitter | None = None,
    vector_store: VectorStore | None = None,
    batch_size: int = 2,
) -> BatchDocumentIndexingPipeline:
    return BatchDocumentIndexingPipeline(
        loader=loader or _make_loader(),
        text_splitter=text_splitter or _make_text_splitter(),
        vector_store=vector_store or _make_vector_store(),
        batch_size=batch_size,
    )


###################################################
#     Tests for BatchDocumentIndexingPipeline     #
###################################################


# --- Inheritance ---


def test_batch_document_indexing_pipeline_is_base_pipeline() -> None:
    assert isinstance(_make_pipeline(), BasePipeline)


# --- execute ---


def test_batch_document_indexing_pipeline_execute_returns_vector_store() -> None:
    assert isinstance(_make_pipeline().execute(), VectorStore)


def test_batch_document_indexing_pipeline_execute_returns_same_vector_store_instance() -> None:
    vector_store = _make_vector_store()
    pipeline = _make_pipeline(vector_store=vector_store)
    assert pipeline.execute() is vector_store


def test_batch_document_indexing_pipeline_execute_calls_lazy_load() -> None:
    loader = MagicMock(spec=BaseLoader)
    loader.lazy_load.return_value = iter(_make_docs())
    _make_pipeline(loader=loader).execute()
    loader.lazy_load.assert_called_once()


def test_batch_document_indexing_pipeline_execute_calls_add_documents() -> None:
    vector_store = MagicMock(spec=VectorStore)
    _make_pipeline(vector_store=vector_store).execute()
    vector_store.add_documents.assert_called()


def test_batch_document_indexing_pipeline_execute_with_empty_loader() -> None:
    vector_store = MagicMock(spec=VectorStore)
    pipeline = _make_pipeline(loader=_make_loader(0), vector_store=vector_store)
    result = pipeline.execute()
    assert result is vector_store
    vector_store.add_documents.assert_not_called()


def test_batch_document_indexing_pipeline_execute_processes_all_documents() -> None:
    n_docs = 7
    batch_size = 3
    vector_store = MagicMock(spec=VectorStore)
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.side_effect = lambda docs: docs
    pipeline = _make_pipeline(
        loader=_make_loader(n_docs),
        text_splitter=text_splitter,
        vector_store=vector_store,
        batch_size=batch_size,
    )
    pipeline.execute()
    # 7 docs with batch_size=3: batches of 3, 3, 1 → 3 add_documents calls
    assert vector_store.add_documents.call_count == 3


def test_batch_document_indexing_pipeline_execute_docs_exactly_divisible_by_batch() -> None:
    n_docs = 6
    batch_size = 3
    vector_store = MagicMock(spec=VectorStore)
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.side_effect = lambda docs: docs
    pipeline = _make_pipeline(
        loader=_make_loader(n_docs),
        text_splitter=text_splitter,
        vector_store=vector_store,
        batch_size=batch_size,
    )
    pipeline.execute()
    # 6 docs with batch_size=3: two full batches → 2 add_documents calls
    assert vector_store.add_documents.call_count == 2


# --- _split_and_index ---


def test_batch_document_indexing_pipeline_split_and_index_calls_text_splitter() -> None:
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.return_value = _make_docs(2)
    pipeline = _make_pipeline(text_splitter=text_splitter)
    pipeline._split_and_index(_make_docs(2))
    text_splitter.split_documents.assert_called_once_with(_make_docs(2))


def test_batch_document_indexing_pipeline_split_and_index_calls_add_documents() -> None:
    chunks = _make_docs(2)
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.return_value = chunks
    vector_store = MagicMock(spec=VectorStore)
    pipeline = _make_pipeline(text_splitter=text_splitter, vector_store=vector_store)
    pipeline._split_and_index(_make_docs(2))
    vector_store.add_documents.assert_called_once_with(chunks)


def test_batch_document_indexing_pipeline_split_and_index_returns_chunks() -> None:
    chunks = _make_docs(3)
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.return_value = chunks
    pipeline = _make_pipeline(text_splitter=text_splitter)
    assert pipeline._split_and_index(_make_docs(2)) == chunks


# --- _get_repr_kwargs ---


def test_batch_document_indexing_pipeline_get_repr_kwargs_keys() -> None:
    assert set(_make_pipeline()._get_repr_kwargs().keys()) == {
        "loader",
        "text_splitter",
        "vector_store",
        "batch_size",
    }


def test_batch_document_indexing_pipeline_get_repr_kwargs_batch_size() -> None:
    assert _make_pipeline(batch_size=64)._get_repr_kwargs()["batch_size"] == 64


def test_batch_document_indexing_pipeline_get_repr_kwargs_values() -> None:
    loader = _make_loader()
    text_splitter = _make_text_splitter()
    vector_store = _make_vector_store()
    pipeline = BatchDocumentIndexingPipeline(
        loader=loader,
        text_splitter=text_splitter,
        vector_store=vector_store,
        batch_size=32,
    )
    kwargs = pipeline._get_repr_kwargs()
    assert kwargs["loader"] is loader
    assert kwargs["text_splitter"] is text_splitter
    assert kwargs["vector_store"] is vector_store
    assert kwargs["batch_size"] == 32


# --- __repr__ and __str__ ---


def test_batch_document_indexing_pipeline_repr_starts_with_class_name() -> None:
    assert repr(_make_pipeline()).startswith("BatchDocumentIndexingPipeline(")


def test_batch_document_indexing_pipeline_str_starts_with_class_name() -> None:
    assert str(_make_pipeline()).startswith("BatchDocumentIndexingPipeline(")


def test_batch_document_indexing_pipeline_repr_equals_str() -> None:
    pipeline = _make_pipeline()
    assert repr(pipeline) == str(pipeline)
