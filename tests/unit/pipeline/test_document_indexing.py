"""Unit tests for DocumentIndexingPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from coola.equality import objects_are_equal
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
from zenpyre.document_loaders import DocumentListLoader

from glyphik.pipeline import BasePipeline, DocumentIndexingPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_docs() -> list[Document]:
    return [
        Document(page_content="Cats sleep up to 16 hours a day."),
        Document(page_content="Cats are obligate carnivores."),
    ]


def _make_loader(documents: list[Document] | None = None) -> DocumentListLoader:
    return DocumentListLoader(documents if documents is not None else _make_docs())


def _make_text_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


def _make_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(FakeEmbeddings(size=128))


def _make_pipeline(
    loader: BaseLoader | None = None,
    text_splitter: TextSplitter | None = None,
    vector_store: VectorStore | None = None,
) -> DocumentIndexingPipeline:
    return DocumentIndexingPipeline(
        loader=loader or _make_loader(),
        text_splitter=text_splitter or _make_text_splitter(),
        vector_store=vector_store or _make_vector_store(),
    )


##################################################
#     Tests for DocumentIndexingPipeline         #
##################################################


# --- Inheritance ---


def test_document_indexing_pipeline_is_base_pipeline() -> None:
    assert isinstance(_make_pipeline(), BasePipeline)


# --- execute ---


def test_document_indexing_pipeline_execute_returns_vector_store() -> None:
    assert isinstance(_make_pipeline().execute(), VectorStore)


def test_document_indexing_pipeline_execute_returns_same_vector_store_instance() -> None:
    vector_store = _make_vector_store()
    pipeline = _make_pipeline(vector_store=vector_store)
    assert pipeline.execute() is vector_store


def test_document_indexing_pipeline_execute_calls_loader_load() -> None:
    loader = MagicMock(spec=BaseLoader)
    loader.load.return_value = _make_docs()
    _make_pipeline(loader=loader).execute()
    loader.load.assert_called_once()


def test_document_indexing_pipeline_execute_calls_add_documents() -> None:
    vector_store = MagicMock(spec=VectorStore)
    _make_pipeline(vector_store=vector_store).execute()
    vector_store.add_documents.assert_called_once()


def test_document_indexing_pipeline_execute_adds_chunks_to_vector_store() -> None:
    docs = _make_docs()
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.return_value = docs
    vector_store = MagicMock(spec=VectorStore)
    _make_pipeline(text_splitter=text_splitter, vector_store=vector_store).execute()
    vector_store.add_documents.assert_called_once_with(docs)


def test_document_indexing_pipeline_execute_with_empty_loader() -> None:
    vector_store = MagicMock(spec=VectorStore)
    pipeline = _make_pipeline(loader=_make_loader([]), vector_store=vector_store)
    result = pipeline.execute()
    assert result is vector_store


# --- _split_documents ---


def test_document_indexing_pipeline_split_documents_calls_text_splitter() -> None:
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.return_value = _make_docs()
    pipeline = _make_pipeline(text_splitter=text_splitter)
    pipeline._split_documents(_make_docs())
    text_splitter.split_documents.assert_called_once_with(_make_docs())


def test_document_indexing_pipeline_split_documents_returns_chunks() -> None:
    chunks = [Document(page_content="chunk")]
    text_splitter = MagicMock(spec=TextSplitter)
    text_splitter.split_documents.return_value = chunks
    pipeline = _make_pipeline(text_splitter=text_splitter)
    assert pipeline._split_documents(_make_docs()) == chunks


# --- _get_repr_kwargs ---


def test_document_indexing_pipeline_get_repr_kwargs() -> None:
    loader = _make_loader()
    text_splitter = _make_text_splitter()
    vector_store = _make_vector_store()
    pipeline = DocumentIndexingPipeline(
        loader=loader,
        text_splitter=text_splitter,
        vector_store=vector_store,
    )
    kwargs = pipeline._get_repr_kwargs()
    assert objects_are_equal(
        kwargs, {"loader": loader, "text_splitter": text_splitter, "vector_store": vector_store}
    )


# --- __repr__ and __str__ ---


def test_document_indexing_pipeline_repr_starts_with_class_name() -> None:
    assert repr(_make_pipeline()).startswith("DocumentIndexingPipeline(")


def test_document_indexing_pipeline_str_starts_with_class_name() -> None:
    assert str(_make_pipeline()).startswith("DocumentIndexingPipeline(")
