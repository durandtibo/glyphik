"""Unit tests for FilingDocumentIngestor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from zenpyre.document_stores import BaseDocumentStore
from zenpyre.ingestors import InMemoryIngestor

from glyphik.ingestors import FilingDocumentIngestor

MODULE = "glyphik.ingestors.filing_document"


def _make_record(id_: str) -> MagicMock:
    record = MagicMock()
    record.id = id_
    return record


def _make_store(present: list[str] | None = None) -> MagicMock:
    store = MagicMock(spec=BaseDocumentStore)
    present = present or []
    store.check_ids.side_effect = lambda ids: (
        [i for i in ids if i in present],
        [i for i in ids if i not in present],
    )
    return store


def _make_ingestor(
    records: list[MagicMock],
    store: MagicMock,
    processor: MagicMock | None = None,
    batch_size: int = 32,
) -> FilingDocumentIngestor:
    if processor is None:
        processor = MagicMock()
        processor.process.return_value = []
    return FilingDocumentIngestor(
        filing_ingestor=InMemoryIngestor(data=records, copy=False),
        processor=processor,
        store=store,
        batch_size=batch_size,
    )


##############################################
#   Tests for FilingDocumentIngestor         #
##############################################


# --- Constructor ---


def test_filing_document_ingestor_stores_filing_ingestor() -> None:
    filing_ingestor = InMemoryIngestor(data=[], copy=False)
    ingestor = FilingDocumentIngestor(
        filing_ingestor=filing_ingestor,
        processor=MagicMock(),
        store=_make_store(),
    )
    assert ingestor._filing_ingestor is filing_ingestor


def test_filing_document_ingestor_stores_processor() -> None:
    processor = MagicMock()
    processor.process.return_value = []
    ingestor = FilingDocumentIngestor(
        filing_ingestor=InMemoryIngestor(data=[], copy=False),
        processor=processor,
        store=_make_store(),
    )
    assert ingestor._processor is processor


def test_filing_document_ingestor_stores_store() -> None:
    store = _make_store()
    ingestor = _make_ingestor([], store)
    assert ingestor._store is store


def test_filing_document_ingestor_batch_size_default() -> None:
    ingestor = _make_ingestor([], _make_store())
    assert ingestor._batch_size == 32


def test_filing_document_ingestor_stores_batch_size() -> None:
    ingestor = _make_ingestor([], _make_store(), batch_size=16)
    assert ingestor._batch_size == 16


def test_filing_document_ingestor_repr_contains_class_name() -> None:
    ingestor = _make_ingestor([], _make_store())
    assert "FilingDocumentIngestor" in repr(ingestor)


def test_filing_document_ingestor_str_contains_class_name() -> None:
    ingestor = _make_ingestor([], _make_store())
    assert "FilingDocumentIngestor" in str(ingestor)


# --- ingest: return value ---


def test_filing_document_ingestor_ingest_returns_store() -> None:
    store = _make_store()
    result = _make_ingestor([], store).ingest()
    assert result is store


# --- ingest: deduplication ---


def test_filing_document_ingestor_ingest_calls_check_ids() -> None:
    records = [_make_record("a"), _make_record("b")]
    store = _make_store()
    _make_ingestor(records, store).ingest()
    store.check_ids.assert_called_once_with(["a", "b"])


def test_filing_document_ingestor_ingest_skips_present_records() -> None:
    records = [_make_record("a"), _make_record("b")]
    store = _make_store(present=["a", "b"])
    processor = MagicMock()
    processor.process.return_value = []
    _make_ingestor(records, store, processor=processor).ingest()
    processor.process.assert_not_called()


def test_filing_document_ingestor_ingest_only_processes_missing_records() -> None:
    record_a = _make_record("a")
    record_b = _make_record("b")
    store = _make_store(present=["a"])
    processor = MagicMock()
    processor.process.return_value = []

    with patch(f"{MODULE}.batchify", return_value=[[record_b]]):
        _make_ingestor([record_a, record_b], store, processor=processor).ingest()

    processor.process.assert_called_once_with([record_b])


# --- ingest: batching ---


def test_filing_document_ingestor_ingest_adds_documents_to_store() -> None:
    record = _make_record("a")
    doc = Document(id="a", page_content="text", metadata={})
    store = _make_store()
    processor = MagicMock()
    processor.process.return_value = [doc]

    with patch(f"{MODULE}.batchify", return_value=[[record]]):
        _make_ingestor([record], store, processor=processor).ingest()

    store.add_documents.assert_called_once_with([doc])


def test_filing_document_ingestor_ingest_empty_filings_returns_store() -> None:
    store = _make_store()
    result = _make_ingestor([], store).ingest()
    assert result is store


def test_filing_document_ingestor_ingest_all_present_does_not_call_add_documents() -> None:
    records = [_make_record("a"), _make_record("b")]
    store = _make_store(present=["a", "b"])
    _make_ingestor(records, store).ingest()
    store.add_documents.assert_not_called()
