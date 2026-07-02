"""Unit tests for SecFilingRecordToDocumentProcessor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from glyphik.data.sec import SecFilingRecord
from glyphik.data_processors import SecFilingRecordToDocumentProcessor

MODULE = "glyphik.data_processors.sec_record_to_document"


@pytest.fixture
def record() -> SecFilingRecord:
    record = MagicMock(
        spec=SecFilingRecord,
        id="test-record-123",
        metadata={"filepath": "/data/filing.txt", "cik": "0000320193"},
    )
    record.load_filing = MagicMock(return_value=MagicMock(name="mock_filing_object"))
    return record


########################################################
#   Tests for SecFilingRecordToDocumentProcessor       #
########################################################


# --- Constructor ---


def test_sec_filing_record_to_document_processor_content_format_default_text() -> None:
    assert SecFilingRecordToDocumentProcessor()._content_format == "text"


def test_sec_filing_record_to_document_processor_stores_content_format() -> None:
    p = SecFilingRecordToDocumentProcessor(content_format="markdown")
    assert p._content_format == "markdown"


def test_sec_filing_record_to_document_processor_repr_contains_class_name() -> None:
    assert "SecFilingRecordToDocumentProcessor" in repr(
        SecFilingRecordToDocumentProcessor(content_format="html")
    )


def test_sec_filing_record_to_document_processor_str_contains_class_name() -> None:
    assert "SecFilingRecordToDocumentProcessor" in str(
        SecFilingRecordToDocumentProcessor(content_format="html")
    )


# --- process ---


def test_sec_filing_record_to_document_processor_process_returns_document(
    record: SecFilingRecord,
) -> None:
    with patch(f"{MODULE}.extract_filing_content") as mock_extract:
        mock_extract.return_value = "Sample extracted content"
        result = SecFilingRecordToDocumentProcessor().process(record)

    assert isinstance(result, Document)
    assert result.id == "test-record-123"
    assert result.page_content == "Sample extracted content"


def test_sec_filing_record_to_document_processor_process_calls_load_filing(
    record: SecFilingRecord,
) -> None:
    with patch(f"{MODULE}.extract_filing_content") as mock_extract:
        mock_extract.return_value = "Sample extracted content"
        SecFilingRecordToDocumentProcessor().process(record)

    record.load_filing.assert_called_once()


def test_sec_filing_record_to_document_processor_process_calls_extract_filing_content(
    record: SecFilingRecord,
) -> None:
    with patch(f"{MODULE}.extract_filing_content") as mock_extract:
        mock_extract.return_value = "Sample extracted content"
        SecFilingRecordToDocumentProcessor(content_format="xml").process(record)

    mock_extract.assert_called_once_with(record.load_filing.return_value, content_format="xml")


def test_sec_filing_record_to_document_processor_process_raises_value_error_on_none(
    record: SecFilingRecord,
) -> None:
    with patch(f"{MODULE}.extract_filing_content") as mock_extract:
        mock_extract.return_value = None
        with pytest.raises(ValueError, match="Filing content could not be extracted"):
            SecFilingRecordToDocumentProcessor().process(record)


def test_sec_filing_record_to_document_processor_process_copies_metadata(
    record: SecFilingRecord,
) -> None:
    with patch(f"{MODULE}.extract_filing_content") as mock_extract:
        mock_extract.return_value = "Sample extracted content"
        result = SecFilingRecordToDocumentProcessor().process(record)

    # Ensure it matches the fixture's metadata
    assert result.metadata == {"filepath": "/data/filing.txt", "cik": "0000320193"}
    # Mutate the resulting Document's metadata
    result.metadata["new_key"] = "mutated"
    # Ensure the original SecFilingRecord's metadata is unharmed (not passed by reference)
    assert "new_key" not in record.metadata
