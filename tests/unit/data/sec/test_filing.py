"""Unit tests for SecFilingRecord."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from glyphik.data.sec import SecFilingRecord

MODULE = "glyphik.data.sec.filing"


#######################################
#     Tests for SecFilingRecord       #
#######################################


# --- Construction ---


def test_sec_filing_record_is_frozen() -> None:
    record = SecFilingRecord(id="abc", metadata={"key": "value"})
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'id'"):
        record.id = "other"


def test_sec_filing_record_default_metadata() -> None:
    record = SecFilingRecord(id="abc")
    assert record.metadata == {}


def test_sec_filing_record_stores_id() -> None:
    record = SecFilingRecord(id="abc", metadata={"key": "value"})
    assert record.id == "abc"


def test_sec_filing_record_stores_metadata() -> None:
    metadata = {"filepath": "tmp/test.pkl", "cik": 123}
    record = SecFilingRecord(id="abc", metadata=metadata)
    assert record.metadata == metadata


# --- from_metadata ---


def test_sec_filing_record_from_metadata_returns_sec_filing_record() -> None:
    assert isinstance(SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"}), SecFilingRecord)


def test_sec_filing_record_from_metadata_sets_metadata() -> None:
    metadata = {"filepath": "tmp/test.pkl", "cik": 123}
    record = SecFilingRecord.from_metadata(metadata)
    assert record.metadata == metadata


def test_sec_filing_record_from_metadata_sets_id() -> None:
    record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    assert record.id is not None


def test_sec_filing_record_from_metadata_id_is_str() -> None:
    record = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl"})
    assert isinstance(record.id, str)


def test_sec_filing_record_from_metadata_id_is_stable() -> None:
    metadata = {"filepath": "tmp/test.pkl", "cik": 123}
    assert SecFilingRecord.from_metadata(metadata).id == SecFilingRecord.from_metadata(metadata).id


def test_sec_filing_record_from_metadata_different_metadata_different_id() -> None:
    record_a = SecFilingRecord.from_metadata({"filepath": "tmp/a.pkl"})
    record_b = SecFilingRecord.from_metadata({"filepath": "tmp/b.pkl"})
    assert record_a.id != record_b.id


def test_sec_filing_record_from_metadata_key_order_independent() -> None:
    record_a = SecFilingRecord.from_metadata({"filepath": "tmp/test.pkl", "cik": 123})
    record_b = SecFilingRecord.from_metadata({"cik": 123, "filepath": "tmp/test.pkl"})
    assert record_a.id == record_b.id


# --- load_filing ---


def test_sec_filing_record_load_filing_missing_filepath_raises() -> None:
    record = SecFilingRecord(id="abc", metadata={})
    with pytest.raises(ValueError, match="filepath"):
        record.load_filing()


def test_sec_filing_record_load_filing_calls_filing_load() -> None:
    record = SecFilingRecord(id="abc", metadata={"filepath": "tmp/test.pkl"})
    with patch(f"{MODULE}.Filing.load") as mock_load:
        record.load_filing()
    mock_load.assert_called_once_with("tmp/test.pkl")


def test_sec_filing_record_load_filing_returns_filing() -> None:
    mock_filing = MagicMock()
    record = SecFilingRecord(id="abc", metadata={"filepath": "tmp/test.pkl"})
    with patch(f"{MODULE}.Filing.load", return_value=mock_filing):
        result = record.load_filing()
    assert result is mock_filing
