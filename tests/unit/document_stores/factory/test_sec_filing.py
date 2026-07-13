from __future__ import annotations

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal
from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.document_stores.base import BaseDocumentStore
from zenpyre.document_stores.factory import DuckDBDocumentStoreFactory
from zenpyre.testing.fixtures import duckdb_available

from glyphik.document_stores.factory import SecFilingDocumentStoreFactory

if TYPE_CHECKING:
    from pathlib import Path

##################################################
#     Tests for SecFilingDocumentStoreFactory    #
##################################################


# --- Inheritance ---


@duckdb_available
def test_sec_filing_document_store_factory_is_duckdb_document_store_factory(
    tmp_path: Path,
) -> None:
    assert isinstance(SecFilingDocumentStoreFactory(tmp_path), DuckDBDocumentStoreFactory)


# --- __init__ path construction ---


@duckdb_available
def test_sec_filing_document_store_factory_builds_expected_path_from_path(
    tmp_path: Path,
) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    expected_path = tmp_path / "document_store" / "sec_filing.duckdb"
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": expected_path})


@duckdb_available
def test_sec_filing_document_store_factory_builds_expected_path_from_str(
    tmp_path: Path,
) -> None:
    factory = SecFilingDocumentStoreFactory(str(tmp_path))
    expected_path = tmp_path / "document_store" / "sec_filing.duckdb"
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": expected_path})


# --- make_document_store ---


@duckdb_available
def test_sec_filing_document_store_factory_make_document_store_returns_base_document_store(
    tmp_path: Path,
) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


@duckdb_available
def test_sec_filing_document_store_factory_make_document_store_returns_duckdb_document_store(
    tmp_path: Path,
) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    assert isinstance(factory.make_document_store(), DuckDBDocumentStore)


@duckdb_available
def test_sec_filing_document_store_factory_make_document_store_returns_new_instance_each_call(
    tmp_path: Path,
) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    assert factory.make_document_store() is not factory.make_document_store()


# --- kwargs forwarding ---


@duckdb_available
def test_sec_filing_document_store_factory_forwards_kwargs_to_repr(tmp_path: Path) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path, read_only=True)
    expected_path = tmp_path / "document_store" / "sec_filing.duckdb"
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": expected_path, "read_only": True})


# --- __repr__ and __str__ ---


@duckdb_available
def test_sec_filing_document_store_factory_repr_starts_with_class_name(tmp_path: Path) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    assert repr(factory).startswith("SecFilingDocumentStoreFactory(")


@duckdb_available
def test_sec_filing_document_store_factory_str_starts_with_class_name(tmp_path: Path) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    assert str(factory).startswith("SecFilingDocumentStoreFactory(")


@duckdb_available
def test_sec_filing_document_store_factory_repr_contains_path(tmp_path: Path) -> None:
    factory = SecFilingDocumentStoreFactory(tmp_path)
    assert "path" in repr(factory)
