r"""Provide a concrete factory that creates a DuckDB-backed
BaseDocumentStore for SEC filings."""

from __future__ import annotations

__all__ = ["SecFilingDocumentStoreFactory"]

from typing import TYPE_CHECKING, Any

from coola.utils.path import sanitize_path
from zenpyre.document_stores.factory import DuckDBDocumentStoreFactory

if TYPE_CHECKING:
    from pathlib import Path


class SecFilingDocumentStoreFactory(DuckDBDocumentStoreFactory):
    """A concrete BaseDocumentStore factory that builds a
        :class:`~zenpyre.document_stores.DuckDBDocumentStore` for SEC
        filings, backed by a DuckDB file under a given base directory.

        Use this when you want a factory that lazily constructs a fresh
        :class:`~zenpyre.document_stores.DuckDBDocumentStore` rooted at
        ``base_dir / "document_store" / "sec_filings.duckdb"`` each time
        :meth:`make_document_store` is called, rather than wrapping an
        already-instantiated store.

    Args:
            base_dir: The base directory under which the DuckDB SEC
                filing document store file is created (at
                ``base_dir / "document_store" / "sec_filings.duckdb"``).
                Accepts a :class:`~pathlib.Path` or a :class:`str`, which
                is sanitized via
                :func:`~coola.utils.path.sanitize_path`.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~zenpyre.document_stores.DuckDBDocumentStore`.

    Example:
    ```pycon
    >>> from pathlib import Path
    >>> from glyphik.document_stores.factory import SecFilingDocumentStoreFactory
    >>> factory = SecFilingDocumentStoreFactory(Path("/tmp/my_app"))
    >>> document_store = factory.make_document_store()  # doctest: +SKIP

    ```
    """

    def __init__(self, base_dir: Path | str, **kwargs: Any) -> None:
        path = sanitize_path(base_dir) / "document_store" / "sec_filing.duckdb"
        super().__init__(path, **kwargs)
