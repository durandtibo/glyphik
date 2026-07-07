"""Explore a document search pipeline."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from zenpyre.data_processors import SequenceProcessor
from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.ingestors import FirstNIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging, print_pretty

from glyphik.data.sec import SecForm
from glyphik.data_processors import (
    SecFilingRecordToDocumentProcessor,
    Sp1500CompanyToIdentifierProcessor,
)
from glyphik.ingestors import (
    SecFilingDocumentStoreIngestor,
    SecFilingIngestor,
    Sp1500CompanyIngestor,
)

if TYPE_CHECKING:
    from zenpyre.ingestors import BaseIngestor

logger: logging.Logger = logging.getLogger(__name__)


def get_document_store(base_dir: Path, **kwargs: Any) -> DuckDBDocumentStore:
    """Return a persisted DuckDB document store."""
    return DuckDBDocumentStore(base_dir / "document_store" / "documents.duckdb", **kwargs)


def build_ingestor(base_dir: Path) -> BaseIngestor:
    """Build the S&P 1500 filing ingestor rooted at ``base_dir``."""
    return SecFilingDocumentStoreIngestor(
        filing_ingestor=SecFilingIngestor(
            company_ingestor=ProcessorIngestor(
                source=FirstNIngestor(
                    Sp1500CompanyIngestor(path=base_dir / "sp1500" / "companies.json"),
                    n=5,
                ),
                processor=SequenceProcessor(Sp1500CompanyToIdentifierProcessor()),
            ),
            output_dir=base_dir / "sec",
            start_date=date(2023, 1, 1),
            end_date=date(2026, 6, 1),
            forms=[SecForm.TEN_K, SecForm.TEN_Q],
        ),
        document_store=get_document_store(base_dir),
        processor=SequenceProcessor(SecFilingRecordToDocumentProcessor()),
    )


def main() -> None:
    """Build the ingestor, run it, and print a sample of filed
    documents.

    Ingests S&P 1500 filings into a DuckDB-backed document store (or
    reuses the cached store if one already exists at the target path),
    then prints the metadata for one ticker/form combination as a sanity
    check.
    """
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628.0"

    ingestor = build_ingestor(base_dir)
    logger.info("%s", ingestor)
    store = ingestor.ingest()

    logger.info("%s", store)
    store.show_columns_info()

    docs = store.filter(ticker="MMM", form=SecForm.TEN_K)
    print_pretty([doc.metadata for doc in docs])


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
