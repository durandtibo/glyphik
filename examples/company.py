r"""Provide code to explore a document search pipeline."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import click
from dotenv import load_dotenv
from zenpyre.data_processors import SequenceProcessor
from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.documents import sort_by_metadata
from zenpyre.ingestors import InMemoryIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging, print_document

from glyphik.data.sec import SecForm
from glyphik.data_processors import (
    EdgarCompanyToIdentifierProcessor,
    SecFilingRecordToDocumentProcessor,
)
from glyphik.ingestors import (
    SecCompanyIngestor,
    SecFilingDocumentStoreIngestor,
    SecFilingIngestor,
)

if TYPE_CHECKING:
    from zenpyre.ingestors import BaseIngestor

logger: logging.Logger = logging.getLogger(__name__)


def get_document_store(base_dir: Path) -> DuckDBDocumentStore:
    """Return a persisted DuckDB document store."""
    return DuckDBDocumentStore(base_dir / "document_store" / "documents.duckdb")


def build_ingestor(base_dir: Path, ticker: str) -> BaseIngestor:
    """Build the S&P 1500 filing ingestor rooted at ``base_dir``."""
    return SecFilingDocumentStoreIngestor(
        filing_ingestor=SecFilingIngestor(
            company_ingestor=ProcessorIngestor(
                source=SecCompanyIngestor(ingestor=InMemoryIngestor([ticker])),
                processor=SequenceProcessor(EdgarCompanyToIdentifierProcessor()),
            ),
            output_dir=base_dir / "sec",
            start_date=date(2025, 1, 1),
            end_date=date(2026, 6, 1),
            forms=[SecForm.TEN_K, SecForm.TEN_Q],
        ),
        document_store=get_document_store(base_dir),
        processor=SequenceProcessor(SecFilingRecordToDocumentProcessor()),
    )


def download_data(base_dir: Path, ticker: str) -> None:
    """Download and index SEC filings for the given ticker."""
    ingestor = build_ingestor(base_dir, ticker=ticker)
    logger.info("%s", ingestor)
    ingestor.ingest()


def process_data(base_dir: Path, ticker: str) -> None:
    """Query the document store and print the filings found for the
    given ticker, ordered by filing date."""
    doc_store = get_document_store(base_dir)
    logger.info("%s", doc_store)

    docs = sort_by_metadata(doc_store.filter(ticker=ticker), metadata_key="filing_date")
    logger.info("Found %d documents", len(docs))
    for doc in docs:
        print_document(doc)


@click.command()
@click.option("--ticker", prompt="Enter a ticker", help="The ticker of the company to analyze")
def main(ticker: str) -> None:
    """Run the document indexing pipeline and inspect the vector
    store."""
    base_dir = Path(__file__).parent.parent / "tmp/examples/company"

    download_data(base_dir=base_dir, ticker=ticker)
    process_data(base_dir=base_dir, ticker=ticker)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
