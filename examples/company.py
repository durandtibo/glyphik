r"""Provide code to explore a document search pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import click
from coola.display import str_pydantic_model
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from zenpyre.data_processors import SequenceProcessor
from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.ingestors import InMemoryIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging, print_markdown, print_pretty

from glyphik.agents import RecentDocumentsAgent
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
from glyphik.pipeline import TickerDocumentAgentPipeline
from glyphik.prompts.summarization import GENERIC_SYSTEM_PROMPT

if TYPE_CHECKING:
    from zenpyre.ingestors import BaseIngestor

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ollama:gemma4:e2b-mlx"
DEFAULT_BASE_DIR = Path(__file__).parent.parent / "tmp/examples/company"
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 1)
DEFAULT_MAX_DOCUMENTS = 4


@dataclass
class Config:
    """Hold the configuration for the document search pipeline example.

    Args:
        ticker: The ticker symbol of the company to analyze.
        base_dir: The root directory used to store downloaded filings
            and the document store.
        start_date: The earliest filing date (inclusive) to ingest.
        end_date: The latest filing date (inclusive) to ingest.
        max_documents: The maximum number of most recent filings to
            pass to the summarization agent.
    """

    ticker: str
    base_dir: Path = field(default_factory=lambda: DEFAULT_BASE_DIR)
    start_date: date = DEFAULT_START_DATE
    end_date: date = DEFAULT_END_DATE
    max_documents: int = DEFAULT_MAX_DOCUMENTS


def get_document_store(base_dir: Path) -> DuckDBDocumentStore:
    """Return a persisted DuckDB document store.

    Args:
        base_dir: The root directory used to store pipeline
            artifacts. The document store is created at
            ``base_dir / "document_store" / "documents.duckdb"``.

    Returns:
        A DuckDB-backed document store rooted at ``base_dir``.
    """
    return DuckDBDocumentStore(base_dir / "document_store" / "documents.duckdb")


def build_ingestor(config: Config) -> BaseIngestor:
    """Build the S&P 1500 filing ingestor rooted at ``config.base_dir``.

    Args:
        config: The pipeline configuration specifying the ticker,
            base directory, and filing date range.

    Returns:
        An ingestor that downloads 10-K and 10-Q filings for
        ``config.ticker`` and stores them in the document store.
    """
    return SecFilingDocumentStoreIngestor(
        filing_ingestor=SecFilingIngestor(
            company_ingestor=ProcessorIngestor(
                source=SecCompanyIngestor(ingestor=InMemoryIngestor([config.ticker])),
                processor=SequenceProcessor(EdgarCompanyToIdentifierProcessor()),
            ),
            output_dir=config.base_dir / "sec",
            start_date=config.start_date,
            end_date=config.end_date,
            forms=[SecForm.TEN_K, SecForm.TEN_Q],
        ),
        document_store=get_document_store(config.base_dir),
        processor=SequenceProcessor(SecFilingRecordToDocumentProcessor()),
    )


def download_data(config: Config) -> None:
    """Download and index SEC filings for the ticker in ``config``.

    Args:
        config: The pipeline configuration specifying the ticker,
            base directory, and filing date range.
    """
    ingestor = build_ingestor(config)
    logger.info("%s", ingestor)
    ingestor.ingest()


def process_data(config: Config) -> None:
    """Query the document store and print the filings found for the
    ticker in ``config``, ordered by filing date.

    Args:
        config: The pipeline configuration specifying the ticker,
            base directory, and the maximum number of documents to
            summarize.

    Raises:
        RuntimeError: If the pipeline produces no output for
            ``config.ticker``, e.g. because no filings were found in
            the document store.
    """
    model = init_chat_model(model=DEFAULT_MODEL, temperature=0)
    logger.info("%s", str_pydantic_model(model, exclude_none=True))
    inner_agent = create_agent(model=model, system_prompt=GENERIC_SYSTEM_PROMPT)
    agent = RecentDocumentsAgent(inner_agent=inner_agent, max_documents=config.max_documents)

    pipeline = TickerDocumentAgentPipeline(
        tickers=[config.ticker],
        document_store=get_document_store(config.base_dir),
        agent=agent,
    )
    logger.info("%s", pipeline)

    outputs = pipeline.execute()
    logger.info("Found %d outputs", len(outputs))
    if not outputs:
        msg = f"No outputs were produced for ticker {config.ticker!r}; is the document store empty?"
        raise RuntimeError(msg)

    print_pretty(outputs[0]["messages"])
    print_markdown(outputs[0]["messages"][-1].content)


@click.command()
@click.option("--ticker", prompt="Enter a ticker", help="The ticker of the company to analyze")
@click.option(
    "--start-date",
    type=str,
    default=DEFAULT_START_DATE.isoformat(),
    show_default=True,
    help="Earliest filing date to ingest, as an ISO string (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    type=str,
    default=DEFAULT_END_DATE.isoformat(),
    show_default=True,
    help="Latest filing date to ingest, as an ISO string (YYYY-MM-DD).",
)
@click.option(
    "--max-documents",
    type=int,
    default=DEFAULT_MAX_DOCUMENTS,
    show_default=True,
    help="Maximum number of most recent filings to summarize.",
)
def main(ticker: str, start_date: str, end_date: str, max_documents: int) -> None:
    """Run the document indexing pipeline and inspect the vector store.

    Args:
        ticker: The ticker symbol of the company to analyze.
        start_date: The earliest filing date to ingest, as an ISO
            string (``YYYY-MM-DD``).
        end_date: The latest filing date to ingest, as an ISO string
            (``YYYY-MM-DD``).
        max_documents: The maximum number of most recent filings to
            pass to the summarization agent.
    """
    config = Config(
        ticker=ticker,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        max_documents=max_documents,
    )

    download_data(config)
    process_data(config)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
