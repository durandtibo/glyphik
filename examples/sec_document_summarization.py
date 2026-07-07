r"""Provide code to explore a document search pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from coola.display import str_pydantic_model
from coola.utils.path import sanitize_path
from coola.utils.string import slugify
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from zenpyre.agents import AgentConfig
from zenpyre.data_processors import SequenceProcessor
from zenpyre.document_stores import DuckDBDocumentStore
from zenpyre.ingestors import InMemoryIngestor, ProcessorIngestor
from zenpyre.runnables import CachingRunnable
from zenpyre.utils.rich import configure_rich_logging, print_markdown, print_pretty

from glyphik.agents import RecentDocumentsAgent
from glyphik.data.sec import CompanyIdentifier, SecForm
from glyphik.data_processors import (
    EdgarCompanyToIdentifierProcessor,
    SecFilingRecordToDocumentProcessor,
)
from glyphik.ingestors import (
    SecCompanyIngestor,
    SecFilingDocumentStoreIngestor,
    SecFilingIngestor,
)
from glyphik.pipeline import CompanyDocumentAgentPipeline
from glyphik.prompts.summarization import GENERIC_SYSTEM_PROMPT

if TYPE_CHECKING:
    from zenpyre.ingestors import BaseIngestor

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 1)
DEFAULT_MAX_DOCUMENTS = 1

# NOTE: verify this against the models actually available in your Ollama
# install (e.g. `ollama list`). "gemma4" / "-mlx" does not match any known
# released Ollama tag as of this writing.
DEFAULT_MODEL = "ollama:gemma4:e2b-mlx"


@dataclass(frozen=True)
class DataConfig:
    r"""Hold the configuration for a data ingestion pipeline.

    Attributes:
        ticker: The ticker symbol of the company to ingest data for.
            Automatically stripped of leading/trailing whitespace and
            upper-cased for consistent keys/paths downstream.
        start_date: The earliest date (inclusive) to ingest.
        end_date: The latest date (inclusive) to ingest.
        forms: The SEC forms to ingest. Accepts any iterable (e.g. a
            list) at construction time; always normalized to a
            ``tuple`` so the config stays hashable and immutable.

    Raises:
        ValueError: If ``start_date`` is after ``end_date``.
    """

    ticker: str
    start_date: date = DEFAULT_START_DATE
    end_date: date = DEFAULT_END_DATE
    forms: tuple[str, ...] = (SecForm.TEN_K, SecForm.TEN_Q)

    def __post_init__(self) -> None:
        """Normalize the ticker and validate the date range.

        Strips whitespace and upper-cases ``ticker``, converts
        ``forms`` to a tuple, and checks that ``start_date`` does not
        come after ``end_date``.

        Raises:
            ValueError: If ``start_date`` is after ``end_date``.
        """
        object.__setattr__(self, "ticker", self.ticker.strip().upper())
        object.__setattr__(self, "forms", tuple(self.forms))
        if self.start_date > self.end_date:
            msg = f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            raise ValueError(msg)


@dataclass(frozen=True)
class DocumentsAgentConfig(AgentConfig):
    r"""Hold the configuration for the document-summarization agent.

    Extends ``AgentConfig`` with a limit on how many of the most
    recent documents the agent should process.

    Attributes:
        max_documents: The maximum number of most recent documents to
            pass to the agent for summarization.
    """

    max_documents: int = DEFAULT_MAX_DOCUMENTS


@dataclass(frozen=True)
class ExperimentConfig:
    r"""Hold the full configuration for a pipeline run.

    Bundles the storage location with the data-ingestion and
    agent-summarization sub-configurations.

    Attributes:
        base_dir: The root directory used to store pipeline
            artifacts, such as the document store and downloaded
            filings.
        data: The configuration for which filings to ingest.
        agent: The configuration for the summarization agent.
    """

    base_dir: Path
    data: DataConfig
    agent: DocumentsAgentConfig


def get_document_store(base_dir: Path, **kwargs: Any) -> DuckDBDocumentStore:
    """Return a persisted DuckDB document store.

    Args:
        base_dir: The root directory used to store pipeline
            artifacts. The document store is created at
            ``base_dir / "document_store" / "documents.duckdb"``.
        **kwargs: Additional keyword arguments.

    Returns:
        A DuckDB-backed document store rooted at ``base_dir``.
    """
    return DuckDBDocumentStore(base_dir / "document_store" / "documents.duckdb", **kwargs)


def build_ingestor(config: ExperimentConfig) -> BaseIngestor:
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
                source=SecCompanyIngestor(ingestor=InMemoryIngestor([config.data.ticker])),
                processor=SequenceProcessor(EdgarCompanyToIdentifierProcessor()),
            ),
            output_dir=config.base_dir / "sec",
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            forms=config.data.forms,
        ),
        document_store=get_document_store(config.base_dir),
        processor=SequenceProcessor(SecFilingRecordToDocumentProcessor()),
    )


def download_data(config: ExperimentConfig) -> None:
    """Download and index SEC filings for the ticker in ``config``.

    Args:
        config: The pipeline configuration specifying the ticker,
            base directory, and filing date range.
    """
    ingestor = build_ingestor(config)
    logger.info("%s", ingestor)
    ingestor.ingest()


def process_data(config: ExperimentConfig) -> None:
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
    model = init_chat_model(model=config.agent.model, temperature=config.agent.temperature)
    logger.info("%s", str_pydantic_model(model, exclude_none=True))
    inner_agent = create_agent(model=model, system_prompt=config.agent.system_prompt)
    cache_dir = (
        config.base_dir / "agent_outputs" / slugify(config.agent.model) / config.agent.cache_key()
    )
    agent = RecentDocumentsAgent(
        inner_agent=CachingRunnable(inner_agent, cache_dir=cache_dir),
        max_documents=config.agent.max_documents,
    )

    pipeline = CompanyDocumentAgentPipeline(
        companies=[CompanyIdentifier.from_ticker(config.data.ticker)],
        document_store=get_document_store(config.base_dir, read_only=True),
        agent=agent,
        log_documents_metadata=True,
    )
    logger.info("%s", pipeline)

    outputs = pipeline.execute()
    logger.info("Found %d outputs", len(outputs))
    if not outputs:
        msg = f"No outputs were produced for ticker {config.data.ticker!r}; is the document store empty?"
        raise RuntimeError(msg)

    for output in outputs:
        print_pretty(output)
        print_markdown(output["messages"][-1].content)


@click.command()
@click.option("--ticker", prompt="Enter a ticker", help="The ticker of the company to analyze")
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=DEFAULT_START_DATE.isoformat(),
    show_default=True,
    help="Earliest filing date to ingest (YYYY-MM-DD).",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=DEFAULT_END_DATE.isoformat(),
    show_default=True,
    help="Latest filing date to ingest (YYYY-MM-DD).",
)
@click.option(
    "--max-documents",
    type=int,
    default=DEFAULT_MAX_DOCUMENTS,
    show_default=True,
    help="Maximum number of most recent filings to summarize.",
)
def main(ticker: str, start_date: object, end_date: object, max_documents: int) -> None:
    """Run the document indexing pipeline and inspect the vector store.

    Args:
        ticker: The ticker symbol of the company to analyze.
        start_date: The earliest filing date to ingest. Parsed and
            validated by Click as ``YYYY-MM-DD``.
        end_date: The latest filing date to ingest. Parsed and
            validated by Click as ``YYYY-MM-DD``.
        max_documents: The maximum number of most recent filings to
            pass to the summarization agent.
    """
    try:
        config = ExperimentConfig(
            base_dir=sanitize_path(Path(__file__).parent.parent / "tmp/examples/company"),
            data=DataConfig(
                ticker=ticker,
                start_date=start_date.date(),
                end_date=end_date.date(),
            ),
            agent=DocumentsAgentConfig(
                model=DEFAULT_MODEL,
                system_prompt=GENERIC_SYSTEM_PROMPT,
                temperature=0,
                max_documents=max_documents,
            ),
        )
    except ValueError as e:
        raise click.BadParameter(str(e)) from e

    print_pretty(config)

    download_data(config)
    try:
        process_data(config)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
