r"""Provide code to explore a document search pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from coola.utils.path import sanitize_path
from coola.utils.string import slugify
from dotenv import load_dotenv
from zenpyre.agents.factory import BaseAgentFactory
from zenpyre.chat_models import ChatModelConfig
from zenpyre.ingestors.factory import BaseIngestorFactory
from zenpyre.utils.config import Config, ExtraFieldsConfig
from zenpyre.utils.resolve import resolve_object
from zenpyre.utils.rich import configure_rich_logging, print_markdown, print_pretty

from glyphik.data.sec import SecForm, get_company_identifiers_from_tickers
from glyphik.pipelines.factory import SecDocumentSummarizationPipelineFactory
from glyphik.prompts.summarization import GENERIC_SYSTEM_PROMPT
from glyphik.utils.config import SecDataConfig

if TYPE_CHECKING:
    from zenpyre.utils.config import BaseConfig

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 1)
DEFAULT_MAX_DOCUMENTS = 1

# NOTE: verify this against the models actually available in your Ollama
# install (e.g. `ollama list`). "gemma4" / "-mlx" does not match any known
# released Ollama tag as of this writing.
DEFAULT_MODEL = "ollama:gemma4:e2b-mlx"

DEFAULT_TICKERS = ["AAPL", "IBM", "MSFT", "NVDA", "GOOGL"]


@dataclass(frozen=True)
class ExperimentConfig(ExtraFieldsConfig):
    r"""Hold the full configuration for a pipelines run.

    Bundles the storage location with the data-ingestion and
    agent-summarization sub-configurations.

    Attributes:
        base_dir: The root directory used to store pipelines
            artifacts, such as the document store and downloaded
            filings.
        data: The configuration for which filings to ingest.
        agent: The configuration for the summarization agent.
    """

    base_dir: Path
    data: BaseConfig
    ingestor: BaseConfig
    agent: BaseConfig

    def __hash__(self) -> int:
        return ExtraFieldsConfig.__hash__(self)


def ingest_data(config: ExperimentConfig) -> None:
    """Download and index SEC filings for the ticker in ``config``.

    Args:
        config: The pipelines configuration specifying the ticker,
            base directory, and filing date range.
    """
    factory = resolve_object(config.ingestor.to_kwargs(), cls=BaseIngestorFactory)
    ingestor = factory.make_ingestor()
    print_pretty(ingestor, title="Ingestor")
    ingestor.ingest()
    logger.info("<<< 🚀 data ingestion complete 🚀 >>>\n\n")


def process_data(config: ExperimentConfig) -> None:
    """Query the document store and print the filings found for the
    ticker in ``config``, ordered by filing date.

    Args:
        config: The pipelines configuration specifying the ticker,
            base directory, and the maximum number of documents to
            summarize.

    Raises:
        RuntimeError: If the pipelines produces no output for
            ``config.ticker``, e.g. because no filings were found in
            the document store.
    """
    agent_factory = resolve_object(config.agent.to_kwargs(), cls=BaseAgentFactory)
    factory = SecDocumentSummarizationPipelineFactory(
        companies=config.data.get_value("companies"),
        agent_factory=agent_factory,
        base_dir=config.get_value("base_dir"),
    )
    pipeline = factory.make_pipeline()
    print_pretty(pipeline, title="Pipeline")

    outputs = pipeline.run()
    logger.info("Found %d outputs", len(outputs))

    for output in outputs:
        # print_pretty(output)
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
def main(ticker: str, start_date: Any, end_date: Any, max_documents: int) -> None:
    """Run the document indexing pipelines and inspect the vector store.

    Args:
        ticker: The ticker symbol of the company to analyze.
        start_date: The earliest filing date to ingest. Parsed and
            validated by Click as ``YYYY-MM-DD``.
        end_date: The latest filing date to ingest. Parsed and
            validated by Click as ``YYYY-MM-DD``.
        max_documents: The maximum number of most recent filings to
            pass to the summarization agent.
    """
    base_dir = sanitize_path(Path(__file__).parent.parent / "tmp/examples/summarization")

    data_config = SecDataConfig.from_kwargs(
        companies=tuple(get_company_identifiers_from_tickers([ticker])),
        start_date=start_date.date(),
        end_date=end_date.date(),
        forms=(SecForm.TEN_K, SecForm.TEN_Q),
    )

    ingestor_config = Config.from_kwargs(
        _target_="glyphik.ingestors.factory.SecFilingIngestorFactory",
        companies=data_config.get_value("companies"),
        start_date=data_config.get_value("start_date"),
        end_date=data_config.get_value("end_date"),
        forms=data_config.get_value("forms"),
        base_dir=base_dir,
    )

    chat_model_config = ChatModelConfig.from_kwargs(
        _target_="zenpyre.chat_models.factory.InitChatModelFactory",
        model=DEFAULT_MODEL,
        temperature=0,
    )

    inner_agent_config = Config.from_kwargs(
        _target_="glyphik.agents.factory.RecentDocumentsAgentFactory",
        agent_factory=Config.from_kwargs(
            _target_="zenpyre.agents.factory.CreateAgentFactory",
            chat_model_factory=chat_model_config,
            system_prompt=GENERIC_SYSTEM_PROMPT,
        ),
        max_documents=max_documents,
    )
    cache_dir = (
        base_dir
        / "agent_outputs"
        / slugify(chat_model_config.model)
        / inner_agent_config.cache_key()
    )
    agent_config = Config.from_kwargs(
        _target_="zenpyre.agents.factory.CachingAgentFactory",
        agent_factory=inner_agent_config,
        cache_dir=cache_dir,
    )
    config = ExperimentConfig.from_kwargs(
        base_dir=base_dir,
        data=data_config,
        agent=agent_config,
        ingestor=ingestor_config,
    )

    print_pretty(config, title="Experiment Config")
    print_pretty(config.cache_key(), title="Experiment ID")

    try:
        ingest_data(config)
        process_data(config)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
