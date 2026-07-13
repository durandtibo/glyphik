r"""Define a script to compare multiple filing formats."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from rich import get_console
from rich.table import Table
from zenpyre.data_processors import SequenceProcessor
from zenpyre.ingestors import InMemoryIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging, make_progressbar

from glyphik.data.sec import SecForm
from glyphik.data_processors import EdgarCompanyToIdentifierProcessor
from glyphik.ingestors import SecCompanyIngestor, SecFilingIngestor

if TYPE_CHECKING:
    from edgar import Filing
    from langchain_core.language_models import BaseChatModel

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL"]
DEFAULT_BASE_DIR = Path(__file__).parent.parent.parent / "tmp/v20260628"
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 1)


@dataclass
class Config:
    """Hold the configuration for the document search pipelines example.

    Args:
        tickers: The ticker symbols of the companies to analyze.
        base_dir: The root directory used to store downloaded filings
            and the document store.
        start_date: The earliest filing date (inclusive) to ingest.
        end_date: The latest filing date (inclusive) to ingest.
    """

    tickers: list[str] = field(default_factory=lambda: DEFAULT_TICKERS)
    base_dir: Path = field(default_factory=lambda: DEFAULT_BASE_DIR)
    start_date: date = DEFAULT_START_DATE
    end_date: date = DEFAULT_END_DATE


def count_input_tokens(model: BaseChatModel, text: str) -> int:
    """Count the number of input tokens a text would consume when sent to
    a chat model.

    Uses the model's own tokenizer via ``get_num_tokens`` when available,
    so the count reflects the model-specific tokenization scheme (e.g.
    tiktoken for OpenAI models) rather than a generic approximation.

    Args:
        model: The chat model that will receive the text.
        text: The text to tokenize and count.

    Returns:
        The number of tokens ``text`` would occupy as input to ``model``.

    Example:
        ```pycon
        >>> from langchain_openai import ChatOpenAI
        >>> model = ChatOpenAI(model="gpt-4o-mini")
        >>> count_input_tokens(model, "Hello, world!")
        4

        ```
    """
    response = model.invoke(
        [SystemMessage("Read the text and return an empty message."), HumanMessage(content=text)]
    )
    return response.usage_metadata["input_tokens"]


def compare_filing_formats(filings: list[Filing], formats: list[str]) -> list[dict]:
    """Compare character counts of arbitrary representations of a list of
    edgar Filing objects.

    Each requested format must correspond to a zero-argument method or a
    plain attribute on ``Filing`` that returns a string (e.g. ``"text"``,
    ``"markdown"``, ``"html"``). If retrieving a format fails or returns
    ``None`` for a given filing, its count is recorded as ``0`` and a
    warning is logged.

    Args:
        filings: The edgar Filing objects to compare.
        formats: Names of zero-arg methods (or attributes) on ``Filing``
            to compare, e.g. ``["text", "markdown", "html"]``.

    Returns:
        A list of dicts, one per filing, in the same order as ``filings``.
        Each dict contains ``"company"``, ``"form"``, ``"accession_number"``,
        and one ``"{format}_chars"`` key per entry in ``formats``. Returns
        an empty list if ``filings`` is empty.

    Example:
        ```pycon
        >>> results = compare_filing_formats(filings, ["text", "markdown", "html"])
        >>> results[0]["text_chars"]
        48213

        ```
    """
    results = []

    with make_progressbar() as progress:
        task = progress.add_task("Comparing filing formats...", total=len(filings))

        for filing in filings:
            row = {
                "company": filing.company,
                "form": filing.form,
                "accession_number": filing.accession_number,
            }

            progress.update(task, description=f"Processing {row['company']}")

            for fmt in formats:
                content = ""
                try:
                    attr = getattr(filing, fmt)
                    content = attr() if callable(attr) else attr
                    content = content or ""
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to retrieve %r for %s", fmt, row["company"], exc_info=True
                    )

                row[f"{fmt}_chars"] = len(content)

            results.append(row)
            progress.advance(task)

    return results


def compare_filing_token_counts(
    filings: list[Filing], formats: list[str], model: BaseChatModel
) -> list[dict]:
    """Compare token counts of arbitrary representations of a list of
    edgar Filing objects, as counted by a given chat model.

    Each requested format must correspond to a zero-argument method or a
    plain attribute on ``Filing`` that returns a string (e.g. ``"text"``,
    ``"markdown"``, ``"html"``). If retrieving a format fails or returns
    ``None`` for a given filing, its token count is recorded as ``0`` and
    a warning is logged.

    Displays a Rich progress bar tracking filing-level progress while
    the formats are retrieved and tokenized.

    Args:
        filings: The edgar Filing objects to compare.
        formats: Names of zero-arg methods (or attributes) on ``Filing``
            to compare, e.g. ``["text", "markdown", "html"]``.
        model: The chat model whose tokenizer is used to count tokens,
            via ``count_input_tokens``.

    Returns:
        A list of dicts, one per filing, in the same order as ``filings``.
        Each dict contains ``"company"``, ``"form"``, ``"accession_number"``,
        and one ``"{format}_chars"`` key per entry in ``formats``, where
        the value is a token count rather than a character count. The key
        name is kept as ``"{format}_chars"`` for compatibility with
        ``print_comparison``. Returns an empty list if ``filings`` is
        empty.

    Example:
        ```pycon
        >>> from langchain_anthropic import ChatAnthropic
        >>> model = ChatAnthropic(model="claude-sonnet-4-5")
        >>> results = compare_filing_token_counts(filings, ["text", "markdown", "html"], model)
        >>> results[0]["text_chars"]
        12044

        ```
    """
    results = []

    with make_progressbar() as progress:
        task = progress.add_task("Comparing filing token counts...", total=len(filings))

        for filing in filings:
            row = {
                "company": filing.company,
                "form": filing.form,
                "accession_number": filing.accession_number,
            }

            progress.update(task, description=f"Processing {row['company']}")

            for fmt in formats:
                content = ""
                try:
                    attr = getattr(filing, fmt)
                    content = attr() if callable(attr) else attr
                    content = content or ""
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to retrieve %r for %s", fmt, row["company"], exc_info=True
                    )

                try:
                    row[f"{fmt}_chars"] = count_input_tokens(model, content) if content else 0
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Failed to count tokens for %r on %s", fmt, row["company"], exc_info=True
                    )
                    row[f"{fmt}_chars"] = 0

            results.append(row)
            progress.advance(task)

    return results


def print_comparison(results: list[dict], formats: list[str]) -> None:
    """Print a Rich table of per-filing character counts and per-format
    averages.

    Renders one row per filing with its character count for each
    requested format, followed by a separated summary row with the
    average count per format. The first format in ``formats`` is
    treated as the reference: every other format's cell shows its
    count alongside its percentage difference relative to the
    reference, formatted as "count (+/-pct%)", colored red if greater
    than the reference and green if lower.

    Args:
        results: Per-filing comparison data, as returned by
            ``compare_filing_formats``.
        formats: Names of the formats to render as columns, in the order
            they should appear. The first entry is used as the
            reference format for the percentage-difference figures.
            Must match the ``"{format}_chars"`` keys present in
            ``results``.

    Example:
        ```pycon
        >>> results = compare_filing_formats(filings, ["text", "markdown", "html"])
        >>> print_comparison(results, ["text", "markdown", "html"])

        ```
    """
    reference, *others = formats

    table = Table(title="Filing Format Character Count Comparison", show_lines=False)
    table.add_column("Company", no_wrap=False, max_width=30)
    table.add_column("Form")
    table.add_column(reference.capitalize(), justify="right")

    for fmt in others:
        table.add_column(fmt.capitalize(), justify="right")

    def format_cell(value: float, reference_value: float, count_decimals: int = 0) -> str:
        count_str = f"{value:,.{count_decimals}f}"
        if not reference_value:
            return f"{count_str} (n/a)"
        diff = (value - reference_value) / reference_value * 100
        sign = "+" if diff >= 0 else ""
        color = "red" if diff > 0 else "green" if diff < 0 else "white"
        return f"[{color}]{count_str} ({sign}{diff:.1f}%)[/{color}]"

    for r in results:
        row = [r["company"], str(r["form"]), f"{r[f'{reference}_chars']:,}"]
        row.extend(format_cell(r[f"{fmt}_chars"], r[f"{reference}_chars"]) for fmt in others)
        table.add_row(*row)

    if results:
        table.add_section()
        avg_reference = mean(r[f"{reference}_chars"] for r in results)
        avg_row = ["[bold]Average[/bold]", "", f"[bold]{avg_reference:,.1f}[/bold]"]
        avg_row.extend(
            f"[bold]{format_cell(mean(r[f'{fmt}_chars'] for r in results), avg_reference, count_decimals=1)}[/bold]"
            for fmt in others
        )
        table.add_row(*avg_row)

    get_console().print(table)


def get_filings(config: Config) -> list[Filing]:
    """Ingest 10-K and 10-Q filings for the configured tickers and date
    range.

    Args:
        config: The pipelines configuration specifying tickers, output
            directory, and date range.

    Returns:
        The loaded ``Filing`` objects, in ingestion order.
    """
    ingestor = SecFilingIngestor(
        company_ingestor=ProcessorIngestor(
            source=SecCompanyIngestor(ingestor=InMemoryIngestor(config.tickers)),
            processor=SequenceProcessor(EdgarCompanyToIdentifierProcessor()),
        ),
        output_dir=config.base_dir / "sec",
        start_date=config.start_date,
        end_date=config.end_date,
        forms=[SecForm.TEN_K, SecForm.TEN_Q],
    )
    return [record.load_filing() for record in ingestor.ingest()]


def main() -> None:
    r"""Define the main function."""
    # config = Config(tickers=["AAPL"])
    config = Config(tickers=["AAPL", "MSFT", "NVDA", "GOOGL"])
    filings = get_filings(config)
    logger.info(f"Found {len(filings)} filings.")

    if not filings:
        logger.warning("No filings found; skipping comparison.")
        return

    formats = ["text", "markdown", "html", "full_text_submission"]
    results = compare_filing_formats(filings, formats)
    print_comparison(results, formats)

    model = init_chat_model(model="ollama:gemma4:e2b-mlx", temperature=0)
    formats = ["text", "markdown"]
    results = compare_filing_token_counts(filings=filings, formats=formats, model=model)
    print_comparison(results, formats)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
