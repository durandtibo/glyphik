r"""Define a script to compare multiple filing formats."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from rich import get_console
from rich.table import Table
from zenpyre.data_processors import SequenceProcessor
from zenpyre.ingestors import InMemoryIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging

from glyphik.data.sec import SecForm
from glyphik.data_processors import EdgarCompanyToIdentifierProcessor
from glyphik.ingestors import SecCompanyIngestor, SecFilingIngestor

if TYPE_CHECKING:
    from edgar import Filing


logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_COLORS = ["green", "yellow", "red", "blue", "magenta", "cyan", "white"]

DEFAULT_TICKERS = ["AAPL"]
DEFAULT_BASE_DIR = Path(__file__).parent.parent.parent / "tmp/v20260628"
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2026, 6, 1)


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
        Each dict contains ``"company"``, ``"form"``, ``"accession_no"``,
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

    for filing in filings:
        row = {
            "company": getattr(filing, "company", None) or str(filing),
            "form": getattr(filing, "form", None),
            "accession_no": getattr(filing, "accession_no", None),
        }

        for fmt in formats:
            content = ""
            try:
                attr = getattr(filing, fmt)
                content = attr() if callable(attr) else attr
                content = content or ""
            except Exception:  # noqa: BLE001
                logger.warning("Failed to retrieve %r for %s", fmt, row["company"], exc_info=True)

            row[f"{fmt}_chars"] = len(content)

        results.append(row)

    return results


def print_comparison(results: list[dict], formats: list[str]) -> None:
    """Print a Rich table of per-filing character counts and per-format
        averages.

        Renders one row per filing with its character count for each
        requested format, followed by a separated summary row with the
        average count per format.

    Args:
        results: Per-filing comparison data, as returned by
            ``compare_filing_formats``.
        formats: Names of the formats to render as columns, in the order
            they should appear. Must match the ``"{format}_chars"`` keys
            present in ``results``.

    Returns:
        None. The table is printed directly to the console.

    Example:
        ```pycon
        >>> results = compare_filing_formats(filings, ["text", "markdown", "html"])
        >>> print_comparison(results, ["text", "markdown", "html"])

        ```
    """
    console = get_console()

    table = Table(title="Filing Format Character Count Comparison", show_lines=False)
    table.add_column("Company", style="cyan", no_wrap=False, max_width=30)
    table.add_column("Form", style="magenta")

    for i, fmt in enumerate(formats):
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        table.add_column(fmt.capitalize(), justify="right", style=color)

    for r in results:
        table.add_row(
            r["company"],
            str(r["form"]),
            *[f"{r[f'{fmt}_chars']:,}" for fmt in formats],
        )

    if results:
        table.add_section()
        avg_row = [
            f"[bold]{mean(r[f'{fmt}_chars'] for r in results):,.0f}[/bold]" for fmt in formats
        ]
        table.add_row("[bold]Average[/bold]", "", *avg_row)

    console.print(table)


@dataclass
class Config:
    """Hold the configuration for the document search pipeline example.

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


def get_filings(config: Config) -> list[Filing]:
    """Ingest 10-K and 10-Q filings for the configured tickers and date
    range.

    Args:
        config: The pipeline configuration specifying tickers, output
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
    config = Config()
    filings = get_filings(config)
    logger.info(f"Found {len(filings)} filings.")

    if not filings:
        logger.warning("No filings found; skipping comparison.")
        return

    formats = ["text", "markdown", "html"]
    results = compare_filing_formats(filings, formats)
    print_comparison(results, formats)


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
