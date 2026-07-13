"""Explore a document search pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from zenpyre.data_processors import SequenceProcessor
from zenpyre.ingestors import FirstNIngestor, ProcessorIngestor
from zenpyre.utils.rich import configure_rich_logging, print_pretty

from glyphik.data_processors import (
    Sp1500CompanyToIdentifierProcessor,
)
from glyphik.ingestors import (
    Sp1500CompanyIngestor,
)

if TYPE_CHECKING:
    from zenpyre.ingestors import BaseIngestor

    from glyphik.data.sec import CompanyIdentifier

logger: logging.Logger = logging.getLogger(__name__)


def build_company_ingestor(base_dir: Path) -> BaseIngestor[list[CompanyIdentifier]]:
    """Build the S&P 1500 filing ingestor rooted at ``base_dir``."""
    return ProcessorIngestor(
        source=FirstNIngestor(
            Sp1500CompanyIngestor(path=base_dir / "sp1500" / "companies.json"),
            n=5,
        ),
        processor=SequenceProcessor(Sp1500CompanyToIdentifierProcessor()),
    )


def main() -> None:
    """Build the ingestor, run it, and print a sample of filed
    documents.

    Ingests S&P 1500 filings into a DuckDB-backed document store (or
    reuses the cached store if one already exists at the target path),
    then prints the metadata for one ticker/form combination as a sanity
    check.
    """
    base_dir = Path(__file__).parent.parent / "tmp/debug/ingestor"

    ingestor = build_company_ingestor(base_dir)
    logger.info("%s", ingestor)
    print_pretty(ingestor.ingest())


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
