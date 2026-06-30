r"""Provide code to explore a document search pipeline."""

from __future__ import annotations

import logging
import random
import time
from datetime import date
from pathlib import Path

from coola.utils.format import str_time_human
from dotenv import load_dotenv
from zenpyre.utils.dataclass_io import load_dataclasses
from zenpyre.utils.rich import configure_rich_logging, make_progressbar

from glyphik.data.sec import SecForm, fetch_filings
from glyphik.data.sp1500 import Sp1500Company

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Download SEC filings for a list of cached S&P 1500 companies.

    Loads the processed S&P 1500 company list from disk and downloads
    their ``10-K`` and ``10-Q`` filings filed between 2024-01-01 and
    2026-06-01 via :func:`~glyphik.data.sec.fetch_filings`.  Companies
    without a known CIK are skipped, and failures for individual
    companies are logged and do not interrupt the run.  Logs the total
    time taken once all companies have been processed.
    """
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"
    companies_path = base_dir / "sp1500" / "companies_processed.json"
    data_path = base_dir / "sec"

    companies = load_dataclasses(companies_path, Sp1500Company)
    # companies = companies[:100]  # limit for local development
    random.shuffle(companies)

    t_start = time.perf_counter()

    with make_progressbar() as progress:
        task = progress.add_task("Downloading SEC filings...", total=len(companies))

        for company in companies:
            logger.info("Downloading SEC filings for %s...", company.ticker)
            if company.cik is None:
                logger.warning("Skipping %s: no CIK available", company.ticker)
                progress.advance(task)
                continue

            try:
                fetch_filings(
                    cik_or_ticker=company.cik,
                    start_date=date(2025, 1, 1),
                    end_date=date(2026, 6, 1),
                    output_dir=data_path,
                    forms=[SecForm.TEN_K, SecForm.TEN_Q],
                )
            except Exception:  # noqa: BLE001 - one company's failure shouldn't abort the batch run
                logger.warning(
                    "Failed to fetch filings for %s (CIK %s)", company.ticker, company.cik
                )

            progress.advance(task)

    logger.info(
        "Downloaded filings for %s companies in %s",
        f"{len(companies):,}",
        str_time_human(time.perf_counter() - t_start),
    )


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO)
    main()
