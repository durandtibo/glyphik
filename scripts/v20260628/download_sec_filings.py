r"""Provide code to download and cache SEC filings for S&P 1500
companies."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from zenpyre.utils.dataclass_io import load_dataclasses
from zenpyre.utils.rich import configure_rich_logging

from glyphik.data.sec import SecForm
from glyphik.data.sp1500 import load_or_fetch_filings
from glyphik.data.sp1500.company import Company

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Download SEC filings for a list of cached S&P 1500 companies.

    Loads the processed S&P 1500 company list from disk and downloads
    their ``10-K`` and ``10-Q`` filings filed between 2025-01-01 and
    2026-06-01 via :func:`process_company`.  Companies without a known
    CIK are skipped, and failures for individual companies are logged
    and do not interrupt the run.  Results are cached on disk so that
    re-running with the same configuration skips already-downloaded
    companies.  Logs the total time taken once all companies have been
    processed.
    """
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"
    companies_path = base_dir / "sp1500" / "companies_processed.json"
    data_path = base_dir / "sec"

    companies = load_dataclasses(companies_path, Company)
    companies = companies[:100]  # limit for local development

    filings = load_or_fetch_filings(
        companies=companies,
        output_dir=data_path,
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=[SecForm.TEN_K, SecForm.TEN_Q],
    )

    logger.info(
        "Downloaded %s filings for %s companies", f"{len(filings):,}", f"{len(companies):,}"
    )


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO)
    main()
