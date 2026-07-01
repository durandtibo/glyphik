r"""Provide code to explore a document search pipeline."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from dotenv import load_dotenv
from zenpyre.utils.dataclass_io import load_dataclasses, save_dataclasses
from zenpyre.utils.rich import configure_rich_logging, make_progressbar, print_pretty

from glyphik.data.sec import fetch_cik_from_ticker
from glyphik.data.sp1500.company import Company, load_or_fetch_sp1500_companies

logger: logging.Logger = logging.getLogger(__name__)


def process_companies(companies: list[Company]) -> list[Company]:
    """Fill in missing CIK numbers for a list of S&P 1500 companies.

    For each company whose ``cik`` is ``None``, looks up its CIK via
    :func:`~glyphik.data.sec.fetch_cik_from_ticker` using its ticker
    symbol, and returns a new company instance with ``cik`` filled in.
    Companies that already have a CIK are returned unchanged.

    Args:
        companies: The list of :class:`~glyphik.data.sp1500.Sp1500Company`
            instances to process.

    Returns:
        A new list of :class:`~glyphik.data.sp1500.Sp1500Company`
        instances, with ``cik`` filled in wherever it was previously
        ``None`` and a matching ticker lookup succeeded.
    """
    with make_progressbar() as progress:
        task = progress.add_task("Filling missing CIKs...", total=len(companies))
        filled_companies = []
        for company in companies:
            if company.cik is None:
                filled_companies.append(
                    dataclasses.replace(company, cik=fetch_cik_from_ticker(company.ticker))
                )
            else:
                filled_companies.append(company)
            progress.advance(task)

    return filled_companies


def main() -> None:
    """Load, process, and cache S&P 1500 company data.

    Loads or fetches the raw S&P 1500 company list, fills in any missing
    CIK numbers via :func:`process_companies`, and caches the processed
    result to disk.  On subsequent runs, the cached processed file is
    loaded directly instead of being recomputed.
    """
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"
    sp1500_dir = base_dir / "sp1500"
    raw_path = sp1500_dir / "companies_raw.json"
    companies_path = sp1500_dir / "companies_processed.json"

    if companies_path.is_file():
        logger.info("Loading cached processed companies from %s...", companies_path)
        companies = load_dataclasses(companies_path, Company)
    else:
        logger.info("No processed cache found, building it...")
        companies = load_or_fetch_sp1500_companies(raw_path)
        companies = process_companies(companies)
        save_dataclasses(companies, companies_path)

    print_pretty(companies, title=f"S&P 1500 Companies (n={len(companies):,})")


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO)
    main()
