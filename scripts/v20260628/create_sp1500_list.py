r"""Provide code to explore a document search pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from zenpyre.utils.rich import configure_rich_logging, print_pretty

from glyphik.data.sp1500 import load_or_fetch_sp1500_companies

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    """Load, process, and cache S&P 1500 company data.

    Loads or fetches the raw S&P 1500 company list, fills in any missing
    CIK numbers via :func:`process_companies`, and caches the processed
    result to disk.  On subsequent runs, the cached processed file is
    loaded directly instead of being recomputed.
    """
    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"
    sp1500_dir = base_dir / "sp1500"
    companies_path = sp1500_dir / "companies.json"

    companies = load_or_fetch_sp1500_companies(companies_path)

    print_pretty(companies[:10], title=f"S&P 1500 Companies (n={len(companies):,})")


if __name__ == "__main__":
    load_dotenv()
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
