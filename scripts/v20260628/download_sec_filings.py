r"""Provide code to download and cache SEC filings for S&P 1500
companies."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from coola.hashing import hash_object
from dotenv import load_dotenv
from edgar.entity.utils import format_cik
from zenpyre.utils.dataclass_io import load_dataclasses, save_dataclasses
from zenpyre.utils.rich import configure_rich_logging

from glyphik.data.sec import SecFilingRecord, SecForm, fetch_filings
from glyphik.data.sp1500 import load_or_fetch_filings
from glyphik.data.sp1500.companies import Sp1500Company

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Configuration for a SEC filings download run.

    Args:
        start_date: The start of the filing date range (inclusive).
        end_date: The end of the filing date range (inclusive).
        forms: The SEC form types to download (e.g. ``("10-K", "10-Q")``).
    """

    start_date: date
    end_date: date
    forms: tuple[str, ...]

    def cache_key(self) -> str:
        """Return a stable hash identifying this configuration.

        Used to namespace cached results so that changing the date
        range or form types invalidates the cache rather than silently
        returning stale results.

        Returns:
            A hexadecimal hash string derived from all config fields.
        """
        return hash_object(asdict(self))


def process_company(
    company: Sp1500Company, config: Config, data_path: Path
) -> list[SecFilingRecord]:
    """Download or load cached SEC filings for a single company.

    If filings for this company and ``config`` were already downloaded
    in a previous run, loads them from the on-disk cache under
    ``<data_path>/<cik>/.cache/<config.cache_key()>.json`` instead of
    re-fetching.  Otherwise, fetches filings via
    :func:`~glyphik.data.sec.fetch_filings` and caches the result for
    future runs.

    Args:
        company: The :class:`~glyphik.data.sp1500.Sp1500Company` to
            fetch filings for.
        config: The date range and form types to fetch.
        data_path: The base directory under which filings and cache
            files are stored.

    Returns:
        A list of :class:`~glyphik.data.sec.SecFilingRecord` instances
        for this company.  Empty if the company has no known CIK or if
        fetching failed.
    """
    if company.cik is None:
        logger.warning("Skipping %s: no CIK available.", company.ticker)
        return []

    cache_filepath = data_path / format_cik(company.cik) / ".cache" / f"{config.cache_key()}.json"
    if cache_filepath.is_file():
        logger.debug("Loading cached filings for %s...", company.ticker)
        return load_dataclasses(cache_filepath, SecFilingRecord)

    logger.info("Downloading SEC filings for %s...", company.ticker)
    try:
        filings = fetch_filings(
            cik_or_ticker=company.cik,
            start_date=config.start_date,
            end_date=config.end_date,
            output_dir=data_path,
            forms=list(config.forms),
        )
    except Exception as e:  # noqa: BLE001 - one company's failure shouldn't abort the batch run
        logger.warning(
            "Failed to fetch filings for %s (CIK %s)\n  %s: %s",
            company.ticker,
            company.cik,
            type(e).__name__,
            e,
        )
        return []

    save_dataclasses(filings, cache_filepath)
    return filings


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
    Config(
        start_date=date(2025, 1, 1),
        end_date=date(2026, 6, 1),
        forms=(SecForm.TEN_K, SecForm.TEN_Q),
    )

    base_dir = Path(__file__).parent.parent.parent / "tmp/v20260628"
    companies_path = base_dir / "sp1500" / "companies_processed.json"
    data_path = base_dir / "sec"

    companies = load_dataclasses(companies_path, Sp1500Company)
    # companies = companies[:100]  # limit for local development

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
