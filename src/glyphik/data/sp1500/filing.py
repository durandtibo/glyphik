r"""Provide cache-or-fetch utilities for SEC filings of S&P 1500
companies."""

from __future__ import annotations

__all__ = ["Config", "load_or_fetch_company_filings", "load_or_fetch_filings"]

import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.hashing import hash_object
from coola.utils.format import str_time_human
from coola.utils.path import sanitize_path
from zenpyre.utils.dataclass_io import load_dataclasses, save_dataclasses
from zenpyre.utils.rich import make_progressbar

from glyphik.data.sec import SecFilingRecord, fetch_filings
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    from edgar.entity.utils import format_cik

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

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


def load_or_fetch_filings(
    companies: list[Sp1500Company],
    output_dir: Path | str,
    start_date: date,
    end_date: date,
    forms: list[str],
) -> list[SecFilingRecord]:
    """Download or load cached SEC filings for a list of companies.

    For each company in ``companies``, delegates to
    :func:`load_or_fetch_company_filings` to either load previously
    cached filings or fetch and cache new ones.  Displays progress via
    a Rich progress bar and logs the total time taken once all
    companies have been processed.

    Args:
        companies: The list of :class:`~glyphik.data.sp1500.Sp1500Company`
            instances to fetch filings for.
        output_dir: The base directory under which filings and cache
            files are stored.
        start_date: The start of the filing date range (inclusive).
        end_date: The end of the filing date range (inclusive).
        forms: The SEC form types to download (e.g. ``["10-K", "10-Q"]``).

    Returns:
        A flat list of :class:`~glyphik.data.sec.SecFilingRecord`
        instances across all companies.  Companies with no known CIK
        or for which fetching failed contribute no records.
    """
    output_dir = sanitize_path(output_dir)
    config = Config(start_date=start_date, end_date=end_date, forms=tuple(forms))
    records: list[SecFilingRecord] = []

    t_start = time.perf_counter()

    with make_progressbar() as progress:
        task = progress.add_task("Downloading SEC filings...", total=len(companies))
        for company in companies:
            records.extend(
                load_or_fetch_company_filings(company=company, config=config, output_dir=output_dir)
            )
            progress.advance(task)

    logger.info(
        "Downloaded filings for %s companies in %s.",
        f"{len(companies):,}",
        str_time_human(time.perf_counter() - t_start),
    )

    return records


def load_or_fetch_company_filings(
    company: Sp1500Company, config: Config, output_dir: Path
) -> list[SecFilingRecord]:
    """Download or load cached SEC filings for a single company.

    If filings for this company and ``config`` were already downloaded
    in a previous run, loads them from the on-disk cache under
    ``<output_dir>/<cik>/.cache/<config.cache_key()>.json`` instead of
    re-fetching.  Otherwise, fetches filings via
    :func:`~glyphik.data.sec.fetch_filings` and caches the result for
    future runs.

    Args:
        company: The :class:`~glyphik.data.sp1500.Sp1500Company` to
            fetch filings for.
        config: The date range and form types to fetch.
        output_dir: The base directory under which filings and cache
            files are stored.

    Returns:
        A list of :class:`~glyphik.data.sec.SecFilingRecord` instances
        for this company.  Empty if the company has no known CIK or if
        fetching failed.
    """
    if company.cik is None:
        logger.warning("Skipping %s: no CIK available.", company.ticker)
        return []

    cache_filepath = output_dir / format_cik(company.cik) / ".cache" / f"{config.cache_key()}.json"
    if cache_filepath.is_file():
        logger.debug("Loading cached filings for %s...", company.ticker)
        return load_dataclasses(cache_filepath, SecFilingRecord)

    logger.info("Downloading SEC filings for %s...", company.ticker)
    try:
        filings = fetch_filings(
            cik_or_ticker=company.cik,
            start_date=config.start_date,
            end_date=config.end_date,
            output_dir=output_dir,
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
