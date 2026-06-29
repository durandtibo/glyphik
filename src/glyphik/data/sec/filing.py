r"""Download SEC filings using the edgartools library."""

from __future__ import annotations

__all__ = ["SecFilingRecord", "fetch_filings", "fetch_form_filings"]

import datetime
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from coola.utils.path import sanitize_path

from glyphik.data.sec import SecForm, fetch_ticker_from_cik
from glyphik.utils.imports import is_edgar_available

if is_edgar_available():
    from edgar import Company
    from edgar.entity.utils import format_cik


if TYPE_CHECKING:
    from pathlib import Path

    from edgar import Filing

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SecFilingRecord:
    """A downloaded SEC filing with optional metadata.

    Args:
        filing: The :class:`~edgar.Filing` object returned by
            edgartools.
        id: Optional unique identifier for the filing. Defaults to
            the accession number when available.
        metadata: Dict of metadata associated with the filing
            (e.g. accession number, CIK, ticker, form type).
    """

    filing: Filing | None = None
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def fetch_filings(
    cik: int,
    start_date: date,
    output_dir: Path | str,
    end_date: date | None = None,
    forms: list[str] | None = None,
    force_download: bool = False,
) -> list[SecFilingRecord]:
    """Load or download SEC filings for a company.

    For each form type in ``forms``, fetches filings filed between
    ``start_date`` and ``end_date`` for the company identified by
    ``cik``.  Each filing is saved to ``output_dir / <cik> / <form> /
    <accession_number>.pkl`` if it does not already exist, or if
    ``force_download=True``.

    Args:
        cik: The SEC Central Index Key (CIK) of the company.
        start_date: The start of the filing date range (inclusive).
        output_dir: The directory to save filings to.
        end_date: The end of the filing date range (inclusive).
            Defaults to today.
        forms: The list of SEC form types to download.  Defaults to
            ``[SecForm.TEN_K, SecForm.TEN_Q]``.
        force_download: If ``True``, re-downloads filings that already
            exist on disk.  Defaults to ``False``.

    Returns:
        A list of :class:`SecFilingRecord` instances, one per
        downloaded or skipped filing across all form types.
    """
    output_dir = sanitize_path(output_dir)
    end_date = end_date or datetime.datetime.now(tz=datetime.UTC).date()
    forms = forms or [SecForm.TEN_K, SecForm.TEN_Q]
    date_range = f"{start_date}:{end_date}"

    company = Company(cik)
    documents: list[SecFilingRecord] = []

    for form in forms:
        form_dir = output_dir / format_cik(company.cik) / form
        form_dir.mkdir(parents=True, exist_ok=True)

        documents.extend(
            fetch_form_filings(
                company=company,
                output_dir=form_dir,
                form=form,
                date_range=date_range,
                force_download=force_download,
            )
        )

    return documents


def fetch_form_filings(
    *,
    company: Company,
    output_dir: Path,
    form: str,
    date_range: str,
    force_download: bool = False,
) -> list[SecFilingRecord]:
    """Download filings of a single form type for a company.

    Iterates over all filings of ``form`` filed within ``date_range``
    and saves each one to ``output_dir / <accession_number>.pkl``.
    Skips existing files unless ``force_download=True``.

    Args:
        company: The :class:`~edgar.Company` instance to fetch filings
            for.
        output_dir: The directory to save filings to.
        form: The SEC form type (e.g. ``"10-K"``).
        date_range: A date range string in the format
            ``"YYYY-MM-DD:YYYY-MM-DD"``.
        force_download: If ``True``, re-downloads filings that already
            exist on disk.  Defaults to ``False``.

    Returns:
        A list of :class:`SecFilingRecord` instances for all filings
        in the date range.
    """
    filings = company.get_filings(form=form).filter(filing_date=date_range)

    cik = company.cik
    ticker = fetch_ticker_from_cik(cik)
    documents: list[SecFilingRecord] = []

    for filing in filings:
        accession = filing.accession_number
        filing_date_raw = filing.filing_date
        filing_date = (
            filing_date_raw
            if isinstance(filing_date_raw, date)
            else date.fromisoformat(str(filing_date_raw))
        )
        filepath = output_dir / f"{accession}.pkl"

        if filepath.is_file() and not force_download:
            logger.info(
                "Skipping %s %s filed on %s — already exists.",
                cik,
                form,
                filing_date,
            )
        else:
            logger.info("Downloading %s %s filed on %s...", cik, form, filing_date)
            filing.save(filepath)
            logger.info("Saved to %s.", filepath)

        documents.append(
            SecFilingRecord(
                filing=filing,
                id=accession,
                metadata={
                    "accession_no": accession,
                    "cik": cik,
                    "company_name": company.name,
                    "filepath": filepath.as_posix(),
                    "form": form,
                    "source": "SEC EDGAR",
                    "ticker": ticker,
                },
            )
        )

    return documents
