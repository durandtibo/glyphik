r"""Define an ingestor that fetches SEC filings for a list of
companies."""

from __future__ import annotations

__all__ = ["Sp1500FilingIngestor"]

import logging
import time
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.format import str_time_human
from coola.utils.path import sanitize_path
from zenpyre.ingestors.base import BaseIngestor

from glyphik.data.sec import SecFilingRecord
from glyphik.data.sp1500 import load_or_fetch_filings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from glyphik.data.sp1500.company import Company

logger: logging.Logger = logging.getLogger(__name__)


class Sp1500FilingIngestor(BaseIngestor[list[SecFilingRecord]], MultilineDisplayMixin):
    """Ingestor that fetches SEC filings for a list of companies.

    Fetches filings of the specified form types, filed between
    ``start_date`` and ``end_date``, for each company returned by
    ``company_ingestor``.  Results are cached on disk under
    ``output_dir`` so that re-running with the same configuration
    skips already-downloaded filings.

    Args:
        company_ingestor: An ingestor that provides the list of
            :class:`~glyphik.data.sp1500.Company` instances to fetch
            filings for.
        output_dir: The base directory under which filings and cache
            files are stored.
        start_date: The start of the filing date range (inclusive).
        end_date: The end of the filing date range (inclusive).
        forms: The SEC form types to download
            (e.g. ``["10-K", "10-Q"]``).

    Example:
        ```pycon
        >>> from datetime import date
        >>> from pathlib import Path
        >>> from glyphik.ingestors import Sp1500FilingIngestor, Sp1500CompanyIngestor
        >>> ingestor = Sp1500FilingIngestor(
        ...     company_ingestor=Sp1500CompanyIngestor(path=Path("sp1500.json")),
        ...     output_dir=Path("sec"),
        ...     start_date=date(2025, 1, 1),
        ...     end_date=date(2026, 6, 1),
        ...     forms=["10-K", "10-Q"],
        ... )
        >>> records = ingestor.ingest()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        company_ingestor: BaseIngestor[list[Company]],
        output_dir: Path | str,
        start_date: date,
        end_date: date,
        forms: Sequence[str],
    ) -> None:
        self._company_ingestor = company_ingestor
        self._output_dir = sanitize_path(output_dir)
        self._start_date = start_date
        self._end_date = end_date
        self._forms = list(forms)

    def ingest(self) -> list[SecFilingRecord]:
        logger.info("Starting to ingest the filings for S&P 1500 companies...")
        t_start = time.perf_counter()
        companies = self._company_ingestor.ingest()
        filings = load_or_fetch_filings(
            companies=companies,
            output_dir=self._output_dir,
            start_date=self._start_date,
            end_date=self._end_date,
            forms=self._forms,
        )
        logger.info(
            "%s filings have been ingested in %s",
            f"{len(filings):,}",
            str_time_human(time.perf_counter() - t_start),
        )
        return filings

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "company_ingestor": self._company_ingestor,
            "output_dir": self._output_dir,
            "start_date": self._start_date,
            "end_date": self._end_date,
            "forms": self._forms,
        }
