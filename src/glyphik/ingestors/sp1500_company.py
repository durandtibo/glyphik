r"""Define an ingestor that loads the S&P 1500 company list from disk or
Wikipedia."""

from __future__ import annotations

__all__ = ["Sp1500CompanyIngestor"]

import logging
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.utils.path import sanitize_path
from zenpyre.ingestors.base import BaseIngestor

from glyphik.data.sp1500 import Company, load_or_fetch_sp1500_companies

if TYPE_CHECKING:
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


class Sp1500CompanyIngestor(BaseIngestor[list[Company]], InlineDisplayMixin):
    """Ingestor that loads the S&P 1500 company list.

    Loads the list of S&P 1500 companies from a cached JSON file if it
    exists, or fetches it from Wikipedia and caches it otherwise.
    Optionally fills in missing CIK numbers via
    :func:`~glyphik.data.sp1500.fill_missing_ciks`.

    Args:
        path: Path to the JSON cache file.
        find_missing_ciks: If ``True`` (the default), looks up CIK
            numbers for companies whose ``cik`` is ``None`` after
            fetching.  Has no effect when loading from an existing
            cache.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from glyphik.ingestors import Sp1500CompanyIngestor
        >>> ingestor = Sp1500CompanyIngestor(path=Path("sp1500.json"))
        >>> companies = ingestor.ingest()  # doctest: +SKIP

        ```
    """

    def __init__(self, path: Path | str, find_missing_ciks: bool = True) -> None:
        self._path = sanitize_path(path)
        self._find_missing_ciks = find_missing_ciks

    def ingest(self) -> list[Company]:
        """Load or fetch the S&P 1500 company list.

        Returns:
            A list of :class:`~glyphik.data.sp1500.Company` instances,
            either loaded from the cache at :attr:`_path` or freshly
            fetched from Wikipedia and cached.
        """
        return load_or_fetch_sp1500_companies(
            path=self._path, find_missing_ciks=self._find_missing_ciks
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"path": self._path, "find_missing_ciks": self._find_missing_ciks}
