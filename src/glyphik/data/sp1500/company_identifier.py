r"""Provide a convenience function to fetch S&P 1500 company
identifiers."""

from __future__ import annotations

__all__ = ["get_sp1500_company_identifiers"]

from typing import TYPE_CHECKING

from zenpyre.data_processors import SequenceProcessor
from zenpyre.ingestors import ProcessorIngestor

from glyphik.data_processors import Sp1500CompanyToIdentifierProcessor
from glyphik.ingestors import Sp1500CompanyIngestor

if TYPE_CHECKING:
    from pathlib import Path

    from glyphik.data.sec import CompanyIdentifier


def get_sp1500_company_identifiers(path: Path | str | None = None) -> list[CompanyIdentifier]:
    r"""Fetch the identifiers (ticker + CIK) of every S&P 1500 company.

    Combines Sp1500CompanyIngestor (which loads the S&P 1500 company
    list from a cached JSON file, or fetches it from Wikipedia and
    caches it otherwise) with Sp1500CompanyToIdentifierProcessor (which
    converts each Company into a CompanyIdentifier).

    Missing CIK numbers are always filled in (i.e. the ingestor's
    find_missing_ciks is effectively always True here, and is not
    exposed as a parameter of this function): unlike Company, whose
    cik field may be None, CompanyIdentifier's cik field is required,
    so every returned identifier is guaranteed to have a real CIK.

    Args:
        path: The path to the JSON cache file used by
            Sp1500CompanyIngestor. If None (the default), caching is
            disabled entirely: the company list is freshly fetched
            from Wikipedia on every call, and nothing is loaded from
            or saved to disk.

    Returns:
        A list of CompanyIdentifier instances, one per S&P 1500
        company.

    Example:
        ```pycon
        >>> from glyphik.data.sp1500 import get_sp1500_company_identifiers
        >>> identifiers = get_sp1500_company_identifiers()  # doctest: +SKIP

        ```
    """
    return ProcessorIngestor(
        source=Sp1500CompanyIngestor(path=path),
        processor=SequenceProcessor(Sp1500CompanyToIdentifierProcessor()),
    ).ingest()
