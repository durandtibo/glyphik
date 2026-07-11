r"""Contain functions for downloading/preparing SP1500-company data."""

from __future__ import annotations

__all__ = [
    "Company",
    "fetch_companies",
    "fill_missing_ciks",
    "get_company_identifiers",
    "load_or_fetch_companies",
    "load_or_fetch_company_filings",
    "load_or_fetch_filings",
]

from glyphik.data.sp1500.cik import fill_missing_ciks
from glyphik.data.sp1500.company import Company
from glyphik.data.sp1500.company_identifier import get_company_identifiers
from glyphik.data.sp1500.fetch import (
    fetch_companies,
    load_or_fetch_companies,
)
from glyphik.data.sp1500.filing import (
    load_or_fetch_company_filings,
    load_or_fetch_filings,
)
