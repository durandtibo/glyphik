r"""Contain code to prepare SEC data."""

from __future__ import annotations

__all__ = [
    "CompanyIdentifier",
    "SecFilingRecord",
    "SecForm",
    "extract_filing_content",
    "fetch_cik_from_ticker",
    "fetch_filings",
    "fetch_form_filings",
    "fetch_ticker_from_cik",
    "has_valid_sgml",
    "load_or_fetch_company_filings",
    "load_or_fetch_filings",
    "normalize_content_format",
]

from glyphik.data.sec.cache_or_fetch_filings import (
    load_or_fetch_company_filings,
    load_or_fetch_filings,
)
from glyphik.data.sec.cik import fetch_ticker_from_cik
from glyphik.data.sec.company_identifier import CompanyIdentifier
from glyphik.data.sec.fetch_filings import (
    fetch_filings,
    fetch_form_filings,
    has_valid_sgml,
)
from glyphik.data.sec.filing_content import (
    extract_filing_content,
    normalize_content_format,
)
from glyphik.data.sec.form import SecForm
from glyphik.data.sec.record import SecFilingRecord
from glyphik.data.sec.ticker import fetch_cik_from_ticker
