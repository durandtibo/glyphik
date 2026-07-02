r"""Contain code to prepare SEC data."""

from __future__ import annotations

__all__ = [
    "SecFilingRecord",
    "SecForm",
    "extract_filing_content",
    "fetch_cik_from_ticker",
    "fetch_filings",
    "fetch_form_filings",
    "fetch_ticker_from_cik",
    "normalize_content_format",
]


from glyphik.data.sec.cik import fetch_ticker_from_cik
from glyphik.data.sec.filing import fetch_filings, fetch_form_filings
from glyphik.data.sec.filing_content import (
    extract_filing_content,
    normalize_content_format,
)
from glyphik.data.sec.form import SecForm
from glyphik.data.sec.record import SecFilingRecord
from glyphik.data.sec.ticker import fetch_cik_from_ticker
