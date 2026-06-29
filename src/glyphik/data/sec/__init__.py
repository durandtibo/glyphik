r"""Contain code to prepare SEC data."""

from __future__ import annotations

__all__ = [
    "SecFilingRecord",
    "SecForm",
    "fetch_filings",
    "fetch_form_filings",
    "fetch_ticker_from_cik",
]


from glyphik.data.sec.cik import fetch_ticker_from_cik
from glyphik.data.sec.filing import SecFilingRecord, fetch_filings, fetch_form_filings
from glyphik.data.sec.form import SecForm
