r"""Contain data processors."""

from __future__ import annotations

__all__ = [
    "EdgarCompanyToIdentifierProcessor",
    "RedownloadSecFilingProcessor",
    "SecFilingRecordToDocumentProcessor",
    "Sp1500CompanyToIdentifierProcessor",
]

from glyphik.data_processors.edgar_company_to_identifier import (
    EdgarCompanyToIdentifierProcessor,
)
from glyphik.data_processors.redownload_sec_filing import RedownloadSecFilingProcessor
from glyphik.data_processors.sec_record_to_document import (
    SecFilingRecordToDocumentProcessor,
)
from glyphik.data_processors.sp1500_company_to_identifier import (
    Sp1500CompanyToIdentifierProcessor,
)
