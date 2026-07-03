r"""Contain ingestors."""

from __future__ import annotations

__all__ = [
    "DocumentStoreIndexingIngestor",
    "SecFilingDocumentStoreIngestor",
    "Sp1500CompanyIngestor",
    "Sp1500FilingIngestor",
]

from glyphik.ingestors.document_store_indexing import DocumentStoreIndexingIngestor
from glyphik.ingestors.sec_filing_document_store import SecFilingDocumentStoreIngestor
from glyphik.ingestors.sp1500_company import Sp1500CompanyIngestor
from glyphik.ingestors.sp1500_filing import Sp1500FilingIngestor
