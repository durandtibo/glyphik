r"""Contain pipelines."""

from __future__ import annotations

__all__ = [
    "BasePipeline",
    "CompanyDocumentAgentPipeline",
    "DocumentIndexingPipeline",
]

from glyphik.pipelines.base import BasePipeline
from glyphik.pipelines.company_document_agent import CompanyDocumentAgentPipeline
from glyphik.pipelines.document_indexing import DocumentIndexingPipeline
