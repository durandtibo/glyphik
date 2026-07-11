r"""Contain pipelines."""

from __future__ import annotations

__all__ = [
    "BasePipeline",
    "CompanyDocumentAgentPipeline",
    "DocumentIndexingPipeline",
]

from glyphik.pipeline.base import BasePipeline
from glyphik.pipeline.company_document_agent import CompanyDocumentAgentPipeline
from glyphik.pipeline.document_indexing import DocumentIndexingPipeline
