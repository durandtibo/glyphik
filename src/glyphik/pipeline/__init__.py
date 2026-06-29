r"""Contain pipelines."""

from __future__ import annotations

__all__ = ["BasePipeline", "BatchDocumentIndexingPipeline", "DocumentIndexingPipeline"]

from glyphik.pipeline.base import BasePipeline
from glyphik.pipeline.batch_document_indexing import BatchDocumentIndexingPipeline
from glyphik.pipeline.document_indexing import DocumentIndexingPipeline
