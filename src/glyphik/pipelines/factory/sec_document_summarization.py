r"""Provide a concrete factory that builds a company document
summarization pipeline."""

from __future__ import annotations

__all__ = ["SecDocumentSummarizationPipelineFactory"]

from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.path import sanitize_path

from glyphik.document_stores.factory import SecFilingDocumentStoreFactory
from glyphik.pipelines import CompanyDocumentAgentPipeline
from glyphik.pipelines.factory.base import BasePipelineFactory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from zenpyre.agents.factory import BaseAgentFactory

    from glyphik.data.sec import CompanyIdentifier
    from glyphik.pipelines.base import BasePipeline


class SecDocumentSummarizationPipelineFactory(BasePipelineFactory[Any], MultilineDisplayMixin):
    """A concrete BasePipeline factory that builds a
        :class:`~glyphik.pipelines.CompanyDocumentAgentPipeline` for
        summarizing SEC filing documents for a set of companies.

        Each call to :meth:`make_pipeline` builds a fresh agent via
        ``agent_factory``, opens a read-only
        :class:`~zenpyre.document_stores.base.BaseDocumentStore` backed
        by the SEC filing DuckDB file located under ``base_dir`` (via
        :class:`~glyphik.document_stores.factory.SecFilingDocumentStoreFactory`),
        and wires both together with ``companies`` into a
        :class:`~glyphik.pipelines.CompanyDocumentAgentPipeline`.

    Args:
            companies: The companies whose documents the pipeline should
                summarize.
            agent_factory: The factory used to build the agent that
                performs summarization.
            base_dir: The base directory containing the SEC filing
                document store to read from. Accepts a
                :class:`~pathlib.Path` or a :class:`str`, which is
                sanitized via :func:`~coola.utils.path.sanitize_path`.

    Example:
    ```pycon
    >>> from pathlib import Path
    >>> from glyphik.data.sec import CompanyIdentifier
    >>> from glyphik.pipelines.factory import (
    ...     SecDocumentSummarizationPipelineFactory,
    ... )
    >>> factory = SecDocumentSummarizationPipelineFactory(  # doctest: +SKIP
    ...     companies=[CompanyIdentifier(cik=320193, ticker="AAPL")],
    ...     agent_factory=...,
    ...     base_dir=Path("/tmp/my_app"),
    ... )
    >>> pipeline = factory.make_pipeline()  # doctest: +SKIP

    ```
    """

    def __init__(
        self,
        companies: Sequence[CompanyIdentifier],
        agent_factory: BaseAgentFactory,
        base_dir: Path | str,
    ) -> None:
        self._companies = list(companies)
        self._agent_factory = agent_factory
        self._base_dir = sanitize_path(base_dir)

    def make_pipeline(self) -> BasePipeline[Any]:
        agent = self._agent_factory.make_agent()

        document_store = SecFilingDocumentStoreFactory(
            base_dir=self._base_dir, read_only=True
        ).make_document_store()

        return CompanyDocumentAgentPipeline(
            companies=self._companies,
            document_store=document_store,
            agent=agent,
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "companies": self._companies,
            "agent_factory": self._agent_factory,
            "base_dir": self._base_dir,
        }
