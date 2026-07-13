r"""Provide a concrete factory that builds a company document
summarization pipeline."""

from __future__ import annotations

__all__ = ["SecDocumentSummarizationPipelineFactory"]

from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.utils.path import sanitize_path
from zenpyre.agents.factory import BaseAgentFactory
from zenpyre.utils.resolve import resolve_object

from glyphik.data.sp1500 import get_company_identifiers
from glyphik.document_stores.factory import SecFilingDocumentStoreFactory
from glyphik.pipelines import CompanyDocumentAgentPipeline
from glyphik.pipelines.factory.base import BasePipelineFactory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Self

    from langchain_core.runnables import RunnableConfig
    from zenpyre.utils.config import BaseConfig

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
        batch_size: The number of companies to process per batch call
            to the agent, forwarded to
            :class:`~glyphik.pipelines.CompanyDocumentAgentPipeline`.
            A value of ``0`` disables batching and processes
            companies one at a time. Must be non-negative.
        config: A config to use when invoking the agent, forwarded to
            :class:`~glyphik.pipelines.CompanyDocumentAgentPipeline`.
        continue_on_error: If ``False`` (the default), an exception
            raised by the agent for any company propagates
            immediately and aborts the whole run. If ``True``, a
            failing company is logged as a warning and skipped
            instead. Forwarded to
            :class:`~glyphik.pipelines.CompanyDocumentAgentPipeline`.
        log_documents_metadata: If ``True``, log each company's
            document metadata before it is passed to the agent.
            Forwarded to
            :class:`~glyphik.pipelines.CompanyDocumentAgentPipeline`.
            Defaults to ``False``.

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
        agent_factory: BaseAgentFactory | dict[str, Any] | BaseConfig,
        base_dir: Path | str,
        batch_size: int = 0,
        config: RunnableConfig | None = None,
        continue_on_error: bool = False,
        log_documents_metadata: bool = False,
    ) -> None:
        self._companies = list(companies)
        self._agent_factory = resolve_object(agent_factory, cls=BaseAgentFactory)
        self._base_dir = sanitize_path(base_dir)
        self._batch_size = batch_size
        self._config = config
        self._continue_on_error = continue_on_error
        self._log_documents_metadata = log_documents_metadata

    def make_pipeline(self) -> BasePipeline[Any]:
        agent = self._agent_factory.make_agent()

        document_store = SecFilingDocumentStoreFactory(
            base_dir=self._base_dir, read_only=True
        ).make_document_store()

        return CompanyDocumentAgentPipeline(
            companies=self._companies,
            document_store=document_store,
            agent=agent,
            batch_size=self._batch_size,
            config=self._config,
            continue_on_error=self._continue_on_error,
            log_documents_metadata=self._log_documents_metadata,
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "companies": self._companies,
            "agent_factory": self._agent_factory,
            "base_dir": self._base_dir,
            "batch_size": self._batch_size,
            "config": self._config,
            "continue_on_error": self._continue_on_error,
            "log_documents_metadata": self._log_documents_metadata,
        }

    @classmethod
    def from_sp1500(
        cls,
        agent_factory: BaseAgentFactory | dict[str, Any] | BaseConfig,
        base_dir: Path | str,
        max_companies: int | None = None,
        batch_size: int = 0,
        config: RunnableConfig | None = None,
        continue_on_error: bool = False,
        log_documents_metadata: bool = False,
    ) -> Self:
        """Create a factory that summarizes SEC filings for S&P 1500
        companies.

        The set of companies is resolved from an SP1500 company
        identifier file located at
        ``base_dir / "SP1500" / "company_identifier.json"`` (via
        :func:`~glyphik.data.sp1500.get_company_identifiers`),
        optionally truncated to the first ``max_companies`` entries.

        Args:
            agent_factory: The factory used to build the agent that
                performs summarization.
            base_dir: The base directory under which the SP1500
                company identifier file is read, and the SEC filing
                document store is read from. Accepts a
                :class:`~pathlib.Path` or a :class:`str`, which is
                sanitized via :func:`~coola.utils.path.sanitize_path`.
            max_companies: If set, only the first ``max_companies``
                companies from the resolved SP1500 list are
                summarized. If ``None``, all companies are
                summarized.
            batch_size: The number of companies to process per batch
                call to the agent. Forwarded to the constructor. A
                value of ``0`` disables batching. Must be
                non-negative.
            config: A config to use when invoking the agent.
                Forwarded to the constructor.
            continue_on_error: If ``True``, a failing company is
                logged as a warning and skipped instead of aborting
                the run. Forwarded to the constructor.
            log_documents_metadata: If ``True``, log each company's
                document metadata before it is passed to the agent.
                Forwarded to the constructor. Defaults to ``False``.

        Returns:
            A configured :class:`SecDocumentSummarizationPipelineFactory`
            for S&P 1500 companies.

        Example:
            ```pycon
            >>> from pathlib import Path
            >>> from glyphik.pipelines.factory import (
            ...     SecDocumentSummarizationPipelineFactory,
            ... )
            >>> factory = SecDocumentSummarizationPipelineFactory.from_sp1500(  # doctest: +SKIP
            ...     agent_factory=...,
            ...     base_dir=Path("/tmp/my_app"),
            ...     max_companies=50,
            ... )
            >>> pipeline = factory.make_pipeline()  # doctest: +SKIP

            ```
        """
        base_dir = sanitize_path(base_dir)
        companies = get_company_identifiers(base_dir / "SP1500" / "company_identifier.json")
        if max_companies is not None:
            companies = companies[:max_companies]
        return cls(
            agent_factory=agent_factory,
            companies=companies,
            base_dir=base_dir,
            batch_size=batch_size,
            config=config,
            continue_on_error=continue_on_error,
            log_documents_metadata=log_documents_metadata,
        )
