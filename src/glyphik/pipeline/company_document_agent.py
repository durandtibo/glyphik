r"""Provide a batch pipeline for running an agent over per-company
document sets retrieved from a document store."""

from __future__ import annotations

__all__ = ["CompanyDocumentAgentPipeline"]

import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import MultilineDisplayMixin
from coola.utils.batching import batchify
from coola.utils.format import str_time_human
from zenpyre.documents import sort_by_metadata
from zenpyre.utils.rich import make_progressbar, print_documents_metadata
from zenpyre.utils.token_usage import log_token_usage

from glyphik.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.vectorstores import BaseDocumentStore

    from glyphik.data.sec import CompanyIdentifier

T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)


class CompanyDocumentAgentPipeline(BasePipeline[T], MultilineDisplayMixin):
    """Run an agent over the documents associated with each company.

    For every company, this pipeline retrieves the matching documents
    from ``document_store``, sorts them chronologically by filing
    date, and passes them (together with the company) to ``agent``.
    Companies can be processed one at a time or in fixed-size batches
    via the ``Runnable.batch`` interface, depending on ``batch_size``.

    Args:
        companies: The sequence of companies to process. Each company
            results in one call to ``agent`` (or one slot in a batched
            call).
        document_store: The store used to look up documents for a
            given company. Must expose a ``filter(cik=...)`` method
            returning an iterable of documents, each carrying a
            ``"filing_date"`` metadata entry.
        agent: The runnable invoked with a dict of the form
            ``{"company": CompanyIdentifier, "documents": list}``.
            Called via ``.invoke`` when ``batch_size == 0``, or via
            ``.batch`` otherwise.
        batch_size: The number of companies to process per batch call
            to ``agent.batch``. A value of ``0`` disables batching
            and processes companies one at a time via ``agent.invoke``.
            Must be non-negative.
        config: A config to use when invoking the ``Runnable``.
        continue_on_error: If ``False`` (the default), an exception
            raised by ``agent`` for any company propagates immediately
            and aborts the whole run. If ``True``, a failing company is
            logged as a warning and skipped instead: processing
            continues with the remaining companies, and the returned
            list simply omits that company's output (so it may be
            shorter than ``companies`` if any failed). Only
            :class:`Exception` subclasses are caught this way --
            :class:`BaseException` types such as ``KeyboardInterrupt``
            always propagate regardless of this setting.
        log_documents_metadata: If ``True``, log each company's
            document metadata (via
            ``zenpyre.utils.rich.print_documents_metadata``) right
            after its documents are retrieved and sorted, before being
            passed to ``agent``. Useful for inspecting which documents
            were selected for each company. Defaults to ``False``.

    Raises:
        ValueError: If ``batch_size`` is negative.

    Example:
        ```pycon
        >>> from glyphik.pipeline import CompanyDocumentAgentPipeline
        >>> from glyphik.data.sec import CompanyIdentifier
        >>> pipeline = CompanyDocumentAgentPipeline(
        ...     companies=[
        ...         CompanyIdentifier.from_ticker("AAPL"),
        ...         CompanyIdentifier.from_ticker("MSFT"),
        ...     ],
        ...     document_store=document_store,
        ...     agent=agent,
        ...     batch_size=8,
        ...     continue_on_error=True,
        ... )
        >>> outputs = pipeline.execute()

        ```
    """

    def __init__(
        self,
        companies: Sequence[CompanyIdentifier],
        document_store: BaseDocumentStore,
        agent: Runnable[dict[str, Any], T],
        batch_size: int = 0,
        config: RunnableConfig | None = None,
        continue_on_error: bool = False,
        log_documents_metadata: bool = False,
    ) -> None:
        if batch_size < 0:
            msg = f"batch_size must be non-negative, got {batch_size}"
            raise ValueError(msg)
        self._companies = companies
        self._document_store = document_store
        self._agent = agent
        self._batch_size = batch_size
        self._config = config
        self._continue_on_error = continue_on_error
        self._log_documents_metadata = log_documents_metadata

    def execute(self) -> list[T]:
        """Run the pipeline over all companies and return the agent
        outputs.

        For each company, the matching documents are retrieved from the
        document store, sorted by filing date, and fed to the agent.
        Companies are processed one at a time if ``batch_size == 0``, or
        in groups of ``batch_size`` via ``agent.batch`` otherwise. Token
        usage is logged for every response (only when nonzero, per
        ``log_token_usage``'s default behavior).

        If ``self._continue_on_error`` is ``True``, a company whose
        agent call raises an exception is logged as a warning and
        skipped rather than aborting the run; see
        :attr:`continue_on_error` on the class docstring for details.

        Returns:
            The list of agent outputs, one per company, in the same
            order as ``self._companies``. If ``continue_on_error`` is
            ``True`` and some companies failed, the list omits their
            outputs and is correspondingly shorter.
        """
        logger.info("Starting company document agent pipeline...")
        t_start = time.perf_counter()

        outputs = self._execute_sequential() if self._batch_size == 0 else self._execute_batch()

        logger.info("Pipeline complete in %s", str_time_human(time.perf_counter() - t_start))
        return outputs

    def _execute_sequential(self) -> list[T]:
        """Process companies one at a time via ``agent.invoke``.

        If ``self._continue_on_error`` is ``True``, a company whose
        ``agent.invoke`` call raises an :class:`Exception` is logged as
        a warning and skipped instead of aborting the run.

        Returns:
            The list of agent outputs, one per company that succeeded,
            in the same order as ``self._companies``.
        """
        outputs: list[T] = []
        with make_progressbar(transient=True) as progress:
            task = progress.add_task("Processing companies...", total=len(self._companies))
            for company in self._companies:
                inp = self._build_agent_input(company)
                try:
                    response = self._agent.invoke(inp, config=self._config)
                except Exception:
                    if not self._continue_on_error:
                        raise
                    logger.warning(
                        "Skipping company %s after agent failure", company, exc_info=True
                    )
                    progress.advance(task)
                    continue
                log_token_usage(response)
                outputs.append(response)
                progress.advance(task)
        return outputs

    def _execute_batch(self) -> list[T]:
        """Process companies in groups of ``self._batch_size`` via
        ``agent.batch``.

        If ``self._continue_on_error`` is ``True``, ``agent.batch`` is
        called with ``return_exceptions=True`` so a failing company
        does not abort the rest of the batch: its exception is returned
        in place of a normal output, logged as a warning, and excluded
        from the returned list. Otherwise, any exception raised while
        processing a batch propagates immediately.

        Returns:
            The list of agent outputs, one per company that succeeded,
            in the same order as ``self._companies``.
        """
        outputs: list[T] = []
        with make_progressbar(transient=True) as progress:
            task = progress.add_task("Processing companies...", total=len(self._companies))
            for companies in batchify(self._companies, size=self._batch_size):
                inputs = [self._build_agent_input(company) for company in companies]
                responses = self._agent.batch(
                    inputs, config=self._config, return_exceptions=self._continue_on_error
                )
                successes = []
                for company, response in zip(companies, responses, strict=True):
                    if self._continue_on_error and isinstance(response, BaseException):
                        logger.warning(
                            "Skipping company %s after agent failure: %s", company, response
                        )
                        continue
                    successes.append(response)
                log_token_usage(successes)
                outputs.extend(successes)
                progress.advance(task, advance=len(companies))
        return outputs

    def _get_repr_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments used for the ``repr``/``str``
        display.

        Returns:
            A dict mapping constructor argument names to their current
            values, used by :class:`MultilineDisplayMixin` to render
            this pipeline.
        """
        return {
            "companies": self._companies,
            "document_store": self._document_store,
            "agent": self._agent,
            "batch_size": self._batch_size,
            "config": self._config,
            "continue_on_error": self._continue_on_error,
            "log_documents_metadata": self._log_documents_metadata,
        }

    def _build_agent_input(self, company: CompanyIdentifier) -> dict[str, Any]:
        """Retrieve and sort the documents for a single company.

        If ``self._log_documents_metadata`` is ``True``, the sorted
        documents' metadata is logged via
        ``zenpyre.utils.rich.print_documents_metadata`` before being
        returned.

        Args:
            company: The company to look up in the document store,
                identified via its ``cik`` attribute.

        Returns:
            A dict of the form ``{"company": company, "documents": documents}``
            where ``documents`` is the list of documents matching
            ``company.cik``, sorted in ascending order by their
            ``"filing_date"`` metadata.
        """
        documents = sort_by_metadata(
            self._document_store.filter(cik=company.cik), metadata_key="filing_date"
        )
        if self._log_documents_metadata:
            print_documents_metadata(documents)
        return {"company": company, "documents": documents}
