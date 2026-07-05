r"""Provide a batch pipeline for running an agent over per-ticker
document sets retrieved from a document store."""

from __future__ import annotations

__all__ = ["TickerDocumentAgentPipeline"]

import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import MultilineDisplayMixin
from coola.utils.batching import batchify
from coola.utils.format import str_time_human
from zenpyre.documents import sort_by_metadata
from zenpyre.utils.rich import make_progressbar
from zenpyre.utils.token_usage import log_token_usage

from glyphik.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.runnables import Runnable
    from langchain_core.vectorstores import BaseDocumentStore

T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)


class TickerDocumentAgentPipeline(BasePipeline[T], MultilineDisplayMixin):
    """Run an agent over the documents associated with each ticker.

    For every ticker, this pipeline retrieves the matching documents
    from ``document_store``, sorts them chronologically by filing
    date, and passes them (together with the ticker) to ``agent``.
    Tickers can be processed one at a time or in fixed-size batches
    via the ``Runnable.batch`` interface, depending on ``batch_size``.

    Args:
        tickers: The sequence of ticker symbols to process. Each
            ticker results in one call to ``agent`` (or one slot in
            a batched call).
        document_store: The store used to look up documents for a
            given ticker. Must expose a ``filter(ticker=...)`` method
            returning an iterable of documents, each carrying a
            ``"filing_date"`` metadata entry.
        agent: The runnable invoked with a dict of the form
            ``{"ticker": str, "documents": list}``. Called via
            ``.invoke`` when ``batch_size == 0``, or via ``.batch``
            otherwise.
        batch_size: The number of tickers to process per batch call
            to ``agent.batch``. A value of ``0`` disables batching
            and processes tickers one at a time via ``agent.invoke``.
            Must be non-negative.

    Raises:
        ValueError: If ``batch_size`` is negative.

    Example:
        ```pycon
        >>> pipeline = TickerDocumentAgentPipeline(
        ...     tickers=["AAPL", "MSFT"],
        ...     document_store=document_store,
        ...     agent=agent,
        ...     batch_size=8,
        ... )
        >>> outputs = pipeline.execute()

        ```
    """

    def __init__(
        self,
        tickers: Sequence[str],
        document_store: BaseDocumentStore,
        agent: Runnable[dict[str, Any], T],
        batch_size: int = 0,
    ) -> None:
        if batch_size < 0:
            msg = f"batch_size must be non-negative, got {batch_size}"
            raise ValueError(msg)
        self._tickers = tickers
        self._document_store = document_store
        self._agent = agent
        self._batch_size = batch_size

    def execute(self) -> list[T]:
        """Run the pipeline over all tickers and return the agent
        outputs.

        For each ticker, the matching documents are retrieved from the
        document store, sorted by filing date, and fed to the agent.
        Tickers are processed one at a time if ``batch_size == 0``, or
        in groups of ``batch_size`` via ``agent.batch`` otherwise. Token
        usage is logged for every response.

        Returns:
            The list of agent outputs, one per ticker, in the same
            order as ``self._tickers``.
        """
        logger.info("Starting ticker document agent pipeline...")
        t_start = time.perf_counter()

        outputs = self._execute_sequential() if self._batch_size == 0 else self._execute_batch()

        logger.info("Pipeline complete in %s", str_time_human(time.perf_counter() - t_start))
        return outputs

    def _execute_sequential(self) -> list[T]:
        """Process tickers one at a time via ``agent.invoke``.

        Returns:
            The list of agent outputs, one per ticker, in the same
            order as ``self._tickers``.
        """
        outputs: list[T] = []
        with make_progressbar(transient=True) as progress:
            task = progress.add_task("Processing tickers...", total=len(self._tickers))
            for ticker in self._tickers:
                inp = self._build_agent_input(ticker)
                response = self._agent.invoke(inp)
                log_token_usage(response)
                outputs.append(response)
                progress.advance(task)
        return outputs

    def _execute_batch(self) -> list[T]:
        """Process tickers in groups of ``self._batch_size`` via
        ``agent.batch``.

        Returns:
            The list of agent outputs, one per ticker, in the same
            order as ``self._tickers``.
        """
        outputs: list[T] = []
        with make_progressbar(transient=True) as progress:
            task = progress.add_task("Processing tickers...", total=len(self._tickers))
            for tickers in batchify(self._tickers, size=self._batch_size):
                inputs = [self._build_agent_input(ticker) for ticker in tickers]
                responses = self._agent.batch(inputs)
                log_token_usage(responses)
                outputs.extend(responses)
                progress.advance(task, advance=len(tickers))
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
            "tickers": self._tickers,
            "document_store": self._document_store,
            "agent": self._agent,
            "batch_size": self._batch_size,
        }

    def _build_agent_input(self, ticker: str) -> dict[str, Any]:
        """Retrieve and sort the documents for a single ticker.

        Args:
            ticker: The ticker symbol to look up in the document store.

        Returns:
            A dict of the form ``{"ticker": ticker, "documents": documents}``
            where ``documents`` is the list of documents matching
            ``ticker``, sorted in ascending order by their
            ``"filing_date"`` metadata.
        """
        documents = sort_by_metadata(
            self._document_store.filter(ticker=ticker), metadata_key="filing_date"
        )
        return {"ticker": ticker, "documents": documents}
