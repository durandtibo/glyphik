r"""Contain an agent to analyze documents sorted by most recent
documents."""

from __future__ import annotations

__all__ = ["RecentDocumentsAgent"]

from typing import Any, TypeVar

from coola.display import MultilineDisplayMixin
from langchain_core.runnables import Runnable, RunnableConfig
from zenpyre.documents import format_documents_as_xml

T = TypeVar("T")


class RecentDocumentsAgent(Runnable[dict[str, Any], T], MultilineDisplayMixin):
    """Wrap an agent to analyze only the ``max_documents`` most recent documents.

    Given an input dict containing a list of documents, this agent
    keeps only the ``max_documents`` most recent ones, formats them
    as an XML string, and forwards that string to ``inner_agent``.

    Assumption:
        The input documents are expected to already be sorted in
        ascending chronological order (oldest first). This agent does
        not sort them; it simply takes the last ``max_documents``
        entries.

    Args:
        inner_agent: The wrapped runnable that receives the formatted
            documents as a single string input.
        max_documents: The maximum number of most recent documents to
            keep. Must be a positive integer.
        include_metadata: If ``True``, include each document's
            metadata in the formatted XML string. Defaults to
            ``False``.

    Raises:
        ValueError: If ``max_documents`` is not a positive integer.

    Example:
        ```pycon
        >>> from langchain_core.documents import Document
        >>> from langchain_core.runnables import RunnableLambda
        >>> from glyphik.agents import RecentDocumentsAgent
        >>> docs = [
        ...     Document(
        ...         page_content="The cat sat on the mat.",
        ...         metadata={"source": "story.txt", "author": "Alice"},
        ...     ),
        ...     Document(
        ...         page_content="The dog chased the ball.",
        ...         metadata={"source": "story.txt", "author": "Bob"},
        ...     ),
        ... ]
        >>> agent = RecentDocumentsAgent(
        ...     inner_agent=RunnableLambda(lambda inp: len(inp)), max_documents=3
        ... )
        >>> output = agent.invoke({"documents": docs})
        >>> output
        109

        ```
    """

    def __init__(
        self,
        inner_agent: Runnable[str, T],
        max_documents: int = 1,
        include_metadata: bool = False,
    ) -> None:
        if max_documents < 1:
            msg = f"max_documents must be a positive integer, got {max_documents}"
            raise ValueError(msg)
        self._inner_agent = inner_agent
        self._max_documents = max_documents
        self._include_metadata = include_metadata

    def invoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Format the ``max_documents`` most recent documents and invoke
        the inner agent.

        Args:
            input: A dict containing a ``"documents"`` key holding the
                list of documents to analyze, assumed to already be
                sorted in ascending chronological order.
            config: Optional runnable configuration forwarded to the
                inner agent.
            **kwargs: Additional keyword arguments forwarded to the
                inner agent's ``invoke``.

        Returns:
            The output of ``inner_agent.invoke`` called with the
            formatted string of the ``max_documents`` most recent
            documents.
        """
        last_documents = self._format_last_documents(input["documents"])
        return self._inner_agent.invoke(last_documents, config, **kwargs)

    def batch(
        self,
        inputs: list[dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        **kwargs: Any,
    ) -> list[T]:
        """Format the ``max_documents`` most recent documents for each
        input and batch-invoke the inner agent.

        Args:
            inputs: A list of dicts, each containing a ``"documents"``
                key holding the list of documents to analyze, assumed
                to already be sorted in ascending chronological order.
            config: Optional runnable configuration(s) forwarded to
                the inner agent.
            **kwargs: Additional keyword arguments forwarded to the
                inner agent's ``batch``.

        Returns:
            The list of outputs from ``inner_agent.batch``, in the
            same order as ``inputs``.
        """
        formatted_inputs = [self._format_last_documents(inp["documents"]) for inp in inputs]
        return self._inner_agent.batch(formatted_inputs, config, **kwargs)

    def _format_last_documents(self, documents: list[Any]) -> str:
        """Format the ``max_documents`` most recent documents as XML.

        Args:
            documents: The documents to format, assumed to already be
                sorted in ascending chronological order.

        Returns:
            An XML-formatted string of the ``max_documents`` most
            recent documents (or fewer, if ``documents`` has fewer
            than ``max_documents`` entries).
        """
        return format_documents_as_xml(
            documents[-self._max_documents :], include_metadata=self._include_metadata
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments used for the ``repr``/``str``
        display.

        Returns:
            A dict mapping constructor argument names to their current
            values, used by :class:`MultilineDisplayMixin` to render
            this agent.
        """
        return {
            "inner_agent": self._inner_agent,
            "max_documents": self._max_documents,
            "include_metadata": self._include_metadata,
        }
