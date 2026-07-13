r"""Contain an agent to analyze documents sorted by most recent
documents."""

from __future__ import annotations

__all__ = ["RecentDocumentsAgent"]

from typing import Any, TypeVar

from coola.display import MultilineDisplayMixin
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from zenpyre.documents import format_documents
from zenpyre.utils.rich import print_documents_metadata

T = TypeVar("T")

_VALID_OUTPUT_FORMATS = frozenset({"xml", "markdown"})
"""Formats accepted by ``format_documents``.

Update if ``zenpyre.documents`` adds or removes supported formats.
"""


class RecentDocumentsAgent(Runnable[dict[str, Any], T], MultilineDisplayMixin):
    """Wrap an agent to analyze only the ``max_documents`` most recent documents.

    Given an input dict containing a list of documents, this agent
    keeps only the ``max_documents`` most recent ones, formats them
    as a string (XML by default), wraps that string in a
    ``HumanMessage``, and forwards it to ``agent`` as
    ``{"messages": [HumanMessage(...)]}``.

    Assumption:
        The input documents are expected to already be sorted in
        ascending chronological order (oldest first). This agent does
        not sort them; it simply takes the last ``max_documents``
        entries.

    Args:
        agent: The wrapped runnable that receives the formatted
            documents as a ``{"messages": [HumanMessage(...)]}`` dict
            input (e.g. a chat model or a LangGraph-style agent).
        max_documents: The maximum number of most recent documents to
            keep. Must be a positive integer.
        include_metadata: If ``True``, include each document's
            metadata in the formatted string. Defaults to ``False``.
        document_format: The format used to render the documents (e.g.
            ``"xml"``, ``"markdown"``). Defaults to ``"xml"``.
        log_documents_metadata: If ``True``, log the document metadata
            for the selected documents. Defaults to ``False``.

    Raises:
        ValueError: If ``max_documents`` is not a positive integer, or
            if ``document_format`` is not a supported format.

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
        ...     agent=RunnableLambda(lambda inp: len(inp["messages"][0].content)),
        ...     max_documents=3,
        ... )
        >>> output = agent.invoke({"documents": docs})
        >>> output
        109

        ```
    """

    def __init__(
        self,
        agent: Runnable[dict[str, Any], T],
        max_documents: int = 1,
        include_metadata: bool = False,
        document_format: str = "xml",
        log_documents_metadata: bool = False,
    ) -> None:
        if max_documents < 1:
            msg = f"max_documents must be a positive integer, got {max_documents}"
            raise ValueError(msg)
        if document_format not in _VALID_OUTPUT_FORMATS:
            msg = (
                f"document_format must be one of {sorted(_VALID_OUTPUT_FORMATS)}, "
                f"got {document_format!r}"
            )
            raise ValueError(msg)
        self._agent = agent
        self._max_documents = max_documents
        self._include_metadata = include_metadata
        self._document_format = document_format
        self._log_documents_metadata = log_documents_metadata

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
            The output of ``agent.invoke`` called with a
            ``{"messages": [HumanMessage(...)]}`` dict containing the
            formatted string of the ``max_documents`` most recent
            documents.

        Raises:
            KeyError: If ``input`` does not contain a ``"documents"``
                key.
        """
        inner_input = self._to_inner_input(self._get_documents(input))
        return self._agent.invoke(inner_input, config, **kwargs)

    async def ainvoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        """Async counterpart of :meth:`invoke`.

        Args:
            input: A dict containing a ``"documents"`` key holding the
                list of documents to analyze, assumed to already be
                sorted in ascending chronological order.
            config: Optional runnable configuration forwarded to the
                inner agent.
            **kwargs: Additional keyword arguments forwarded to the
                inner agent's ``ainvoke``.

        Returns:
            The output of ``agent.ainvoke`` called with a
            ``{"messages": [HumanMessage(...)]}`` dict containing the
            formatted string of the ``max_documents`` most recent
            documents.

        Raises:
            KeyError: If ``input`` does not contain a ``"documents"``
                key.
        """
        inner_input = self._to_inner_input(self._get_documents(input))
        return await self._agent.ainvoke(inner_input, config, **kwargs)

    @staticmethod
    def _get_documents(input: dict[str, Any]) -> list[Any]:  # noqa: A002
        """Extract the ``"documents"`` list from the input dict.

        Args:
            input: The runnable input dict.

        Returns:
            The list of documents.

        Raises:
            KeyError: If ``input`` does not contain a ``"documents"``
                key.
        """
        try:
            return input["documents"]
        except KeyError as e:
            msg = "input dict must contain a 'documents' key"
            raise KeyError(msg) from e

    def _to_inner_input(self, documents: list[Any]) -> dict[str, Any]:
        """Format the ``max_documents`` most recent documents and wrap
        them into the inner agent's expected input shape.

        Args:
            documents: The documents to format, assumed to already be
                sorted in ascending chronological order.

        Returns:
            A dict of the form ``{"messages": [HumanMessage(...)]}``,
            where the message content is the formatted string of the
            ``max_documents`` most recent documents.
        """
        return {"messages": [HumanMessage(content=self._format_last_documents(documents))]}

    def _format_last_documents(self, documents: list[Any]) -> str:
        """Format the ``max_documents`` most recent documents.

        Args:
            documents: The documents to format, assumed to already be
                sorted in ascending chronological order.

        Returns:
            A formatted string of the ``max_documents`` most recent
            documents (or fewer, if ``documents`` has fewer than
            ``max_documents`` entries). If ``documents`` is empty,
            returns whatever ``format_documents([])`` produces.
        """
        docs = documents[-self._max_documents :]
        if self._log_documents_metadata:
            print_documents_metadata(docs)
        return format_documents(
            docs,
            include_metadata=self._include_metadata,
            output_format=self._document_format,
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
            "agent": self._agent,
            "max_documents": self._max_documents,
            "include_metadata": self._include_metadata,
            "document_format": self._document_format,
            "log_documents_metadata": self._log_documents_metadata,
        }
