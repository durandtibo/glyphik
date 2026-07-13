r"""Provide a concrete factory that builds a
:class:`~glyphik.agents.RecentDocumentsAgent`."""

from __future__ import annotations

__all__ = ["RecentDocumentsAgentFactory"]

from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from zenpyre.agents.factory.base import BaseAgentFactory
from zenpyre.utils.resolve import resolve_object

from glyphik.agents.recent_documents import RecentDocumentsAgent

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from zenpyre.utils.config import BaseConfig


class RecentDocumentsAgentFactory(BaseAgentFactory, MultilineDisplayMixin):
    """A concrete agent factory that builds a
    :class:`~glyphik.agents.RecentDocumentsAgent`.

    Each call to :meth:`make_agent` builds a fresh inner agent via
    ``agent_factory`` and wraps it in a
    :class:`~glyphik.agents.RecentDocumentsAgent` so that only the
    ``max_documents`` most recent documents are formatted and
    forwarded to the inner agent.

    Args:
        agent_factory: The factory used to build the wrapped
            agent that receives the formatted documents.
        max_documents: The maximum number of most recent documents to
            keep. Must be a positive integer.
        include_metadata: If ``True``, include each document's
            metadata in the formatted string. Defaults to ``False``.
        document_format: The format used to render the documents (e.g.
            ``"xml"``, ``"markdown"``). Defaults to ``"xml"``.
        log_documents_metadata: If ``True``, log the document metadata
            for the selected documents. Defaults to ``False``.

    Example:
        ```pycon
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from zenpyre.agents import AgentChatModel
        >>> from zenpyre.agents.factory import AgentFactory
        >>> from glyphik.agents.factory import RecentDocumentsAgentFactory
        >>> agent = AgentChatModel(model=FakeListChatModel(responses=["hello"]))
        >>> factory = RecentDocumentsAgentFactory(
        ...     agent_factory=AgentFactory(agent),
        ...     max_documents=3,
        ... )
        >>> agent = factory.make_agent()

        ```
    """

    def __init__(
        self,
        agent_factory: BaseAgentFactory | dict[str, Any] | BaseConfig,
        max_documents: int = 1,
        include_metadata: bool = False,
        document_format: str = "xml",
        log_documents_metadata: bool = False,
    ) -> None:
        self._agent_factory = resolve_object(agent_factory, cls=BaseAgentFactory)
        self._max_documents = max_documents
        self._include_metadata = include_metadata
        self._document_format = document_format
        self._log_documents_metadata = log_documents_metadata

    def make_agent(self) -> Runnable[dict[str, Any], Any]:
        return RecentDocumentsAgent(
            agent=self._agent_factory.make_agent(),
            max_documents=self._max_documents,
            include_metadata=self._include_metadata,
            document_format=self._document_format,
            log_documents_metadata=self._log_documents_metadata,
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "agent_factory": self._agent_factory,
            "max_documents": self._max_documents,
            "include_metadata": self._include_metadata,
            "document_format": self._document_format,
            "log_documents_metadata": self._log_documents_metadata,
        }
