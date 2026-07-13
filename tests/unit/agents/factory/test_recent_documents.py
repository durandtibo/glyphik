from __future__ import annotations

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from zenpyre.agents import AgentChatModel
from zenpyre.agents.factory import AgentFactory, BaseAgentFactory

from glyphik.agents.factory.recent_documents import RecentDocumentsAgentFactory
from glyphik.agents.recent_documents import RecentDocumentsAgent


def _agent_factory() -> AgentFactory:
    return AgentFactory(AgentChatModel(model=FakeListChatModel(responses=["hello"])))


######################################################
#     Tests for RecentDocumentsAgentFactory          #
######################################################


def test_recent_documents_agent_factory_stores_agent_factory() -> None:
    agent_factory = _agent_factory()
    factory = RecentDocumentsAgentFactory(agent_factory=agent_factory)
    assert factory._agent_factory is agent_factory


def test_recent_documents_agent_factory_default_max_documents() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    assert factory._max_documents == 1


def test_recent_documents_agent_factory_stores_max_documents() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory(), max_documents=3)
    assert factory._max_documents == 3


def test_recent_documents_agent_factory_default_include_metadata() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    assert factory._include_metadata is False


def test_recent_documents_agent_factory_default_document_format() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    assert factory._document_format == "xml"


def test_recent_documents_agent_factory_default_log_documents_metadata() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    assert factory._log_documents_metadata is False


def test_recent_documents_agent_factory_repr_contains_class_name() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    assert "RecentDocumentsAgentFactory" in repr(factory)


def test_recent_documents_agent_factory_str_contains_class_name() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    assert "RecentDocumentsAgentFactory" in str(factory)


def test_recent_documents_agent_factory_make_agent_returns_recent_documents_agent() -> None:
    factory = RecentDocumentsAgentFactory(
        agent_factory=_agent_factory(),
        max_documents=2,
        include_metadata=True,
        document_format="markdown",
        log_documents_metadata=True,
    )
    agent = factory.make_agent()
    assert isinstance(agent, RecentDocumentsAgent)
    assert agent._max_documents == 2
    assert agent._include_metadata is True
    assert agent._document_format == "markdown"
    assert agent._log_documents_metadata is True


def test_recent_documents_agent_factory_make_agent_uses_agent_factory() -> None:
    inner_agent = RunnableLambda(lambda inp: inp["messages"][0].content)

    class _StubAgentFactory(BaseAgentFactory):
        def make_agent(self) -> RunnableLambda:
            return inner_agent

    factory = RecentDocumentsAgentFactory(agent_factory=_StubAgentFactory())
    agent = factory.make_agent()
    assert agent._agent is inner_agent


def test_recent_documents_agent_factory_make_agent_builds_fresh_agent_each_call() -> None:
    factory = RecentDocumentsAgentFactory(agent_factory=_agent_factory())
    agent1 = factory.make_agent()
    agent2 = factory.make_agent()
    assert agent1 is not agent2


def test_recent_documents_agent_factory_resolves_agent_factory_from_dict() -> None:
    factory = RecentDocumentsAgentFactory(
        agent_factory={
            "_target_": "zenpyre.agents.factory.AgentFactory",
            "agent": AgentChatModel(model=FakeListChatModel(responses=["hello"])),
        }
    )
    assert isinstance(factory._agent_factory, AgentFactory)


def test_recent_documents_agent_factory_invalid_agent_factory_raises() -> None:
    with pytest.raises(TypeError, match="Received object is not a BaseAgentFactory instance"):
        RecentDocumentsAgentFactory(agent_factory="not an agent factory")
