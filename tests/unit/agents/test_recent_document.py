from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from glyphik.agents.recent_documents import RecentDocumentsAgent


def _echo_agent() -> RunnableLambda:
    """Return an inner agent that echoes back the content of the single
    HumanMessage it receives."""
    return RunnableLambda(lambda inp: inp["messages"][0].content)


######################################################
#     Tests for RecentDocumentsAgent                 #
######################################################


# --- Constructor ---


def test_recent_documents_agent_stores_inner_agent() -> None:
    inner_agent = _echo_agent()
    agent = RecentDocumentsAgent(inner_agent=inner_agent)
    assert agent._inner_agent is inner_agent


def test_recent_documents_agent_default_max_documents() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    assert agent._max_documents == 1


def test_recent_documents_agent_stores_max_documents() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=3)
    assert agent._max_documents == 3


def test_recent_documents_agent_default_include_metadata() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    assert agent._include_metadata is False


def test_recent_documents_agent_stores_include_metadata() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), include_metadata=True)
    assert agent._include_metadata is True


def test_recent_documents_agent_max_documents_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_documents must be a positive integer"):
        RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=0)


def test_recent_documents_agent_max_documents_negative_raises() -> None:
    with pytest.raises(ValueError, match="max_documents must be a positive integer"):
        RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=-1)


def test_recent_documents_agent_repr_contains_class_name() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    assert "RecentDocumentsAgent" in repr(agent)


def test_recent_documents_agent_str_contains_class_name() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    assert "RecentDocumentsAgent" in str(agent)


# --- invoke ---


def test_recent_documents_agent_invoke_keeps_only_last_max_documents() -> None:
    documents = [
        Document(page_content="oldest"),
        Document(page_content="middle"),
        Document(page_content="newest"),
    ]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=2)
    result = agent.invoke({"documents": documents})
    assert result == (
        '<document id="1">\nmiddle\n</document>\n\n<document id="2">\nnewest\n</document>'
    )


def test_recent_documents_agent_invoke_default_max_documents_keeps_only_last() -> None:
    documents = [Document(page_content="oldest"), Document(page_content="newest")]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    result = agent.invoke({"documents": documents})
    assert result == '<document id="1">\nnewest\n</document>'


def test_recent_documents_agent_invoke_max_documents_larger_than_available() -> None:
    documents = [Document(page_content="only one")]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=5)
    result = agent.invoke({"documents": documents})
    assert result == '<document id="1">\nonly one\n</document>'


def test_recent_documents_agent_invoke_without_metadata_excludes_metadata() -> None:
    documents = [Document(page_content="text", metadata={"source": "a.txt"})]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), include_metadata=False)
    result = agent.invoke({"documents": documents})
    assert result == '<document id="1">\ntext\n</document>'


def test_recent_documents_agent_invoke_with_metadata_includes_metadata() -> None:
    documents = [Document(page_content="text", metadata={"source": "a.txt"})]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), include_metadata=True)
    result = agent.invoke({"documents": documents})
    assert result == '<document id="1">\nsource: a.txt\n\ntext\n</document>'


def test_recent_documents_agent_invoke_passes_messages_dict_to_inner_agent() -> None:
    documents = [Document(page_content="text")]
    inner_agent = RunnableLambda(lambda inp: inp)
    agent = RecentDocumentsAgent(inner_agent=inner_agent)
    result = agent.invoke({"documents": documents})
    assert result == {"messages": [HumanMessage(content='<document id="1">\ntext\n</document>')]}


def test_recent_documents_agent_invoke_returns_inner_agent_output() -> None:
    documents = [Document(page_content="text")]
    inner_agent = RunnableLambda(lambda inp: {"formatted": inp["messages"][0].content})
    agent = RecentDocumentsAgent(inner_agent=inner_agent)
    result = agent.invoke({"documents": documents})
    assert result == {"formatted": '<document id="1">\ntext\n</document>'}


def test_recent_documents_agent_invoke_empty_documents() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    result = agent.invoke({"documents": []})
    assert result == ""


# --- batch ---


def test_recent_documents_agent_batch_returns_full_outputs() -> None:
    inputs = [
        {"documents": [Document(page_content="old1"), Document(page_content="new1")]},
        {"documents": [Document(page_content="old2"), Document(page_content="new2")]},
    ]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=1)
    result = agent.batch(inputs)
    assert result == [
        '<document id="1">\nnew1\n</document>',
        '<document id="1">\nnew2\n</document>',
    ]


def test_recent_documents_agent_batch_passes_messages_dicts_to_inner_agent() -> None:
    inputs = [
        {"documents": [Document(page_content="old1"), Document(page_content="new1")]},
        {"documents": [Document(page_content="old2"), Document(page_content="new2")]},
    ]
    inner_agent = RunnableLambda(lambda inp: inp)
    agent = RecentDocumentsAgent(inner_agent=inner_agent, max_documents=1)
    result = agent.batch(inputs)
    assert result == [
        {"messages": [HumanMessage(content='<document id="1">\nnew1\n</document>')]},
        {"messages": [HumanMessage(content='<document id="1">\nnew2\n</document>')]},
    ]


def test_recent_documents_agent_batch_empty_inputs() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    assert agent.batch([]) == []


def test_recent_documents_agent_invoke_and_batch_agree() -> None:
    documents = [Document(page_content="old"), Document(page_content="new")]
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=1)
    invoke_result = agent.invoke({"documents": documents})
    batch_result = agent.batch([{"documents": documents}])
    assert [invoke_result] == batch_result


# --- _to_inner_input ---


def test_recent_documents_agent_to_inner_input_wraps_human_message() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=1)
    documents = [Document(page_content="old"), Document(page_content="new")]
    result = agent._to_inner_input(documents)
    assert result == {"messages": [HumanMessage(content='<document id="1">\nnew\n</document>')]}


def test_recent_documents_agent_to_inner_input_empty_documents() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    result = agent._to_inner_input([])
    assert result == {"messages": [HumanMessage(content="")]}


# --- _format_last_documents ---


def test_recent_documents_agent_format_last_documents_single_document() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=1)
    documents = [Document(page_content="old"), Document(page_content="new")]
    result = agent._format_last_documents(documents)
    assert result == '<document id="1">\nnew\n</document>'


def test_recent_documents_agent_format_last_documents_multiple_documents() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent(), max_documents=2)
    documents = [
        Document(page_content="a"),
        Document(page_content="b"),
        Document(page_content="c"),
    ]
    result = agent._format_last_documents(documents)
    assert result == ('<document id="1">\nb\n</document>\n\n<document id="2">\nc\n</document>')


def test_recent_documents_agent_format_last_documents_empty_documents() -> None:
    agent = RecentDocumentsAgent(inner_agent=_echo_agent())
    assert agent._format_last_documents([]) == ""
