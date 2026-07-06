from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from glyphik.agents import StructuredLLMAgent


class OutputModel:
    r"""A minimal stand-in for a structured output type (e.g. a Pydantic
    model) used as ``output_type`` in tests."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OutputModel) and self.value == other.value

    def __repr__(self) -> str:
        return f"OutputModel(value={self.value!r})"


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock ``BaseChatModel`` whose ``with_structured_output``
    returns a mock structured runnable."""
    llm = MagicMock(name="llm")
    structured_llm = MagicMock(name="structured_llm")
    llm.with_structured_output.return_value = structured_llm
    return llm


def make_agent(mock_llm: MagicMock, system_prompt: str = "You are helpful.") -> StructuredLLMAgent:
    """Build a ``StructuredLLMAgent`` wrapping ``mock_llm``."""
    return StructuredLLMAgent(llm=mock_llm, system_prompt=system_prompt, output_type=OutputModel)


def test_init_calls_with_structured_output_with_include_raw(mock_llm: MagicMock) -> None:
    make_agent(mock_llm)
    mock_llm.with_structured_output.assert_called_once_with(OutputModel, include_raw=True)


def test_invoke_returns_parsed_output_on_success(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm)
    expected = OutputModel(value="ok")
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": expected,
        "parsing_error": None,
    }

    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert result == expected


def test_invoke_raises_value_error_on_parsing_error(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm)
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": None,
        "parsing_error": ValueError("bad output"),
    }

    with pytest.raises(ValueError, match="Failed to parse LLM output"):
        agent.invoke({"messages": [HumanMessage(content="hi")]})


def test_invoke_prepends_system_prompt(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm, system_prompt="Be concise.")
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": OutputModel(value="ok"),
        "parsing_error": None,
    }
    human_message = HumanMessage(content="hi")

    agent.invoke({"messages": [human_message]})

    structured_llm = mock_llm.with_structured_output.return_value
    sent_payload = structured_llm.invoke.call_args.args[0]
    sent_messages = sent_payload["messages"]
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == "Be concise."
    assert sent_messages[1] is human_message


def test_invoke_replaces_existing_system_message(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm, system_prompt="New prompt.")
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": OutputModel(value="ok"),
        "parsing_error": None,
    }
    human_message = HumanMessage(content="hi")
    old_system_message = SystemMessage(content="Old prompt.")

    agent.invoke({"messages": [old_system_message, human_message]})

    structured_llm = mock_llm.with_structured_output.return_value
    sent_messages = structured_llm.invoke.call_args.args[0]["messages"]
    assert len(sent_messages) == 2
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == "New prompt."
    assert sent_messages[1] is human_message


def test_invoke_handles_missing_messages_key(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm, system_prompt="Be concise.")
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": OutputModel(value="ok"),
        "parsing_error": None,
    }

    agent.invoke({})

    structured_llm = mock_llm.with_structured_output.return_value
    sent_messages = structured_llm.invoke.call_args.args[0]["messages"]
    assert len(sent_messages) == 1
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == "Be concise."


def test_invoke_preserves_other_input_keys(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm)
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": OutputModel(value="ok"),
        "parsing_error": None,
    }

    agent.invoke({"messages": [HumanMessage(content="hi")], "extra": "value"})

    structured_llm = mock_llm.with_structured_output.return_value
    sent_payload = structured_llm.invoke.call_args.args[0]
    assert sent_payload["extra"] == "value"


def test_invoke_forwards_config_and_kwargs(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm)
    mock_llm.with_structured_output.return_value.invoke.return_value = {
        "raw": MagicMock(),
        "parsed": OutputModel(value="ok"),
        "parsing_error": None,
    }
    config = {"tags": ["test"]}

    agent.invoke({"messages": []}, config, some_kwarg="value")

    structured_llm = mock_llm.with_structured_output.return_value
    call = structured_llm.invoke.call_args
    assert call.args[1] == config
    assert call.kwargs == {"some_kwarg": "value"}


def test_get_repr_kwargs_contains_expected_fields(mock_llm: MagicMock) -> None:
    agent = make_agent(mock_llm, system_prompt="Be concise.")

    repr_kwargs = agent._get_repr_kwargs()

    assert repr_kwargs == {
        "llm": mock_llm,
        "system_prompt": "Be concise.",
        "output_type": OutputModel,
    }
