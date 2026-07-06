from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from glyphik.agents import StructuredLLMAgent


class Answer(BaseModel):
    """Simple structured output type used across the tests."""

    value: str


class _FakeLLM:
    """Duck-typed stand-in for a chat model.

    Records the messages it receives on
    ``with_structured_output(...).invoke`` and returns pre-programmed
    ``include_raw``-style results, so no real network call or LangChain
    chat-model machinery is needed.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.call_count = 0
        self.received_messages: list[Any] | None = None
        self.received_output_type: Any = None
        self.received_include_raw: bool | None = None

    def with_structured_output(self, output_type: Any, include_raw: bool = False) -> RunnableLambda:
        self.received_output_type = output_type
        self.received_include_raw = include_raw

        def _respond(
            messages: list[Any],
            config: Any = None,  # noqa: ARG001
            **kwargs: Any,  # noqa: ARG001
        ) -> dict[str, Any]:
            self.received_messages = messages
            result = self._responses[self.call_count]
            self.call_count += 1
            return result

        return RunnableLambda(_respond)


def _fake_llm(responses: list[dict[str, Any]]) -> _FakeLLM:
    """Return a ``_FakeLLM`` pre-programmed with the given responses."""
    return _FakeLLM(responses)


def _parsed_result(value: str) -> dict[str, Any]:
    """Return an ``include_raw``-style result for a successful parse."""
    return {
        "raw": AIMessage(content=value),
        "parsed": Answer(value=value),
        "parsing_error": None,
    }


def _failed_result(error: Exception) -> dict[str, Any]:
    """Return an ``include_raw``-style result for a failed parse."""
    return {"raw": AIMessage(content=""), "parsed": None, "parsing_error": error}


def _failed_result_with_raw(error: Exception, raw_content: str) -> dict[str, Any]:
    """Return an ``include_raw``-style failed-parse result with custom
    raw content."""
    return {"raw": AIMessage(content=raw_content), "parsed": None, "parsing_error": error}


######################################################
#     Tests for StructuredLLMAgent                  #
######################################################


# --- Constructor ---


def test_structured_llm_agent_stores_llm() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert agent._llm is llm


def test_structured_llm_agent_stores_system_prompt() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert agent._system_prompt == "You are helpful."


def test_structured_llm_agent_stores_output_type() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert agent._output_type is Answer


def test_structured_llm_agent_calls_with_structured_output_include_raw() -> None:
    llm = _fake_llm([_parsed_result("a")])
    StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert llm.received_output_type is Answer
    assert llm.received_include_raw is True


def test_structured_llm_agent_repr_contains_class_name() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert "StructuredLLMAgent" in repr(agent)


def test_structured_llm_agent_str_contains_class_name() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert "StructuredLLMAgent" in str(agent)


# --- invoke ---


def test_structured_llm_agent_invoke_returns_parsed_output() -> None:
    llm = _fake_llm([_parsed_result("hello")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert result == Answer(value="hello")


def test_structured_llm_agent_invoke_prepends_system_prompt() -> None:
    llm = _fake_llm([_parsed_result("hello")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert llm.received_messages == [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="hi"),
    ]


def test_structured_llm_agent_invoke_missing_messages_key_uses_empty_list() -> None:
    llm = _fake_llm([_parsed_result("hello")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    agent.invoke({})
    assert llm.received_messages == [SystemMessage(content="You are helpful.")]


def test_structured_llm_agent_invoke_replaces_existing_leading_system_message() -> None:
    llm = _fake_llm([_parsed_result("hello")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    agent.invoke(
        {
            "messages": [
                SystemMessage(content="old prompt"),
                HumanMessage(content="hi"),
            ]
        }
    )
    assert llm.received_messages == [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="hi"),
    ]


def test_structured_llm_agent_invoke_ignores_extra_input_keys() -> None:
    llm = _fake_llm([_parsed_result("hello")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent.invoke({"messages": [HumanMessage(content="hi")], "extra": "ignored"})
    assert result == Answer(value="hello")


def test_structured_llm_agent_invoke_raises_on_parsing_error() -> None:
    error = ValueError("bad output")
    llm = _fake_llm([_failed_result(error)])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    with pytest.raises(ValueError, match="Failed to parse LLM output"):
        agent.invoke({"messages": [HumanMessage(content="hi")]})


def test_structured_llm_agent_invoke_falls_back_to_manual_json_parsing() -> None:
    error = ValueError("no tool call emitted")
    llm = _fake_llm([_failed_result_with_raw(error, '{"value": "hello"}')])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert result == Answer(value="hello")


def test_structured_llm_agent_invoke_raises_when_both_parsing_paths_fail() -> None:
    error = ValueError("no tool call emitted")
    llm = _fake_llm([_failed_result_with_raw(error, "not json at all")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    with pytest.raises(ValueError, match="Failed to parse LLM output"):
        agent.invoke({"messages": [HumanMessage(content="hi")]})


# --- batch ---


def test_structured_llm_agent_batch_returns_parsed_outputs() -> None:
    llm = _fake_llm([_parsed_result("a"), _parsed_result("b")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent.batch(
        [
            {"messages": [HumanMessage(content="1")]},
            {"messages": [HumanMessage(content="2")]},
        ]
    )
    assert result == [Answer(value="a"), Answer(value="b")]


def test_structured_llm_agent_batch_empty_inputs() -> None:
    llm = _fake_llm([])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    assert agent.batch([]) == []


def test_structured_llm_agent_invoke_and_batch_agree() -> None:
    llm_invoke = _fake_llm([_parsed_result("a")])
    llm_batch = _fake_llm([_parsed_result("a")])
    agent_invoke = StructuredLLMAgent(
        llm=llm_invoke, system_prompt="You are helpful.", output_type=Answer
    )
    agent_batch = StructuredLLMAgent(
        llm=llm_batch, system_prompt="You are helpful.", output_type=Answer
    )
    invoke_result = agent_invoke.invoke({"messages": [HumanMessage(content="hi")]})
    batch_result = agent_batch.batch([{"messages": [HumanMessage(content="hi")]}])
    assert [invoke_result] == batch_result


# --- _unwrap ---


def test_structured_llm_agent_unwrap_returns_parsed_value() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent._unwrap(_parsed_result("hello"))
    assert result == Answer(value="hello")


def test_structured_llm_agent_unwrap_raises_with_output_type_and_error_in_message() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("boom")
    with pytest.raises(ValueError, match="Answer") as exc_info:
        agent._unwrap(_failed_result(error))
    assert "boom" in str(exc_info.value)


# --- _unwrap: fallback to parse_json_to_structured ---


def test_structured_llm_agent_unwrap_falls_back_to_manual_json_parsing() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("no tool call emitted")
    result = agent._unwrap(_failed_result_with_raw(error, '{"value": "hello"}'))
    assert result == Answer(value="hello")


def test_structured_llm_agent_unwrap_fallback_handles_fenced_json() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("no tool call emitted")
    raw_content = '```json\n{"value": "hello"}\n```'
    result = agent._unwrap(_failed_result_with_raw(error, raw_content))
    assert result == Answer(value="hello")


def test_structured_llm_agent_unwrap_fallback_handles_json_with_surrounding_prose() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("no tool call emitted")
    raw_content = 'Sure, here you go:\n{"value": "hello"}\nHope that helps!'
    result = agent._unwrap(_failed_result_with_raw(error, raw_content))
    assert result == Answer(value="hello")


def test_structured_llm_agent_unwrap_raises_when_fallback_also_fails() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("no tool call emitted")
    with pytest.raises(ValueError, match="Answer") as exc_info:
        agent._unwrap(_failed_result_with_raw(error, "not json at all"))
    assert "Fallback manual JSON parsing also failed" in str(exc_info.value)


def test_structured_llm_agent_unwrap_raises_when_fallback_json_fails_validation() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("no tool call emitted")
    # Missing the required "value" field.
    raw_content = '{"other_field": "oops"}'
    with pytest.raises(ValueError, match="Fallback manual JSON parsing also failed"):
        agent._unwrap(_failed_result_with_raw(error, raw_content))


def test_structured_llm_agent_unwrap_original_error_preserved_in_message() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    error = ValueError("original parsing error")
    with pytest.raises(ValueError, match="original parsing error"):
        agent._unwrap(_failed_result_with_raw(error, "not json"))


# --- _build_messages ---


def test_structured_llm_agent_build_messages_prepends_system_prompt() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent._build_messages({"messages": [HumanMessage(content="hi")]})
    assert result == [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="hi"),
    ]


def test_structured_llm_agent_build_messages_empty_input() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent._build_messages({})
    assert result == [SystemMessage(content="You are helpful.")]


def test_structured_llm_agent_build_messages_drops_only_leading_system_message() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent._build_messages(
        {
            "messages": [
                SystemMessage(content="old"),
                HumanMessage(content="hi"),
            ]
        }
    )
    assert result == [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="hi"),
    ]


def test_structured_llm_agent_build_messages_non_system_first_message_kept() -> None:
    llm = _fake_llm([_parsed_result("a")])
    agent = StructuredLLMAgent(llm=llm, system_prompt="You are helpful.", output_type=Answer)
    result = agent._build_messages({"messages": [HumanMessage(content="hi")]})
    assert result[1] == HumanMessage(content="hi")
