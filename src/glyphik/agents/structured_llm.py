r"""Contain a structured LLM agent that parses messages into structured
output."""

from __future__ import annotations

__all__ = ["StructuredLLMAgent"]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import MultilineDisplayMixin
from coola.utils.string import truncate_str
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from zenpyre.utils.json_to_structured import (
    JsonStructuredOutputParseError,
    parse_json_to_structured,
)
from zenpyre.utils.token_usage import log_token_usage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

T = TypeVar("T")


class StructuredLLMAgent(Runnable[dict[str, Any], T], MultilineDisplayMixin):
    r"""Wrap a chat model to return a validated, structured output.

    Prepends ``system_prompt`` as a system message to the input and
    invokes the underlying LLM with structured-output parsing enabled.
    The raw LLM response is requested internally (``include_raw``) so
    parsing failures can be raised with a clear error instead of
    silently returning ``None``.

    Args:
        llm: The chat model to wrap.
        system_prompt: The system prompt prepended to every input.
        output_type: The type (e.g. a Pydantic model) that the LLM
            output should be parsed into.
    """

    def __init__(self, llm: BaseChatModel, system_prompt: str, output_type: type[T]) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._output_type = output_type

        self._structured_llm = llm.with_structured_output(output_type, include_raw=True)

    def invoke(
        self,
        input: dict[str, Any],  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        r"""Invoke the LLM and return the parsed structured output.

        Args:
            input: A dict expected to optionally contain a
                ``"messages"`` key holding the conversation so far.
                Only this key is used; any other keys are ignored,
                since the underlying chat model only accepts a
                message list. The configured system prompt is
                prepended to the messages before invocation.
            config: Optional runnable configuration forwarded to the
                underlying LLM.
            **kwargs: Additional keyword arguments forwarded to the
                underlying LLM's ``invoke``.

        Returns:
            The LLM output parsed into ``self._output_type``.

        Raises:
            ValueError: If the LLM output could not be parsed into
                ``self._output_type``.
        """
        messages = self._build_messages(input)
        result = self._structured_llm.invoke(messages, config, **kwargs)
        return self._unwrap(result)

    def _unwrap(self, result: dict[str, Any]) -> T:
        """Extract the parsed output from a raw structured-output
        result.

        If the underlying chat model already parsed the output
        successfully, that parsed value is returned directly. If
        parsing failed (e.g. the model didn't emit a proper tool call,
        which is common with small or local models), this falls back
        to manually parsing the raw message content as JSON via
        :func:`parse_json_to_structured` before giving up.

        Args:
            result: The dict returned by the underlying LLM when
                ``include_raw=True``, containing ``"raw"``,
                ``"parsed"``, and ``"parsing_error"`` keys.

        Returns:
            The value under ``"parsed"``, or the result of manually
            parsing ``result["raw"].content`` if the initial parsing
            failed.

        Raises:
            ValueError: If ``result["parsing_error"]`` is not ``None``
                and the raw content also cannot be parsed into
                ``self._output_type``.
        """
        log_token_usage(result)
        if result["parsing_error"] is None:
            return result["parsed"]

        raw_content = result["raw"].content
        try:
            return parse_json_to_structured(raw_content, self._output_type)
        except JsonStructuredOutputParseError as e:
            msg = (
                f"Failed to parse LLM output into {self._output_type!r}: "
                f"{result['parsing_error']!r}. Fallback manual JSON parsing "
                f"also failed: {e}"
            )
            raise ValueError(msg) from e

    def _build_messages(self, input: dict[str, Any]) -> list[Any]:  # noqa: A002
        """Build the message list to send to the underlying chat model.

        Chat models (and the runnable returned by
        ``with_structured_output``) accept ``LanguageModelInput`` —
        a string, a ``PromptValue``, or a list of messages — not an
        arbitrary dict. This extracts ``input["messages"]`` and
        prepends the configured system prompt, discarding any other
        keys in ``input`` since the chat model has no use for them.

        If ``input["messages"]`` already starts with a
        :class:`~langchain_core.messages.SystemMessage`, it is replaced
        rather than duplicated.

        Args:
            input: The agent-level input dict, expected to optionally
                contain a ``"messages"`` key holding a list of
                ``BaseMessage``-like objects.

        Returns:
            A list of messages with ``self._system_prompt`` as the
            first message, ahead of any existing non-system messages.
        """
        messages = list(input.get("messages", []))
        if messages and isinstance(messages[0], SystemMessage):
            messages = messages[1:]
        return [SystemMessage(content=self._system_prompt), *messages]

    def _get_repr_kwargs(self) -> dict[str, Any]:
        """Return the fields shown in this object's multiline repr.

        Returns:
            A mapping of field name to value for ``llm``,
            ``system_prompt``, and ``output_type``.
        """
        return {
            "llm": self._llm,
            "system_prompt": truncate_str(self._system_prompt),
            "output_type": self._output_type,
        }
