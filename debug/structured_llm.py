r"""Example: extract structured movie review data using a local Ollama
model wrapped in ``StructuredLLMAgent``."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from zenpyre.utils.rich import configure_rich_logging

from glyphik.agents import StructuredLLMAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You extract structured movie review information from user text. "
    "Respond ONLY with a JSON object with exactly these keys: "
    '"title" (string), "sentiment" (string: positive, negative, or mixed), '
    '"score" (integer, 0 to 10). '
    "No prose, no markdown code fences, no extra keys — JSON only."
)

REVIEW_TEXT = (
    "Dune: Part Two was visually stunning and the pacing "
    "was much better than the first film. I'd give it a 9/10."
)


class MovieReview(BaseModel):
    """Structured extraction target for a movie review."""

    title: str = Field(description="Title of the movie being reviewed.")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed.")
    score: int = Field(description="Reviewer's rating out of 10.")


def build_agent() -> StructuredLLMAgent[MovieReview]:
    r"""Construct a ``StructuredLLMAgent`` backed by a local Ollama
    model.

    The underlying ``ChatOllama`` is configured with ``format="json"`` so
    Ollama constrains sampling to syntactically valid JSON. This only
    guarantees valid JSON syntax, not schema alignment, so the system
    prompt explicitly spells out the expected keys and types.

    Returns:
        An agent that parses free-text movie reviews into
        :class:`MovieReview` instances.
    """
    llm = ChatOllama(model="gemma4:e2b-mlx", temperature=0, format="json")
    return StructuredLLMAgent(llm=llm, system_prompt=SYSTEM_PROMPT, output_type=MovieReview)


def main() -> None:
    r"""Run a single example review through the agent and log the result.

    Raises:
        ValueError: If the LLM output could not be parsed into
            :class:`MovieReview`. The exception is logged with its
            traceback before being re-raised.
    """
    agent = build_agent()
    logger.info("Agent: %s", agent)

    logger.info("Sending review text to agent: %s", REVIEW_TEXT)
    try:
        result = agent.invoke({"messages": [HumanMessage(content=REVIEW_TEXT)]})
    except ValueError:
        logger.exception("Failed to parse structured output from LLM")
        raise

    logger.info("Parsed result: %s", result)


if __name__ == "__main__":
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
