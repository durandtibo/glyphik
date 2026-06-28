r"""Contain code to explore deepagents lib."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from deepagents import (
    create_deep_agent,
)
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from glyphik.utils.logging import log_markdown, log_pretty

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

_LLM_EXCLUDED_FIELDS: frozenset[str] = frozenset({"default_headers", "default_query"})


def log_model(model: BaseChatModel) -> None:  # noqa: D103
    config: dict[str, Any] = {
        k: v
        for k, v in model.model_dump().items()
        if k not in _LLM_EXCLUDED_FIELDS and not isinstance(v, SecretStr)
    }
    config["class"] = type(model).__qualname__
    logger.info(config)
    log_pretty(config, title="model configuration")


def build_agent() -> None:  # noqa: D103
    research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report. \
"""

    model = init_chat_model(model="ollama:gemma4:e2b-mlx", temperature=0)
    log_model(model)
    agent = create_agent(model=model, system_prompt=research_instructions)
    logger.info(agent)
    logger.info(agent.__class__.__mro__)

    response = agent.invoke({"messages": [{"role": "user", "content": "What is an atom?"}]})
    logger.info("type=%s  keys=%s", type(response), sorted(response.keys()))
    log_pretty(response["messages"])
    log_markdown(response["messages"][-1].content)


def build_deep_agent() -> None:  # noqa: D103
    research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report. \
"""

    # TODO: how to limit the number of calls?  # noqa: TD002, TD003

    model = init_chat_model(model="ollama:gemma4:e2b-mlx", temperature=0)
    log_model(model)
    agent = create_deep_agent(model=model, subagents=[], system_prompt=research_instructions)
    logger.info(agent)
    logger.info(agent.__class__.__mro__)
    # log_model(agent)

    # register_harness_profile(
    #     "ollama:gemma4",
    #     HarnessProfile(
    #         general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    #     ),
    # )

    # Run the agent
    response = agent.invoke({"messages": [{"role": "user", "content": "What is an atom?"}]})
    logger.info("type=%s  keys=%s", type(response), sorted(response.keys()))
    log_pretty(response["messages"])

    log_markdown(response["messages"][-1].content)


def main() -> None:  # noqa: D103
    build_agent()
    # build_deep_agent()


if __name__ == "__main__":
    main()
