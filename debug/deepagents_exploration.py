r"""Contain code to explore deepagents lib."""

from __future__ import annotations

import logging

from coola.display import str_pydantic_model
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from zenpyre.utils.rich import configure_rich_logging, print_markdown, print_pretty
from zenpyre.utils.token_usage import log_token_usage

logger: logging.Logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)


def build_agent() -> None:  # noqa: D103
    research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report. \
"""

    model = init_chat_model(model="ollama:gemma4:e2b-mlx", temperature=0)
    logger.info(str_pydantic_model(model, exclude_none=True))
    agent = create_agent(model=model, system_prompt=research_instructions)
    logger.info(agent)
    # logger.info(agent.__class__.__mro__)

    response = agent.invoke({"messages": [{"role": "user", "content": "What is an atom?"}]})
    log_token_usage(response)
    logger.info("type=%s  keys=%s", type(response), sorted(response.keys()))
    print_pretty(response["messages"])
    print_markdown(response["messages"][-1].content)


def build_deep_agent() -> None:  # noqa: D103
    research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report. \
"""

    # TODO: how to limit the number of calls?  # noqa: TD002, TD003

    model = init_chat_model(model="ollama:gemma4:e2b-mlx", temperature=0)
    logger.info(str_pydantic_model(model))
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
    print_pretty(response["messages"])
    print_markdown(response["messages"][-1].content)


def main() -> None:  # noqa: D103
    build_agent()
    # build_deep_agent()


if __name__ == "__main__":
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
