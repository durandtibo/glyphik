from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from zenpyre.utils.rich import configure_rich_logging, print_pretty


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    predefined_queries: list[str]
    agent_calls_used: int
    max_agent_calls: int


def build_agent(llm, tools):
    """Build a LangGraph agent that:
      1. Runs a list of predefined queries deterministically.
      2. Then lets the LLM decide up to `max_agent_calls` additional tool calls.

    Assumes each tool takes a single `query` argument (e.g. a search tool).
    """
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    # Only really makes sense with exactly one tool for the "predefined
    # queries" phase, since we need to know which tool to call directly.
    # If you pass multiple tools, we use the first one for predefined queries.
    predefined_tool = tools[0]

    # ---- Node: run predefined queries deterministically ----
    def run_predefined_queries(state: AgentState) -> AgentState:
        queries = state["predefined_queries"]
        if not queries:
            return {}

        query = queries[0]
        result = predefined_tool.invoke({"query": query})

        msg = HumanMessage(
            content=f"[predefined {predefined_tool.name}] query='{query}'\nresult: {result}"
        )

        return {
            "messages": [msg],
            "predefined_queries": queries[1:],
        }

    def predefined_queries_remaining(state: AgentState) -> str:
        return "run_predefined_queries" if state["predefined_queries"] else "agent"

    # ---- Node: the LLM, which can decide to call tools ----
    def agent_node(state: AgentState) -> AgentState:
        remaining = state["max_agent_calls"] - state["agent_calls_used"]

        system = SystemMessage(
            content=(
                "You are a research agent. You have already run some predefined "
                "searches (see above). You may now call available tools up to "
                f"{remaining} more time(s) if you need additional information. "
                "If you don't need more tool calls, just answer directly without "
                "calling any tool."
            )
        )

        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    # ---- Node: execute tool call(s) requested by the LLM, enforcing budget ----
    def tool_node(state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        outputs = []
        calls_used = state["agent_calls_used"]

        for call in last_msg.tool_calls:
            if calls_used >= state["max_agent_calls"]:
                outputs.append(
                    ToolMessage(
                        content="Tool call budget exhausted. No more tool calls allowed.",
                        tool_call_id=call["id"],
                    )
                )
                continue

            tool_result = tools_by_name[call["name"]].invoke(call["args"])
            outputs.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
            calls_used += 1

        return {"messages": outputs, "agent_calls_used": calls_used}

    # ---- Routing after the agent node ----
    def route_after_agent(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        has_tool_calls = bool(getattr(last_msg, "tool_calls", None))

        if has_tool_calls and state["agent_calls_used"] < state["max_agent_calls"]:
            return "tools"
        return END

    # ---- Build the graph ----
    graph = StateGraph(AgentState)

    graph.add_node("run_predefined_queries", run_predefined_queries)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("run_predefined_queries")

    graph.add_conditional_edges(
        "run_predefined_queries",
        predefined_queries_remaining,
        {"run_predefined_queries": "run_predefined_queries", "agent": "agent"},
    )

    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


@tool
def search(query: str) -> str:
    """Search the web.

    `query` should describe the goal of the search.
    """
    # Replace with your real search implementation
    return f"[fake results for: {query}]"


def main() -> None:
    model = "ollama:gemma4:e2b-mlx"
    llm = init_chat_model(model=model, temperature=0)
    tools = [search]

    agent = build_agent(llm, tools)
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Research the current state of solid-state batteries.")
            ],
            "predefined_queries": [
                "solid-state battery energy density 2026",
                "solid-state battery manufacturers commercial timeline",
            ],
            "agent_calls_used": 0,
            "max_agent_calls": 3,
        }
    )
    print_pretty(result)


if __name__ == "__main__":
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
