from __future__ import annotations

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from rag_pageindex.agent.tools import TOOLS
from rag_pageindex.agent.tracing import langchain_callbacks
from rag_pageindex.core.config import settings
from rag_pageindex.pageindex.prompts import render


def _build_chat_model() -> ChatOpenAI:
    if settings.llm_api_key is None:
        raise RuntimeError("LLM_API_KEY is not set; cannot build the agent's ChatOpenAI.")
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
    )


def build_graph() -> Runnable:
    """Compile the LangGraph that backs the agent-chat-ui chat session.

    When `settings.tracing_enabled`, binds the Langfuse LangChain callback
    handler to the compiled graph so every node, tool call, and LLM call
    is recorded under one Langfuse trace.
    """
    compiled: CompiledStateGraph = create_react_agent(
        model=_build_chat_model(),
        tools=TOOLS,
        prompt=render("agent_system.j2"),
    )
    callbacks = langchain_callbacks()
    if callbacks:
        return compiled.with_config({"callbacks": callbacks})
    return compiled


graph: Runnable = build_graph()
