import os
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv

from agent.state import AgentState
from tools.spec_search import spec_search
from tools.log_classifier import log_classifier
from tools.code_explainer import code_explainer

load_dotenv()

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

tools = [spec_search, log_classifier, code_explainer]


def agent_node(state: AgentState):
    """Agent node that decides which tool to use"""
    messages = state["messages"]

    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools]
    )

    system_prompt = f"""You are an AI assistant helping SSD/NAND engineers.
Available tools:
{tool_descriptions}

If the user asks about specifications, use spec_search.
If the user asks to classify a log, use log_classifier.
If the user asks to explain code, use code_explainer.
Otherwise, answer directly with your knowledge.

Be concise and technical."""

    messages_formatted = []
    for m in messages:
        if isinstance(m, HumanMessage):
            messages_formatted.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            messages_formatted.append({"role": "assistant", "content": m.content})

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            *messages_formatted,
        ],
        temperature=0.0,
        max_tokens=500,
    )

    answer = response.choices[0].message.content
    return {"messages": messages + [AIMessage(content=answer)]}


def should_continue(state: AgentState) -> Literal["end"]:
    """Simple routing - just return end for now"""
    return "end"


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

compiled_graph = graph.compile()
