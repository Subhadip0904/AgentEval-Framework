import os
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolNode
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
tool_executor = ToolExecutor(tools)


def should_use_tool(state: AgentState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def agent_node(state: AgentState):
    messages = state["messages"]
    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools]
    )
    system_prompt = f"""You are an AI assistant helping SSD/NAND engineers.
Available tools:
{tool_descriptions}

Respond with structured answers. If you need information, use the appropriate tool.
If the user asks about specifications, use spec_search.
If the user asks to classify a log, use log_classifier.
Otherwise, answer directly."""

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            *[
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in messages
            ],
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return {"messages": messages + [AIMessage(content=response.choices[0].message.content)]}


def tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if not hasattr(last_message, "tool_calls"):
        return state

    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]

        result = tool_executor.invoke(
            {"tool_name": tool_name, "tool_input": tool_input}
        )
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": messages + tool_results}


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent", should_use_tool, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

compiled_graph = graph.compile()
