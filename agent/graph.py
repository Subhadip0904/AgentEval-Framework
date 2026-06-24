"""
LangGraph agent with local tool routing.

gemma3:4b-it-q4_K_M does not support Ollama/OpenAI tool calling, so this graph
routes to project tools in Python and uses Gemma only for normal chat synthesis.
Model and tool list are driven by config.yaml.
"""
import re
import yaml
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from agent.state import AgentState
from tools import TOOL_REGISTRY, get_tools

load_dotenv()

_cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_cfg_path, encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_llm_cfg = _cfg["llm"]
_enabled_tools = _cfg.get("tools", {}).get("enabled", [])
tools = get_tools(_enabled_tools)

llm = ChatOpenAI(
    model=_llm_cfg["model"],
    temperature=_llm_cfg.get("temperature", 0.0),
    max_tokens=_llm_cfg.get("max_tokens", 500),
    openai_api_base=_llm_cfg.get("base_url", "http://localhost:11434/v1"),
    openai_api_key=_llm_cfg.get("api_key", "ollama"),
)


def _latest_user_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return str(state["messages"][-1].content)


def _route_tool(question: str) -> str | None:
    text = question.lower()
    if "log:" in text or "classify" in text or "severity" in text:
        return "log_classifier"
    if "```" in question or re.search(r"\b(explain|what does|code|function|register|driver|verilog|systemverilog)\b", text):
        return "code_explainer"
    if re.search(r"\b(nvme|nve|pcie|ufs|emmc|nand|ssd|ecc|ftl|queue|wear|garbage|protocol|spec|speed|transfer)\b", text):
        return "spec_search"
    return None


def _strip_log_prefix(question: str) -> str:
    match = re.search(r"(?:classify\s+this\s+log|log)\s*:\s*(.+)", question, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else question.strip()


def _language_from_question(question: str) -> str:
    for language in ("SystemVerilog", "Verilog", "Python", "Rust", "C++", "C"):
        if language.lower() in question.lower():
            return language
    return "C"


def _tool_answer(tool_name: str, question: str) -> str:
    tool = TOOL_REGISTRY[tool_name]
    if tool_name == "log_classifier":
        return tool.invoke({"log_entry": _strip_log_prefix(question)})
    if tool_name == "code_explainer":
        return tool.invoke({"code_snippet": question, "language": _language_from_question(question)})
    if tool_name == "spec_search":
        sections = tool.invoke({"query": question})
        response = llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Answer the engineer's question using only these retrieved spec sections. "
                        "Be concise and cite the bracketed section numbers when useful.\n\n"
                        f"Question: {question}\n\nRetrieved sections:\n{sections}"
                    )
                )
            ]
        )
        return str(response.content).strip()
    raise ValueError(f"Unknown tool: {tool_name}")


def agent_node(state: AgentState) -> dict:
    question = _latest_user_text(state)
    tool_name = _route_tool(question)
    if tool_name in _enabled_tools:
        try:
            return {"messages": [AIMessage(content=_tool_answer(tool_name, question))]}
        except Exception as e:
            print(f"[agent_node] tool '{tool_name}' failed: {e}")
            # fall through to plain LLM answer instead of crashing

    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Sorry, I couldn't process that: {e}")]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")
builder.set_finish_point("agent")
compiled_graph = builder.compile()


if __name__ == "__main__":
    import time

    queries = [
        "Classify this log: Uncorrectable ECC error on block 0x3F2A, wear count 98%",
        "How fast is UFS 3.1 and what protocol does it use?",
    ]
    for query in queries:
        print(f"\n{'=' * 60}\nQUERY: {query}\n{'=' * 60}")
        result = compiled_graph.invoke({"messages": [HumanMessage(content=query)]})
        print(f"ANSWER: {result['messages'][-1].content}")
        time.sleep(2)
