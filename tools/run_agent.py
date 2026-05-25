import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import operator, sys

load_dotenv(r"C:\Users\User\Desktop\micronprep\.env")

sys.path.append(r"C:\Users\User\Desktop\micronprep")
from tools.spec_search import spec_search
from tools.log_classifier import log_classifier

tools = [spec_search, log_classifier]

llm = ChatOpenAI(
    model="llama3.2",
    temperature=0.1,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama"
).bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def agent_node(state: AgentState):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

queries = [
    "Classify this log: Uncorrectable ECC error on block 0x3F2A, wear count 98%",
    "How fast is UFS 3.1 and what protocol does it use?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)
    result = graph.invoke({"messages": [HumanMessage(content=q)]})
    print(f"ANSWER: {result['messages'][-1].content}")
    time.sleep(5)  # wait 5 seconds between queries to avoid rate limit