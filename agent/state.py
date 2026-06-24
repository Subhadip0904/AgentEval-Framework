import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Annotated[list, operator.add] tells LangGraph to append new messages
    # rather than replace the whole list on each graph step.
    messages: Annotated[list[BaseMessage], operator.add]
