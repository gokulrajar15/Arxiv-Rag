from typing import Annotated, Optional

from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class MAINAGENTSTATE(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]
    date: Optional[str] = None
   
