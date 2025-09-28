from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState



class DocumentRetrieverState(BaseModel):
    """State for the Document Retriever tool."""
    query: Optional[str] = Field(None, description="The user query for document retrieval")
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(...)
    state: Annotated[dict, InjectedState] = Field(...)