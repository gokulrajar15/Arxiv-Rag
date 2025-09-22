from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class DocumentRetrieverState(BaseModel):
    """State for the Document Retriever tool."""
    query: Optional[str] = Field(None, description="The user query for document retrieval")
    multi_queries: List[str] = Field(default_factory=list, description="Multiple search queries generated from the original query")
    query_embeddings: List[List[float]] = Field(default_factory=list, description="Embeddings for each generated query")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents from the database")
    error: Optional[str] = Field(None, description="Error message if the document retrieval fails")
