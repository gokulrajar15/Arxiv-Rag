import asyncio
import json
from typing import List, Annotated
from async_lru import alru_cache 
from langchain_core.documents import Document
from langchain_core.tools import tool
from app.agent_infrastructure.infrastructure.llm_clients import gpt_41_mini
from app.db.client import Database
from app.agent_infrastructure.infrastructure.embeddings import CustomEmbedding
from app.agent_infrastructure.prompt_templates import multi_query_retriever_prompt
from app.schema.langgraph_tools_state import DocumentRetrieverState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId

embed = CustomEmbedding()

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """
    Remove duplicate documents based on title and abstract.
    
    Args:
        documents (List[Document]): List of documents that may contain duplicates.
        
    Returns:
        List[Document]: List of unique documents.
    """
    seen = set()
    unique_docs = []
    
    for doc in documents:
        identifier = (doc.metadata.get('title', ''), doc.page_content[:100])
        if identifier not in seen:
            seen.add(identifier)
            unique_docs.append(doc)
    
    return unique_docs

async def multi_query_retriever(query: str, num_queries: int) -> List[str]:
    """
    Generate multiple search queries from the original query

    Args:
        query (str): The original user query.

    Returns:
        list[str]: A list of generated search queries.
    """
    messages = multi_query_retriever_prompt(query, num_queries)
    try:
        response = await gpt_41_mini.ainvoke(messages)
    except Exception as e:
        print(f"Error with gpt_41_mini: {e}")
        return [query]
    try:
        queries = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        queries = [query]
    return queries

async def process_natural_language_query(query: str, num_queries: int) -> List[List[float]]:
    """
    Process a natural language query by:
    1. Passing it to the multi_query_retriever to generate multiple search queries
    2. Batch embedding all generated queries for optimal performance
    
    Args:
        query (str): The original natural language query from the user.

    Returns:
        List[List[float]]: A list of embeddings for each generated query.
    """
    multi_queries = await multi_query_retriever(query, num_queries)
    query_embeddings = embed.embed_documents(multi_queries)
    return query_embeddings

@alru_cache(maxsize=128)
async def document_retriever_utils(query: str) -> List[Document]:
    """
    Retrieves relevant documents based on a user query using optimized MultiQueryRetriever.

    Args:
        user_query (str): The user query for document retrieval.

    Returns:
        list[Document]: A list of relevant documents.
    """
    num_queries = 5 
    try:
        query_embeddings = await process_natural_language_query(query, num_queries)

        if not query_embeddings:
            print("No query embeddings generated.")
            return []
    
        search_results = await Database.fetch_batch_vector_search(query_vectors = query_embeddings, limit=5)  
        documents = [
            Document(
                page_content=result["abstract"], 
                metadata={"title": result["title"], "category": result["category"]}
            ) 
            for result in search_results
        ]
        unique_documents = deduplicate_documents(documents)

        return unique_documents
    except Exception as e:
        print(f"Error in document_retriever: {e}")
        return []


@tool(
    name_or_callable="document_retriever",
    description="This tool retrieves the documents relevant to the user query from the database",
    args_schema=DocumentRetrieverState
)
async def document_retriever(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[dict, InjectedState] = None,
    ) -> list[Document]:
    """
    Retrieves relevant documents based on a user query using optimized MultiQueryRetriever.

    Args:
        user_query (str): The user query for document retrieval.

    Returns:
        list[Document]: A list of relevant documents.
    """
    try:
        unique_documents = await document_retriever_utils(query)
        return Command(
            update={
                "retrieved_docs": [unique_documents],
                "messages": [ToolMessage(str(unique_documents), tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        print(f"Error in document_retriever: {e}")
        return Command(
            update={
                "retrieved_docs": [],
                "messages": [ToolMessage(f"Error: {e}", tool_call_id=tool_call_id)]
            }
        )
    

def docs_to_dicts(docs: List[Document]) -> List[dict]:
    """Helper for DeepEval + API responses."""
    return [
        {
            "title": d.metadata.get("title"),
            "category": d.metadata.get("category"),
            "abstract": d.page_content,
        }
        for d in docs
    ]
