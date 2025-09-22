import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool
from app.agent_infrastrure.infrastructure.llm_clients import gpt_41
from app.db.client import Database
from app.agent_infrastrure.infrastructure.embeddings import CustomEmbedding
from app.agent_infrastrure.prompt_templates import multi_query_retriever_prompt

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

    
    return unique_docs

async def multi_query_retriever(query: str) -> List[str]:
    """
    Generate multiple search queries from the original query

    Args:
        query (str): The original user query.

    Returns:
        list[str]: A list of generated search queries.
    """
    retriever = multi_query_retriever_prompt(query)
    messages = [retriever]
    response = await gpt_41.ainvoke(messages)
    queries = response.content.strip().split('\n')
    return queries

async def process_natural_language_query(query: str) -> List[List[float]]:
    """
    Process a natural language query by:
    1. Passing it to the multi_query_retriever to generate multiple search queries
    2. Batch embedding all generated queries for optimal performance
    
    Args:
        query (str): The original natural language query from the user.

    Returns:
        List[List[float]]: A list of embeddings for each generated query.
    """
    multi_queries = await multi_query_retriever(query)
    print("multi queries:", multi_queries)

    query_embeddings = embed.embed_documents(multi_queries)
    print(f"Generated {len(query_embeddings)} query embeddings")
    
    return query_embeddings, multi_queries
    

@tool(
    name_or_callable="document_retriever",
    description="This tool retrieves the documents relevant to the user query from the database")

async def document_retriever(user_query: str) -> list[Document]:
    """
    Retrieves relevant documents based on a user query using optimized MultiQueryRetriever.
    
    This tool performs semantic search using embeddings to find relevant documents 
    from the database. It uses parallel processing for both embedding generation 
    and database queries to optimize performance.

    Args:
        user_query (str): The query provided by the user.

    Returns:
        list[Document]: A list of relevant documents.
    """
    import time
    start_time = time.time()
    
    try:
        await Database.init()
        
        embedding_start = time.time()
        query_embeddings, multi_queries = await process_natural_language_query(user_query)
        embedding_time = time.time() - embedding_start
        print(f"Embedding generation took: {embedding_time:.2f} seconds")
        
        if not query_embeddings:
            print("No query embeddings generated.")
            return []
     
        db_start = time.time()
        search_results = await Database.fetch_document(query_embeddings, k=5)  
        db_time = time.time() - db_start
        print(f"Database queries took: {db_time:.2f} seconds")
        
        documents = [
            Document(
                page_content=result["abstract"], 
                metadata={"title": result["title"], "category": result["category"]}
            ) 
            for result in search_results
        ]
        
        unique_documents = deduplicate_documents(documents)
        
        total_time = time.time() - start_time
        print(f"Total retrieval time: {total_time:.2f} seconds")
        print(f"Retrieved {len(search_results)} documents, {len(unique_documents)} unique")
        
        return unique_documents
        
    except Exception as e:  
        print(f"Error in document_retriever: {str(e)}")
        return []
    

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

async def example_usage():
    user_query = "Brownian motion"
    documents = await document_retriever.ainvoke({"user_query": user_query})

    print(f"Retrieved {len(documents)} documents for query: '{user_query}'\n")
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}")
        print(f"Title: {doc.metadata['title']}")
        print(f"Category: {doc.metadata['category']}")
        print("Abstract:")
        print(f"{doc.page_content}")
        print("-" * 80)  

# if __name__ == "__main__":
#     asyncio.run(example_usage())
