import requests 
import json 
from app.core.config import settings

url = settings.embedding_api_url

headers = { 
    "Content-Type": "application/json", 
    "Authorization": f"Bearer {settings.embedding_auth_token}"  
} 


def get_embeddings_from_api(data: list) -> list: 
    response = requests.post(url, headers=headers, data=json.dumps(data)) 
    if response.status_code == 200: 
        result = response.json() 
        if isinstance(result, dict):
            return result.get("embeddings", [])
        elif isinstance(result, list):
            return result
        else:
            print(f"Unexpected API response format: {type(result)}")
            print(f"Response: {result}")
            return []
    else: 
        print(f"Error: {response.status_code}") 
        print(response.text) 
        return [] 


class CustomEmbedding: 
    def embed_documents(self, texts: list) -> list: 
        """Get embeddings for documents (texts) using the custom API.""" 
        return get_embeddings_from_api(texts) 

    def embed_query(self, query: str) -> list: 
        """Get embedding for a single query using the custom API.""" 
        embeddings = get_embeddings_from_api([query]) 
        return embeddings[0] if embeddings else []

if __name__ == "__main__":
    # Example usage
    texts = ["Hello world", "Custom embedding API"]
    embedding_model = CustomEmbedding()
    embeddings = embedding_model.embed_documents(texts)
    print(str(embeddings[0]))