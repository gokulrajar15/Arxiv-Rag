from sentence_transformers import SentenceTransformer
import litserve as ls
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import os

class EmbeddingAPI(ls.LitAPI):
    def setup(self, device):
        self.model = SentenceTransformer(
            'GokulRajaR/embeddinggemma-300m-qat-q8_0-unquantized',
            device=device,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN")
        )

    def decode_request(self, request):
        return request

    def predict(self, query):
            return self.model.encode_query(query)

    def encode_response(self, output):
        return output.tolist()
    
    def authorize(self, auth: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if auth.scheme != "Bearer" or auth.credentials != os.getenv("auth_token"):
            raise HTTPException(status_code=401, detail="Bad token")

if __name__ == "__main__":
    api = EmbeddingAPI()
    server = ls.LitServer(api, devices="cpu", accelerator="cpu")
    server.run(port=7860)
