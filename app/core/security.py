from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

security = HTTPBearer()

# Authentication function
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.api_auth_token:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials.credentials
