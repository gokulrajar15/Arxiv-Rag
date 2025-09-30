from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List

from app.agent_infrastructure.agents.main_agent import rag_agent
from app.agent_infrastructure.evaluation.deepeval import run_deep_eval
from app.agent_infrastructure.guardrails.guardrails import guardrails_validator
from app.core.security import verify_token
from app.db.client import Database

@asynccontextmanager
async def lifespan(app: FastAPI):
    await Database.init()
    yield
    await Database.close()


app = FastAPI(
    title="ArXiv RAG API",
    version="1.0.0",
    description="API for chatting with ArXiv RAG agent",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict[str, str]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the ArXiv RAG API"}


@app.post("/chat/")
async def chat(request_data: ChatRequest, background_tasks: BackgroundTasks, token: str = Depends(verify_token)):
    """
    Chat with the ArXiv agent
    """
    messages = request_data.messages
    prompts = messages[-1]['content']
    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages'")

    guardrails_result = await guardrails_validator(prompts)
    if not guardrails_result.get("validationPassed", False):
        return JSONResponse({"response": "Sorry, I cannot assist with that request."})

    current_date = datetime.now().strftime("%Y-%m-%d")
    response = await rag_agent.ainvoke({
        "messages": messages ,
        "date": current_date,
        "retrieved_docs": []
    })
    current_response = response["messages"][-1].content
    tool_used = []
    for msg in response['messages']:
        if getattr(msg, 'tool_calls', []):
            tool_used.append(msg.tool_calls[0]['name'])
    retrieved_docs = response['retrieved_docs']
    background_tasks.add_task(run_deep_eval,
        user_query=prompts,
        agent_output=current_response,
        retrieved_docs=retrieved_docs,
        tools_used=tool_used,
    )
    return response
