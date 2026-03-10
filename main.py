"""main.py: this is the main entry point for the FastAPI.
We will define POST /chat endpoint to accept a text query and return agent's answer.
For answert returnig we will use the GET /chat maybe
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional


app = FastAPI(
    title = "Retriever Agent API",
    description="AI Agent that uses CRM tools to asnwer or retrieve information from the Knowledge base"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    """This class defines the structure of incoming chat message"""
    query: str


class ChatResponse(BaseModel):
    """This class is to define the response structure given by the agent"""
    answer: str

#Endpoints
@app.get("/")
def root():
    """Root endpoint to check if the API is running or not"""
    return {
        "status": "ok",
        "message": "API is running!"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try: 
        answer = f"Query: {request.query}"
        return ChatResponse(answer=answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    