from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Micron AI Engineering Assistant", version="0.1.0")

from agent.graph import compiled_graph
from langchain_core.messages import HumanMessage


class AskRequest(BaseModel):
    question: str
    context: Optional[str] = None   # optional additional context from the caller
    user_role: Optional[str] = "engineer"

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str  # "high" | "low" | "not_found"

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        question = req.question
        if req.context:
            question = f"{question}\n\nContext: {req.context}"
        result = compiled_graph.invoke({
            "messages": [HumanMessage(content=question)]
        })
        answer = result["messages"][-1].content
        sources = []  # extract from tool results if needed
        confidence = "high" if len(answer) > 50 else "low"
        return AskResponse(answer=answer, sources=sources, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
async def health():
    return {"status": "ok", "version": "0.1.0"}

# Run: uvicorn api.main:app --reload
# Then open: http://localhost:8000/docs for auto-generated Swagger UI
