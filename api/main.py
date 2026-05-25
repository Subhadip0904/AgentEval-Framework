from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Micron AI Engineering Assistant", version="0.1.0")

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
    """
    Main endpoint. Accepts a natural language question,
    runs it through the LangGraph agent, returns a structured response.
    """
    try:
        # In production: invoke the LangGraph agent here
        # result = graph.invoke({"messages": [HumanMessage(req.question)]})
        # For demo purposes:
        answer = f"Answer to: {req.question}"
        sources = ["NVMe_2.0_spec section 4.1"]
        return AskResponse(answer=answer, sources=sources, confidence="high")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "ok", "version": "0.1.0"}

# Run: uvicorn api.main:app --reload
# Then open: http://localhost:8000/docs for auto-generated Swagger UI