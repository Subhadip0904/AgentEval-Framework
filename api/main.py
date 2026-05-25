from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from agent.graph import compiled_graph
from api.schemas import AskRequest, AskResponse

app = FastAPI(title="Micron AI Engineering Assistant", version="0.1.0")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Main endpoint. Accepts a natural language question,
    runs it through the LangGraph agent, returns a structured response.
    """
    try:
        result = compiled_graph.invoke({
            "messages": [HumanMessage(content=req.question)],
            "current_tool": "",
            "results": {}
        })

        answer = result["messages"][-1].content if result["messages"] else "No answer"

        sources = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") or "source" in str(msg).lower():
                sources.append(str(msg)[:100])

        confidence = "high" if sources else "medium"

        return AskResponse(
            answer=answer,
            sources=sources or ["Unknown"],
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# Run: uvicorn api.main:app --reload
# Then open: http://localhost:8000/docs for auto-generated Swagger UI

@app.get("/")
async def health():
    return {"status": "ok", "version": "0.1.0"}

# Run: uvicorn api.main:app --reload
# Then open: http://localhost:8000/docs for auto-generated Swagger UI