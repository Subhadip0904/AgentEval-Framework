from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    question: str
    context: Optional[str] = None
    user_role: Optional[str] = "engineer"


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str
