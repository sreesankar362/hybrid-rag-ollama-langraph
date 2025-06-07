from pydantic import BaseModel
from typing import List, Optional

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class QueryRequest(BaseModel):
    question: str
    provider: Optional[str] = "ollama"
    model_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    reasoning: Optional[str] = None
    thought_process: Optional[str] = None 