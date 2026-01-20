from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class StrategyFile(BaseModel):
    id: str
    name: str
    content: str
    last_modified: datetime

class StrategyListResponse(BaseModel):
    items: List[StrategyFile]

class SaveStrategyRequest(BaseModel):
    name: str
    content: str

class AICodeGenRequest(BaseModel):
    context_code: str
    instruction: str

class AICodeGenResponse(BaseModel):
    generated_code: str
    explanation: Optional[str] = None
