from typing import Dict, Any

from pydantic import BaseModel


class LLMRequest(BaseModel):
    code: str = ""
    inputs: Dict[str, Any] = {}
