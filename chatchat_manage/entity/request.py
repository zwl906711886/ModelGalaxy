from typing import Dict, Any

from pydantic import BaseModel


class LLMRequest:
    code: str = ""
    inputs: Dict[str, Any] = {}
    json: Dict[str, Any] = {}

    def __init__(
            self,
            code: str,
            inputs: Dict[str, Any],
    ):
        self.code = code
        self.inputs = inputs
        self.json = {"code": self.code, "inputs": self.inputs}


class ChatRequest(BaseModel):
    query: str = ""
