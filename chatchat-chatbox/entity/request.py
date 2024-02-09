from typing import Dict, Any


class ChatRequest:
    query: str = ""
    json: Dict[str, Any] = {}

    def __init__(
            self,
            query,
    ):
        self.query = query
        self.json = {"query": self.query}
