import heapq
from datetime import datetime
from typing import List


class KeyObj:
    def __init__(
            self,
            key: str,
    ):
        self.key = key
        self.time = datetime.now()

    def __lt__(self, other):
        return self.time < other.time

    def refresh(self):
        self.time = datetime.now()

    def __str__(self):
        return f"key:{self.key}, time:{self.time}"


class KeyPool:
    def __init__(
            self,
            key_list: List[str],
    ):
        self.heap = [KeyObj(key) for key in key_list]
        heapq.heapify(self.heap)

    def get(self):
        key_obj = heapq.heappop(self.heap)
        key_obj.refresh()
        heapq.heappush(self.heap, key_obj)
        return key_obj.key

    def __str__(self):
        return str([str(key) for key in self.heap])
