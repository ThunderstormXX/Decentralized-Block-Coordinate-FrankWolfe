from __future__ import annotations

from time import perf_counter


class Timer:
    def __init__(self) -> None:
        self.start = perf_counter()

    def elapsed(self) -> float:
        return perf_counter() - self.start
