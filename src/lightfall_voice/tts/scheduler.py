from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Literal

Priority = Literal["normal", "high"]


@dataclass
class SpeechItem:
    text: str
    voice: str
    speed: float
    priority: Priority = "normal"
    deadline: float | None = None


class SpeechScheduler:
    def __init__(
        self,
        start: Callable[[SpeechItem], None],
        cancel: Callable[[SpeechItem], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._start = start
        self._cancel = cancel
        self._clock = clock
        self.current: SpeechItem | None = None
        self.high_queue: deque[SpeechItem] = deque()
        self.pending_normal: SpeechItem | None = None

    def _is_expired(self, item: SpeechItem) -> bool:
        return item.deadline is not None and item.deadline < self._clock()

    def _start_next(self) -> None:
        while self.high_queue:
            candidate = self.high_queue.popleft()
            if not self._is_expired(candidate):
                self.current = candidate
                self._start(candidate)
                return

        if self.pending_normal is not None:
            candidate = self.pending_normal
            self.pending_normal = None
            if not self._is_expired(candidate):
                self.current = candidate
                self._start(candidate)

    def submit(self, item: SpeechItem) -> None:
        if item.priority == "high":
            if self.current is None:
                self.high_queue.append(item)
                self._start_next()
            elif self.current.priority == "normal":
                prev = self.current
                self.current = None
                self._cancel(prev)
                self.high_queue.append(item)
                self._start_next()
            else:
                self.high_queue.append(item)
        else:
            if self.current is None:
                self.pending_normal = item
                self._start_next()
            elif self.current.priority == "normal":
                prev = self.current
                self.current = None
                self._cancel(prev)
                self.pending_normal = item
                self._start_next()
            else:
                self.pending_normal = item

    def finished(self, item: SpeechItem) -> None:
        if self.current is None or item is not self.current:
            return
        self.current = None
        self._start_next()

    def stop_all(self) -> list[SpeechItem]:
        interrupted = self.current
        queued_high = list(self.high_queue)
        self.current = None
        self.high_queue.clear()
        self.pending_normal = None
        saved: list[SpeechItem] = []
        if interrupted is not None:
            self._cancel(interrupted)
            if interrupted.priority == "high":
                saved.append(interrupted)
        saved.extend(queued_high)
        return saved
