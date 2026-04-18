from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .schemas import SessionSummary, TokenEvent

if TYPE_CHECKING:
    from transformer_lens import ActivationCache


@dataclass
class Session:
    id: str
    model_name: str
    prompt: str
    prompt_token_strs: list[str] = field(default_factory=list)
    tokens: list[TokenEvent] = field(default_factory=list)
    cache: ActivationCache | None = None
    status: str = "running"  # "running" | "complete" | "error"
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> SessionSummary:
        return SessionSummary(
            id=self.id,
            model_name=self.model_name,
            prompt=self.prompt,
            n_generated=len(self.tokens),
            created_at=self.created_at,
            status=self.status,
        )


class SessionStore:
    def __init__(self, capacity: int = 10) -> None:
        self._capacity = capacity
        self._items: OrderedDict[str, Session] = OrderedDict()

    def new_session(self, model_name: str, prompt: str) -> Session:
        sid = uuid.uuid4().hex
        session = Session(id=sid, model_name=model_name, prompt=prompt)
        self._items[sid] = session
        self._items.move_to_end(sid)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._items.get(session_id)
        if session is not None:
            self._items.move_to_end(session_id)
        return session

    def all(self) -> list[Session]:
        return list(self._items.values())


store = SessionStore()
