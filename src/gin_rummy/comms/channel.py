"""Broadcast message channel — the atom of every communication mode.

Every message is stamped with (sender, turn, kind, payload) and appended
to a shared history. Games attach the channel to their state so any
policy that wants to observe partner signals can read the log; policies
that ignore it work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class ChatMessage:
    sender_id: int
    turn: int
    kind: str  # "signal" | "chat" | "byzantine" | "orch"
    payload: str


@dataclass
class MessageChannel:
    """Thread-safe-enough (single-writer) append-only log of messages."""

    messages: list[ChatMessage] = field(default_factory=list)

    def broadcast(self, sender_id: int, turn: int, kind: str, payload: str) -> None:
        self.messages.append(ChatMessage(sender_id, turn, kind, payload))

    def since(self, turn: int) -> Iterator[ChatMessage]:
        return (m for m in self.messages if m.turn >= turn)

    def by_sender(self, sender_id: int) -> Iterator[ChatMessage]:
        return (m for m in self.messages if m.sender_id == sender_id)

    def __len__(self) -> int:
        return len(self.messages)
