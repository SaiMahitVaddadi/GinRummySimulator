"""Blackboard orchestrator — LangGraph-lite for multi-agent coordination.

Rather than pull in LangGraph / AutoGen / CrewAI (each adds substantial
dependency weight), we reproduce the *pattern* they converge on: a shared
mutable state dict routed through a sequence of named nodes.

For a card game this maps to:

    state = {"channel": MessageChannel(), "history": [], ...}
    orch = BlackboardOrchestrator(state)
    orch.add_node("meta_coach", coach_fn)      # LLM meta-orchestrator
    orch.add_node("action_router", route_fn)   # per-turn routing
    ...
    orch.run(max_steps=200)

The game engine itself remains untouched — the orchestrator is a
side-channel that inspects the message log and can advise agents through
the channel. Meant as scaffolding for LLM-heavy hybrid experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


NodeFn = Callable[[dict[str, Any]], None]


@dataclass
class BlackboardOrchestrator:
    state: dict[str, Any] = field(default_factory=dict)
    _nodes: list[tuple[str, NodeFn]] = field(default_factory=list)

    def add_node(self, name: str, fn: NodeFn) -> None:
        self._nodes.append((name, fn))

    def step(self) -> None:
        for _, fn in self._nodes:
            fn(self.state)

    def run(self, *, max_steps: int, stop_key: str = "done") -> None:
        for _ in range(max_steps):
            self.step()
            if self.state.get(stop_key):
                return
