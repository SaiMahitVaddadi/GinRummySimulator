"""Multi-agent communication for card-game policies.

Modules
-------
* ``channel``  — a broadcast message channel that lives alongside game state;
  message history is captured and can be attached to observations.
* ``signals``  — a small structured vocabulary (SignalKind) that any policy
  can emit, giving cheap-talk experiments a well-typed baseline.
* ``metrics``  — information-theoretic coordination scores (mutual
  information between partner actions and hidden hands; per-turn regret
  proxy for signal quality).
* ``byzantine`` — stress tests where one partner emits deceptive signals
  (arXiv:2606.07790 pattern).
* ``orchestrator`` — LangGraph-lite blackboard router used to coordinate
  LLM/hybrid agents around a shared game state.
"""

from gin_rummy.comms.analyze import (
    CoordinationReport,
    PlaySample,
    action_bucket,
    compare_policies,
    coordination_study,
    hand_bucket,
    public_bucket,
    random_policy_factory,
)
from gin_rummy.comms.byzantine import ByzantinePolicy, byzantine_swap
from gin_rummy.comms.channel import ChatMessage, MessageChannel
from gin_rummy.comms.metrics import (
    action_hand_mutual_information,
    signal_cooperation_index,
)
from gin_rummy.comms.orchestrator import BlackboardOrchestrator
from gin_rummy.comms.signals import SignalKind, StructuredSignal, TalkingPolicy

__all__ = [
    "BlackboardOrchestrator",
    "ByzantinePolicy",
    "ChatMessage",
    "CoordinationReport",
    "MessageChannel",
    "PlaySample",
    "SignalKind",
    "StructuredSignal",
    "TalkingPolicy",
    "action_bucket",
    "action_hand_mutual_information",
    "byzantine_swap",
    "compare_policies",
    "coordination_study",
    "hand_bucket",
    "public_bucket",
    "random_policy_factory",
    "signal_cooperation_index",
]
