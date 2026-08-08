"""Information-theoretic communication metrics.

Two quantities matter for a coordination study:

* ``action_hand_mutual_information`` — I(A; H | pub), an empirical estimate
  of how much a player's action *leaks* about their own hidden hand once
  the public history is conditioned on. Higher = the action is a more
  informative signal (implicit signaling); lower = the action reveals
  little. This is the honest, information-theoretic frame that the Hanabi
  literature treats only qualitatively (Bard et al. 2020; SAD; OBL).

* ``signal_cooperation_index`` — for a partnership game, the fraction of
  turns where a partner's action is consistent with the signal that was
  broadcast on the previous turn. A crude proxy for "does cheap talk
  actually change behavior?" — useful for Byzantine cheap-talk studies.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Sequence


def action_hand_mutual_information(
    samples: Sequence[tuple[Hashable, Hashable, Hashable]],
) -> float:
    """Empirical MI, in nats, between action and hand given public state.

    Each sample is ``(action, hand_bucket, public_bucket)``. The estimator
    is the plug-in ``I(A;H|P) = H(A|P) - H(A|H,P)`` computed on the joint
    empirical distribution. Cheap and correct for coarse buckets; biased
    upward for tiny samples (Miller-Madow correction not applied).
    """
    if not samples:
        return 0.0

    n = len(samples)
    p_joint: Counter[tuple[Hashable, Hashable, Hashable]] = Counter(samples)
    p_pub: Counter[Hashable] = Counter(s[2] for s in samples)
    p_a_pub: Counter[tuple[Hashable, Hashable]] = Counter((s[0], s[2]) for s in samples)
    p_h_pub: Counter[tuple[Hashable, Hashable]] = Counter((s[1], s[2]) for s in samples)

    mi = 0.0
    for (a, h, p), c_ahp in p_joint.items():
        p_ahp = c_ahp / n
        p_p = p_pub[p] / n
        p_ap = p_a_pub[(a, p)] / n
        p_hp = p_h_pub[(h, p)] / n
        if p_ap > 0 and p_hp > 0 and p_p > 0:
            mi += p_ahp * math.log((p_ahp * p_p) / (p_ap * p_hp))
    return max(0.0, mi)


def signal_cooperation_index(
    signals: Sequence[tuple[int, str]],
    partner_responses: Sequence[tuple[int, str]],
    expected_response: dict[str, str],
) -> float:
    """Fraction of (partner_signal, partner_response) pairs where the
    response matches ``expected_response[signal]``.

    ``signals`` and ``partner_responses`` are lists of (turn, value); we
    match a response at turn t+1 to the signal at turn t.
    """
    by_turn = {t: v for t, v in signals}
    matches = 0
    total = 0
    for t_resp, resp in partner_responses:
        prev = by_turn.get(t_resp - 1)
        if prev is None or prev not in expected_response:
            continue
        total += 1
        if resp == expected_response[prev]:
            matches += 1
    return matches / total if total else 0.0
