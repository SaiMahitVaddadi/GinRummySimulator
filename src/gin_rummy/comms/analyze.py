"""Coordination analysis for partnership card games (Euchre first).

Implements an empirical mutual-information decomposition of implicit
signalling in a partnership trick-taking game:

* I(action; own_hand   | public)  — how much the action reflects the
  actor's own hand (baseline: any competent policy will have some).
* I(action; partner_hand | own_hand, public) — the "telepathy" quantity.
  For an independent-actor policy this is roughly zero once the actor's
  own hand is conditioned out; a coordinated policy drives it positive
  by choosing actions whose statistics depend on inferences a partner
  can also make. We must condition on own_hand because partner_hand is
  anti-correlated with the actor's hand under a fixed-size deck (24
  cards, 5-card hands), so *unconditional* MI is contaminated by a
  common-cause deck-constraint term unrelated to signalling. This is
  the empirical instantiation of the implicit-signalling channel
  discussed qualitatively in the Hanabi literature (Bard et al. 2020;
  SAD; OBL) but rarely measured in classical card games.
* I(action; opponent_hand | own_hand, public) — how much the action
  leaks about opponent private state after subtracting the same
  deck-constraint common cause. Rising values with hand-crafted
  coordination policies warn that the "signal" is really an information
  leak, not partnership-specific.

The estimator is the plug-in H(A|P) - H(A|H,P) computed on coarse
buckets. Coarse buckets keep the empirical distribution well populated
so that plug-in MI is not swamped by finite-sample bias. Bootstrap
confidence intervals resample *games* (block bootstrap over
trajectories), preserving within-game correlation between plays.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping, Sequence

from gin_rummy.cards import Card
from gin_rummy.comms.metrics import action_hand_mutual_information
from gin_rummy.eval.stats import BootstrapCI
from gin_rummy.variants.euchre import (
    EuchreGame,
    EuchreObservation,
    EuchrePolicy,
    RandomEuchrePolicy,
    Suit,
    TrickPlay,
    is_trump,
)


# ---------------------------------------------------------------- buckets ---


def hand_bucket(hand: Sequence[Card], trump_suit: Suit | None) -> str:
    """Compact, deterministic strength summary of a Euchre hand.

    Buckets are chosen to keep the alphabet tiny (well-conditioned MI):

    * ``strong_trump`` — ≥3 trumps, or the right bower plus any other trump,
      or ≥2 aces.
    * ``medium`` — 1–2 trumps or 1 ace.
    * ``weak`` — no trumps and no aces.

    If ``trump_suit`` is unknown (called before trump is set) we fall back
    to a rank-only bucket derived from aces + bowers counted across suits.
    """
    aces = sum(1 for c in hand if c.rank == "A")
    if trump_suit is None:
        jacks = sum(1 for c in hand if c.rank == "J")
        if aces >= 2 or jacks >= 2:
            return "strong_trump"
        if aces == 1 or jacks == 1:
            return "medium"
        return "weak"

    trumps = [c for c in hand if is_trump(c, trump_suit)]
    right_bower = any(c.rank == "J" and c.suit == trump_suit for c in trumps)

    if len(trumps) >= 3 or (right_bower and len(trumps) >= 2) or aces >= 2:
        return "strong_trump"
    if len(trumps) >= 1 or aces >= 1:
        return "medium"
    return "weak"


def public_bucket(obs: EuchreObservation) -> str:
    """Public state summary usable before the play is made.

    Deliberately coarse: only the lead-suit *category relative to trump*
    and a two-bucket trick phase. A larger public alphabet fragments the
    conditional distributions and inflates plug-in MI with finite
    samples; keeping the alphabet small (~4 bins in practice) keeps the
    estimator honest.
    """
    trump = obs.trump_suit
    lead = _current_lead_kind(obs, trump)
    phase = "early" if obs.trick_number <= 2 else "late"
    return f"lead={lead}|phase={phase}"


def _current_lead_kind(obs: EuchreObservation, trump: Suit | None) -> str:
    """Categorise the current trick's lead card, if any, as trump / offsuit
    / none. Approximated from ``trick_history[-1]`` (the last visible
    play), which for followers matches the current lead."""
    if not obs.trick_history:
        return "none"
    last_card = obs.trick_history[-1][0]
    if trump is not None and is_trump(last_card, trump):
        return "trump"
    return "offsuit"


def action_bucket(card: Card, trump_suit: Suit | None) -> str:
    """Coarsen a played card into a 5-symbol action alphabet."""
    if trump_suit is not None and is_trump(card, trump_suit):
        if card.rank in ("J", "A", "K"):
            return "trump_high"
        return "trump_low"
    if card.rank == "A":
        return "offsuit_ace"
    if card.rank in ("J", "Q", "K"):
        return "offsuit_face"
    return "offsuit_low"


# --------------------------------------------------------------- samples ---


@dataclass(frozen=True)
class PlaySample:
    """A single play, with everything MI needs to be computed on.

    ``partner_hand_bucket`` / ``opponent_hand_bucket`` are captured from
    the *actor's* perspective: partner = seat (actor+2)%4;
    opponent hand is a summary of the *pair* of opponents' hands combined
    (so we get one bucket per public row, not two).
    """

    game_id: int
    seat: int
    action: str
    own_hand: str
    partner_hand: str
    opponent_hand: str
    public: str


def _combined_opponent_bucket(
    hands: Sequence[Sequence[Card]], actor_seat: int, trump_suit: Suit | None
) -> str:
    """Combine both opponents' hand strengths into a single small bucket."""
    left = (actor_seat + 1) % 4
    right = (actor_seat + 3) % 4
    lb = hand_bucket(hands[left], trump_suit)
    rb = hand_bucket(hands[right], trump_suit)
    # Sort for symmetry: opponent pair is unordered.
    return "|".join(sorted([lb, rb]))


class _InstrumentedEuchreGame(EuchreGame):
    """EuchreGame subclass that records a PlaySample for every card played.

    Overrides ``_play_trick`` to snapshot every actor's own / partner /
    opponents' hands *before* the card is removed from the hand.
    """

    def __init__(
        self,
        *,
        policies: Sequence[EuchrePolicy],
        seed: int | None,
        dealer: int = 3,
        game_id: int = 0,
    ) -> None:
        super().__init__(policies=policies, seed=seed, dealer=dealer)
        self.game_id = game_id
        self.samples: list[PlaySample] = []

    def _play_trick(self, leader: int) -> TrickPlay:  # type: ignore[override]
        trick = TrickPlay(trick_number=len(self.tricks))
        lead_suit: Suit | None = None
        for i in range(4):
            seat = (leader + i) % 4
            legal = self._legal_plays(seat, lead_suit)
            obs = self._make_obs(seat)
            card = self.policies[seat].choose_play(obs, legal)
            if card not in legal:
                card = legal[0]

            # --- snapshot BEFORE the card leaves the hand ---
            own = hand_bucket(self.hands[seat], self.trump_suit)
            partner_seat = (seat + 2) % 4
            partner = hand_bucket(self.hands[partner_seat], self.trump_suit)
            opp = _combined_opponent_bucket(self.hands, seat, self.trump_suit)
            act = action_bucket(card, self.trump_suit)
            pub = public_bucket(obs)
            self.samples.append(
                PlaySample(
                    game_id=self.game_id,
                    seat=seat,
                    action=act,
                    own_hand=own,
                    partner_hand=partner,
                    opponent_hand=opp,
                    public=pub,
                )
            )

            self.hands[seat].remove(card)
            trick.plays.append((seat, card))
            if lead_suit is None:
                lead_suit = card.suit

        assert self.trump_suit is not None
        assert lead_suit is not None
        trick.lead_suit = lead_suit
        best_seat, best_card = trick.plays[0]
        from gin_rummy.variants.euchre import card_beats  # local import to avoid cycles

        for seat, card in trick.plays[1:]:
            if card_beats(card, best_card, lead_suit=lead_suit, trump=self.trump_suit):
                best_seat, best_card = seat, card
        trick.winner_id = best_seat
        return trick


# ------------------------------------------------------------- report ------


PolicyFactory = Callable[[random.Random], EuchrePolicy]


@dataclass
class CoordinationReport:
    n_games: int
    n_plays: int
    mi_action_own_hand: BootstrapCI
    mi_action_partner_hand: BootstrapCI
    mi_action_opponent_hand: BootstrapCI
    policy_name: str = "policy"

    def summary(self) -> str:
        def fmt(ci: BootstrapCI) -> str:
            return f"{ci.point:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]"

        return (
            f"CoordinationReport({self.policy_name}, "
            f"games={self.n_games}, plays={self.n_plays})\n"
            f"  I(action; own_hand      | public)            = {fmt(self.mi_action_own_hand)}\n"
            f"  I(action; partner_hand  | own_hand, public)  = {fmt(self.mi_action_partner_hand)}\n"
            f"  I(action; opponent_hand | own_hand, public)  = {fmt(self.mi_action_opponent_hand)}"
        )


# ---------------------------------------------------------- study runner ---


def _run_games(
    *,
    policy_factory: PolicyFactory,
    num_games: int,
    seed: int,
) -> list[list[PlaySample]]:
    """Play ``num_games`` and return samples grouped by game (for block
    bootstrap)."""
    master = random.Random(seed)
    per_game: list[list[PlaySample]] = []
    for gid in range(num_games):
        # Independent RNG per game so a policy_factory can consume it freely
        # without perturbing the next game's shuffle.
        game_seed = master.randrange(1 << 31)
        policy_rng = random.Random(master.randrange(1 << 31))
        policies = [policy_factory(policy_rng) for _ in range(4)]
        game = _InstrumentedEuchreGame(
            policies=policies, seed=game_seed, game_id=gid
        )
        game.play()
        per_game.append(game.samples)
    return per_game


def _mi_over_samples(
    samples: Iterable[PlaySample],
    hand_field: str,
    *,
    condition_on_own: bool,
) -> float:
    """Empirical I(action; hand_field | conditioner), in nats.

    ``condition_on_own``:
      * ``False`` — conditioner is the public bucket alone. Used for
        ``mi_action_own_hand`` where "own hand" is the target.
      * ``True`` — conditioner is (public bucket, own-hand bucket). Used
        for partner / opponent MI: without this, the shared-deck
        constraint (partner and actor draw from the same 24-card deck)
        creates a common-cause coupling between the actor's action and
        the partner's hand even under a fully random policy, drowning
        the true "coordination" signal. Conditioning on own_hand removes
        that common cause. See Bard et al. 2020 §5 for the analogous
        Hanabi decomposition.
    """
    triples: list[tuple[Hashable, Hashable, Hashable]] = []
    for s in samples:
        cond = (s.public, s.own_hand) if condition_on_own else s.public
        triples.append((s.action, getattr(s, hand_field), cond))
    return action_hand_mutual_information(triples)


def _bootstrap_mi(
    per_game: Sequence[Sequence[PlaySample]],
    hand_field: str,
    *,
    condition_on_own: bool,
    n_resamples: int,
    level: float,
    rng: random.Random,
) -> BootstrapCI:
    """Block bootstrap: resample games (with replacement) and recompute
    MI on the flattened per-play sample list."""
    if not per_game:
        return BootstrapCI(0.0, 0.0, 0.0, level, n_resamples)
    flat_all: list[PlaySample] = [s for game in per_game for s in game]
    point = _mi_over_samples(flat_all, hand_field, condition_on_own=condition_on_own)
    n = len(per_game)
    if n == 1:
        return BootstrapCI(point, point, point, level, n_resamples)

    stats: list[float] = []
    for _ in range(n_resamples):
        chosen = [per_game[rng.randrange(n)] for _ in range(n)]
        flat = [s for game in chosen for s in game]
        if not flat:
            stats.append(0.0)
            continue
        stats.append(
            _mi_over_samples(flat, hand_field, condition_on_own=condition_on_own)
        )
    stats.sort()
    alpha = (1 - level) / 2
    lower = stats[int(alpha * n_resamples)]
    upper = stats[min(int((1 - alpha) * n_resamples), n_resamples - 1)]
    return BootstrapCI(point, lower, upper, level, n_resamples)


def coordination_study(
    *,
    policy_factory: PolicyFactory,
    num_games: int,
    seed: int,
    n_resamples: int = 1000,
    level: float = 0.95,
    policy_name: str = "policy",
) -> CoordinationReport:
    """Run ``num_games`` Euchre hands under ``policy_factory`` and produce
    bootstrapped MI estimates for own/partner/opponent hand.

    ``policy_factory`` is called ``4 * num_games`` times with a shared
    ``random.Random`` so agents built from it can differ per seat while
    remaining reproducible from ``seed``.
    """
    per_game = _run_games(
        policy_factory=policy_factory, num_games=num_games, seed=seed
    )
    flat = [s for game in per_game for s in game]
    boot_rng = random.Random(seed ^ 0xC0FFEE)

    own = _bootstrap_mi(
        per_game,
        "own_hand",
        condition_on_own=False,
        n_resamples=n_resamples,
        level=level,
        rng=boot_rng,
    )
    partner = _bootstrap_mi(
        per_game,
        "partner_hand",
        condition_on_own=True,
        n_resamples=n_resamples,
        level=level,
        rng=boot_rng,
    )
    opp = _bootstrap_mi(
        per_game,
        "opponent_hand",
        condition_on_own=True,
        n_resamples=n_resamples,
        level=level,
        rng=boot_rng,
    )
    return CoordinationReport(
        n_games=num_games,
        n_plays=len(flat),
        mi_action_own_hand=own,
        mi_action_partner_hand=partner,
        mi_action_opponent_hand=opp,
        policy_name=policy_name,
    )


# -------------------------------------------------------- comparison util --


def compare_policies(
    *,
    policies: Mapping[str, PolicyFactory],
    num_games: int,
    seed: int,
    n_resamples: int = 1000,
    level: float = 0.95,
) -> dict[str, CoordinationReport]:
    """Run ``coordination_study`` per named policy and print a table.

    Returns the per-policy reports so callers can inspect them further.
    """
    reports: dict[str, CoordinationReport] = {}
    for i, (name, factory) in enumerate(policies.items()):
        reports[name] = coordination_study(
            policy_factory=factory,
            num_games=num_games,
            seed=seed + i,
            n_resamples=n_resamples,
            level=level,
            policy_name=name,
        )

    _print_comparison(reports)
    return reports


def _print_comparison(reports: Mapping[str, CoordinationReport]) -> None:
    header = (
        f"{'policy':<22} {'plays':>7}  {'MI(a;own|pub)':>18}  "
        f"{'MI(a;partner|own,pub)':>24}  {'MI(a;opp|own,pub)':>20}"
    )
    print(header)
    print("-" * len(header))
    for name, r in reports.items():
        def cell(ci: BootstrapCI) -> str:
            return f"{ci.point:.3f} [{ci.lower:.3f},{ci.upper:.3f}]"

        print(
            f"{name:<22} {r.n_plays:>7}  "
            f"{cell(r.mi_action_own_hand):>18}  "
            f"{cell(r.mi_action_partner_hand):>24}  "
            f"{cell(r.mi_action_opponent_hand):>20}"
        )


# ---------------------------------------------------------------- default --


def random_policy_factory(rng: random.Random) -> EuchrePolicy:
    """Convenience factory: fresh ``RandomEuchrePolicy`` sharing ``rng``."""
    return RandomEuchrePolicy(rng)


__all__ = [
    "CoordinationReport",
    "PlaySample",
    "PolicyFactory",
    "action_bucket",
    "compare_policies",
    "coordination_study",
    "hand_bucket",
    "public_bucket",
    "random_policy_factory",
]
