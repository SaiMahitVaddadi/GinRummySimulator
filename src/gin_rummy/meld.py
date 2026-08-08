"""Optimal meld decomposition.

Given a hand of cards, partition it into (melds, deadwood) that minimises
the total point value of the deadwood. Uses exact bitmask dynamic
programming over hand subsets. For a 10-card hand this is ~1k states ×
a few hundred candidate melds — well under a millisecond.

A **set** is 3+ cards of the same rank.
A **run** is 3+ consecutive ranks of the same suit (Ace low, no wrap).

Jokers (rank == "JKR") are treated as inert here — games with wildcard
substitution should extend this module.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from gin_rummy.cards import Card


def _enumerate_sets(cards: list[Card]) -> list[tuple[int, ...]]:
    by_rank: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cards):
        if c.rank == "JKR":
            continue
        by_rank[c.rank].append(i)

    out: list[tuple[int, ...]] = []
    for indices in by_rank.values():
        n = len(indices)
        if n < 3:
            continue
        for size in range(3, min(n, 4) + 1):
            out.extend(combinations(indices, size))
    return out


def _enumerate_runs(cards: list[Card]) -> list[tuple[int, ...]]:
    by_suit: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(cards):
        if c.rank == "JKR":
            continue
        by_suit[c.suit][c.order].append(i)

    out: list[tuple[int, ...]] = []
    for by_order in by_suit.values():
        orders = sorted(by_order)
        i = 0
        while i < len(orders):
            j = i
            while j + 1 < len(orders) and orders[j + 1] == orders[j] + 1:
                j += 1
            span = orders[i : j + 1]
            for length in range(3, len(span) + 1):
                for start in range(len(span) - length + 1):
                    ranks = span[start : start + length]
                    # Cartesian expansion over duplicates at each rank.
                    frontier: list[list[int]] = [[]]
                    for r in ranks:
                        frontier = [
                            path + [idx] for path in frontier for idx in by_order[r]
                        ]
                    out.extend(tuple(path) for path in frontier)
            i = j + 1
    return out


def _candidate_melds(cards: list[Card]) -> list[tuple[int, int, tuple[int, ...]]]:
    """Return (bitmask, value, indices) for every candidate meld."""
    candidates = _enumerate_sets(cards) + _enumerate_runs(cards)
    out: list[tuple[int, int, tuple[int, ...]]] = []
    for meld in candidates:
        mask = 0
        value = 0
        for idx in meld:
            mask |= 1 << idx
            value += cards[idx].value
        out.append((mask, value, meld))
    return out


def optimal_decomposition(
    cards: list[Card],
) -> tuple[list[list[Card]], list[Card], int]:
    """Return (melds, deadwood_cards, deadwood_value) minimising deadwood value.

    The bitmask DP is exact for hands up to ~20 cards. Beyond that, we fall
    back to a greedy heuristic to keep memory bounded.
    """
    n = len(cards)
    if n == 0:
        return [], [], 0

    if n > 20:
        return _greedy_decomposition(cards)

    candidates = _candidate_melds(cards)
    full = 1 << n

    # dp[mask] = max meld value coverable using exactly the cards in `mask`.
    # Canonical form: each state resolves the lowest set bit either as
    # deadwood or as part of a meld that includes it, avoiding duplicate
    # transitions.
    dp = [0] * full
    parent: list[tuple[int, int]] = [(-1, 0)] * full  # (meld_index or -1, prev_mask)

    # Bucket melds by their lowest-set-bit for O(1) lookup per state.
    by_lowest: dict[int, list[int]] = defaultdict(list)
    for mi, (mask, _, _) in enumerate(candidates):
        by_lowest[mask & -mask].append(mi)

    for mask in range(1, full):
        lowest = mask & -mask
        without = mask ^ lowest
        best_val = dp[without]
        best_parent = (-1, without)
        for mi in by_lowest.get(lowest, ()):
            meld_mask, meld_val, _ = candidates[mi]
            if (meld_mask & mask) != meld_mask:
                continue
            prev = mask ^ meld_mask
            val = dp[prev] + meld_val
            if val > best_val:
                best_val = val
                best_parent = (mi, prev)
        dp[mask] = best_val
        parent[mask] = best_parent

    melds: list[list[Card]] = []
    used = 0
    mask = full - 1
    while mask:
        mi, prev = parent[mask]
        if mi >= 0:
            meld_indices = candidates[mi][2]
            melds.append([cards[i] for i in meld_indices])
            for i in meld_indices:
                used |= 1 << i
        mask = prev

    deadwood = [cards[i] for i in range(n) if not (used >> i) & 1]
    return melds, deadwood, sum(c.value for c in deadwood)


def _greedy_decomposition(
    cards: list[Card],
) -> tuple[list[list[Card]], list[Card], int]:
    """Legacy greedy: prefer sets first, then runs. Fast but suboptimal."""
    remaining = cards.copy()
    melds: list[list[Card]] = []

    by_rank: dict[str, list[Card]] = defaultdict(list)
    for c in remaining:
        by_rank[c.rank].append(c)
    for group in by_rank.values():
        if len(group) >= 3:
            take = group[: min(4, len(group))]
            melds.append(take)
            for c in take:
                remaining.remove(c)

    by_suit: dict[str, list[Card]] = defaultdict(list)
    for c in remaining:
        by_suit[c.suit].append(c)
    for group in by_suit.values():
        group.sort(key=lambda c: c.order)
        i = 0
        while i < len(group):
            j = i
            while j + 1 < len(group) and group[j + 1].order == group[j].order + 1:
                j += 1
            if j - i + 1 >= 3:
                run = group[i : j + 1]
                melds.append(run)
                for c in run:
                    remaining.remove(c)
            i = j + 1

    return melds, remaining, sum(c.value for c in remaining)
