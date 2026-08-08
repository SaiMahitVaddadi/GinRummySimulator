#!/usr/bin/env python3
import random
from typing import List, Tuple, Optional
from collections import defaultdict


class Card:
    def __init__(self, rank: str, suit: str, deck_id: int = 0):
        self.rank = rank
        self.suit = suit
        self.value = self._get_value()
        self.deck_id = deck_id  # Identifier for multi-deck games

    def _get_value(self) -> int:
        if self.rank == "A":
            return 1
        elif self.rank in ["J", "Q", "K"]:
            return 10
        else:
            return int(self.rank)

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            self.rank == other.rank
            and self.suit == other.suit
            and hasattr(other, "deck_id")
            and self.deck_id == getattr(other, "deck_id", 0)
        )

    def __hash__(self):
        return hash((self.rank, self.suit, getattr(self, "deck_id", 0)))


class Card:
    def __init__(self, rank: str, suit: str, deck_id: int = 0):
        self.rank = rank
        self.suit = suit
        self.value = self._get_value()
        self.deck_id = deck_id  # Identifier for multi-deck games

    def _get_value(self) -> int:
        if self.rank == "A":
            return 1
        elif self.rank in ["J", "Q", "K"]:
            return 10
        else:
            return int(self.rank)

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (
            self.rank == other.rank
            and self.suit == other.suit
            and hasattr(other, "deck_id")
            and self.deck_id == getattr(other, "deck_id", 0)
        )

    def __hash__(self):
        return hash((self.rank, self.suit, getattr(self, "deck_id", 0)))


class Deck:
    def __init__(self, num_decks: int = 1):
        self.suits = ["♠", "♥", "♦", "♣"]
        self.ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        self.num_decks = num_decks
        self.reset()

    def reset(self):
        """Reset deck with specified number of standard 52-card decks"""
        self.cards = []
        for deck_id in range(self.num_decks):
            for suit in self.suits:
                for rank in self.ranks:
                    self.cards.append(Card(rank, suit, deck_id))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, num_cards: int = 1) -> List[Card]:
        drawn = []
        for _ in range(num_cards):
            if self.cards:
                drawn.append(self.cards.pop())
        return drawn

    def draw_one(self) -> Optional[Card]:
        if self.cards:
            return self.cards.pop()
        return None

    def cards_remaining(self) -> int:
        return len(self.cards)

    def total_cards(self) -> int:
        return self.num_decks * 52

    @staticmethod
    def calculate_optimal_decks(num_players: int) -> int:
        """Calculate optimal number of decks based on player count"""
        if num_players <= 2:
            return 1
        elif num_players <= 4:
            return 2
        elif num_players <= 6:
            return 3
        else:
            return min(num_players // 2, 8)  # Cap at 8 decks


class Player:
    def __init__(self, name: str, player_id: int = 0):
        self.name = name
        self.player_id = player_id
        self.hand = []
        self.melds = []
        self.deadwood = []

    def add_card(self, card: Card):
        self.hand.append(card)

    def remove_card(self, card: Card):
        if card in self.hand:
            self.hand.remove(card)

    def get_hand_value(self) -> int:
        return sum(card.value for card in self.hand)

    def __str__(self):
        return f"{self.name}: {[str(card) for card in self.hand]}"


class Game:
    """Base class for all Gin Rummy variants supporting multiple players"""

    def __init__(self, num_players: int = 2):
        self.num_players = max(2, min(num_players, 12))  # Support 2-12 players
        self.num_decks = Deck.calculate_optimal_decks(self.num_players)
        self.deck = Deck(self.num_decks)
        self.players = [Player(f"Player {i + 1}", i) for i in range(self.num_players)]
        self.discard_pile = []
        self.analyzer = MeldAnalyzer()
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.scores = {player.name: 0 for player in self.players}

    def setup_game(self):
        """Setup game with adaptive card dealing"""
        cards_per_player = self._calculate_cards_per_player()

        for player in self.players:
            for _ in range(cards_per_player):
                player.add_card(self.deck.draw_one())

        # Start discard pile with one card
        self.discard_pile.append(self.deck.draw_one())

    def _calculate_cards_per_player(self) -> int:
        """Calculate optimal cards per player based on deck size and player count"""
        total_cards_needed = self.num_players * 10 + 20  # 10 cards per player + buffer
        if self.num_decks * 52 < total_cards_needed:
            # Adjust to fewer cards if deck is insufficient
            return min(10, (self.num_decks * 52 - 20) // self.num_players)
        return 10

    def get_next_player(self) -> Player:
        """Get next player in rotation"""
        player = self.players[self.current_player]
        self.current_player = (self.current_player + 1) % self.num_players
        return player

    def get_current_player(self) -> Player:
        """Get current player without advancing"""
        return self.players[self.current_player]


class MeldAnalyzer:
    @staticmethod
    def find_sets_and_runs(cards: List[Card]) -> Tuple[List[List[Card]], List[Card]]:
        remaining_cards = cards.copy()
        melds = []

        # Find sets (3+ cards of same rank)
        rank_groups = defaultdict(list)
        for card in remaining_cards:
            rank_groups[card.rank].append(card)

        for rank, group in rank_groups.items():
            if len(group) >= 3:
                # In multi-deck games, we can have multiple cards of same rank+deck_id
                set_meld = group[:3]
                melds.append(set_meld)
                for card in set_meld:
                    if card in remaining_cards:
                        remaining_cards.remove(card)

        # Find runs (3+ consecutive cards of same suit)
        suit_groups = defaultdict(list)
        for card in remaining_cards:
            suit_groups[card.suit].append(card)

        for suit, group in suit_groups.items():
            # Sort by value, then by deck_id to handle duplicates
            group.sort(key=lambda x: (x.value, x.deck_id))
            i = 0
            while i < len(group):
                run_start = i
                while (
                    i + 1 < len(group)
                    and group[i + 1].value == group[i].value + 1
                    and group[i + 1].deck_id == group[i].deck_id
                ):  # Same deck for runs
                    i += 1
                if i - run_start >= 2:
                    run_meld = group[run_start : i + 1]
                    melds.append(run_meld)
                    for card in run_meld:
                        if card in remaining_cards:
                            remaining_cards.remove(card)
                i += 1

        return melds, remaining_cards
