#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from game_logic import Player, Card, MeldAnalyzer
from typing import List, Optional


class IndianDeck:
    """Enhanced deck for Indian Rummy with jokers and multiple stacks"""

    def __init__(self, num_decks: int = 1, include_jokers: bool = True):
        self.suits = ["♠", "♥", "♦", "♣"]
        self.ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        self.num_decks = num_decks
        self.include_jokers = include_jokers
        self.cards = []
        self.joker_cards = []
        self.reset()

    def reset(self):
        """Reset deck with specified number of standard 52-card decks plus jokers"""
        self.cards = []
        self.joker_cards = []

        for deck_id in range(self.num_decks):
            # Add regular cards
            for suit in self.suits:
                for rank in self.ranks:
                    self.cards.append(Card(rank, suit, deck_id))

            # Add jokers (2 per deck in Indian Rummy)
            if self.include_jokers:
                self.joker_cards.append(Card("JKR", "🃏", deck_id))
                self.joker_cards.append(Card("JKR", "🎭", deck_id))

        # Combine all cards and shuffle
        all_cards = self.cards + self.joker_cards
        random.shuffle(all_cards)

        # Split into draw pile and joker pile
        self.draw_pile = all_cards[: -len(self.joker_cards) // 2]  # Main draw pile
        self.joker_pile = all_cards[-len(self.joker_cards) // 2 :]  # Joker stack

    def draw(self, num_cards: int = 1, from_joker_pile: bool = False) -> List[Card]:
        """Draw cards from specified pile"""
        pile = self.joker_pile if from_joker_pile else self.draw_pile
        drawn = []
        for _ in range(num_cards):
            if pile:
                drawn.append(pile.pop())
        return drawn

    def draw_one(self, from_joker_pile: bool = False) -> Optional[Card]:
        """Draw one card from specified pile"""
        pile = self.joker_pile if from_joker_pile else self.draw_pile
        if pile:
            return pile.pop()
        return None

    def cards_remaining(self, include_jokers: bool = True) -> int:
        count = len(self.draw_pile)
        if include_jokers:
            count += len(self.joker_pile)
        return count

    def total_cards(self) -> int:
        jokers = len(self.joker_cards) if self.include_jokers else 0
        return self.num_decks * 52 + jokers


class IndianRummyPlayer(Player):
    """Enhanced player for Indian Rummy"""

    def __init__(self, name: str, player_id: int = 0):
        super().__init__(name, player_id)
        self.pure_sequences = []  # Pure sequences (no joker)
        self.impure_sequences = []  # Sequences with joker
        self.sets = []  # Sets of same rank

    def get_valid_melds(self, wild_card: Optional[Card] = None):
        """Get all valid melds including wild card usage"""
        # For Indian Rummy: need at least one pure sequence
        melds, remaining = MeldAnalyzer.find_sets_and_runs(self.hand)

        # Mark jokers and wild cards
        for card in self.hand:
            if card.is_joker or (
                wild_card
                and card.rank == wild_card.rank
                and card.suit == wild_card.suit
            ):
                card.is_wild = True

        return melds, remaining

    def can_show(self, wild_card: Optional[Card] = None) -> bool:
        """Check if player has valid melds to show"""
        melds, remaining = self.get_valid_melds(wild_card)

        # Need at least one pure sequence and all cards should be in melds
        has_pure_sequence = False
        for meld in melds:
            if self._is_pure_sequence(meld):
                has_pure_sequence = True
                break

        return has_pure_sequence and len(remaining) == 0

    def _is_pure_sequence(self, meld: List[Card]) -> bool:
        """Check if meld is a pure sequence (no joker, same suit, consecutive)"""
        if len(meld) < 3:
            return False

        # No jokers in pure sequence
        if any(card.is_joker for card in meld):
            return False

        # Check if all same suit and consecutive
        suits = [card.suit for card in meld]
        if len(set(suits)) != 1:
            return False

        values = sorted([card.value for card in meld])
        for i in range(1, len(values)):
            if values[i] != values[i - 1] + 1:
                return False

        return True


class IndianRummy:
    """Base class for Indian Rummy variants"""

    def __init__(
        self, num_players: int = 2, cards_per_player: int = 10, num_decks: int = 2
    ):
        self.num_players = max(2, min(num_players, 20))  # Support 2-20 players
        self.cards_per_player = cards_per_player
        self.num_decks = num_decks

        # Enhanced deck for Indian Rummy
        self.deck = IndianDeck(num_decks, include_jokers=True)
        self.players = [
            IndianRummyPlayer(f"Player {i + 1}", i) for i in range(self.num_players)
        ]

        self.discard_pile = []
        self.analyzer = MeldAnalyzer()
        self.current_player = 0
        self.game_over = False
        self.winner = None
        self.scores = {player.name: 0 for player in self.players}
        self.wild_card = None
        self.round_number = 1

    def setup_game(self):
        """Setup game with Indian Rummy rules"""
        # Deal cards to each player
        for player in self.players:
            for _ in range(self.cards_per_player):
                card = self.deck.draw_one()
                if card:
                    player.add_card(card)

        # Set wild card (random from remaining deck)
        self.wild_card = self.deck.draw_one()

        # Start discard pile with one card
        discard_card = self.deck.draw_one()
        if discard_card:
            self.discard_pile.append(discard_card)

    def simulate_turn(self, player_num: int = None):
        """Simulate one turn for specified player or current player"""
        if player_num is None:
            player = self.players[self.current_player]
        else:
            player = self.players[player_num]

        # Draw card (randomly choose from deck, discard, or joker pile)
        draw_choice = random.choice(["deck", "discard", "joker"])

        if draw_choice == "discard" and self.discard_pile:
            drawn_card = self.discard_pile.pop()
            action = "drew from discard pile"
        elif draw_choice == "joker" and self.deck.joker_pile:
            drawn_card = self.deck.draw_one(from_joker_pile=True)
            action = "drew from joker pile"
        else:
            drawn_card = self.deck.draw_one()
            action = "drew from main deck"

        if drawn_card:
            player.add_card(drawn_card)

        # Check if player can show (valid melds)
        if player.can_show(self.wild_card):
            self.winner = player
            self.game_over = True

            # Calculate score
            melds, remaining = player.get_valid_melds(self.wild_card)
            deadwood_points = sum(card.value for card in remaining)
            self.scores[player.name] = max(
                0, 100 - deadwood_points
            )  # Winner gets points

            return f"{player.name} SHOWS! Wins with {len(melds)} melds!"

        # Discard random card
        if player.hand:
            discard_card = random.choice(player.hand)
            player.remove_card(discard_card)
            self.discard_pile.append(discard_card)
            return f"{player.name} {action} and discarded {discard_card}"

        return f"{player.name} {action} (no discard)"

    def play_game(self, max_turns: int = 100):
        """Play a complete Indian Rummy game"""
        self.setup_game()

        print(f"Starting Indian Rummy ({self.cards_per_player}-card)")
        print(f"Players: {self.num_players}, Decks: {self.num_decks}")
        print(f"Wild Card: {self.wild_card}")
        print(
            f"Cards in deck: {self.deck.total_cards()}, Jokers: {len(self.deck.joker_cards)}"
        )

        for i, player in enumerate(self.players):
            print(
                f"{player.name}: {[str(card) for card in player.hand[:5]]}...({len(player.hand)} cards)"
            )
        print(f"Top discard: {self.discard_pile[-1] if self.discard_pile else 'None'}")
        print("-" * 60)

        turn = 0
        while turn < max_turns and not self.game_over:
            result = self.simulate_turn()
            print(
                f"Turn {turn + 1} ({self.players[self.current_player].name}): {result}"
            )

            if self.game_over:
                print(f"\nGame Over! Winner: {self.winner.name}")
                print(f"Final scores: {self.scores}")
                return self.winner

            self.current_player = (self.current_player + 1) % self.num_players
            turn += 1

        print(f"\nGame ended after {turn} turns (no winner)")
        return None


# Game variant classes
class ThreeCardIndianRummy(IndianRummy):
    def __init__(self, num_players: int = 2):
        super().__init__(num_players, cards_per_player=3, num_decks=1)


class SevenCardIndianRummy(IndianRummy):
    def __init__(self, num_players: int = 2):
        super().__init__(num_players, cards_per_player=7, num_decks=2)


class TenCardIndianRummy(IndianRummy):
    def __init__(self, num_players: int = 2):
        super().__init__(num_players, cards_per_player=10, num_decks=2)


class FifteenCardIndianRummy(IndianRummy):
    def __init__(self, num_players: int = 2):
        super().__init__(num_players, cards_per_player=15, num_decks=3)
