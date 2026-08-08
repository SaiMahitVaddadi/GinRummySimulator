#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from game_logic import Game, Player, Card, MeldAnalyzer


class OklahomaGin(Game):
    def __init__(self, num_players: int = 2):
        super().__init__(num_players)
        self.knock_limit = None  # Set by first discard

    def setup_game(self):
        """Deal 10 cards to each player, set knock limit from first discard"""
        for player in self.players:
            for _ in range(10):
                player.add_card(self.deck.draw_one())

        # First card sets the knock limit
        first_card = self.deck.draw_one()
        self.discard_pile.append(first_card)

        if first_card.rank == "A":
            self.knock_limit = 0  # Only gin allowed
        elif first_card.rank in ["J", "Q", "K"]:
            self.knock_limit = 10
        else:
            self.knock_limit = int(first_card.rank)


def simulate_turn(self, player_num: int = None):
    """Simulate one turn for specified player or current player"""
    if player_num is None:
        player = self.get_current_player()
    else:
        player = self.players[player_num]

    # Draw card (randomly choose from deck or discard pile)
    if random.choice([True, False]) and self.discard_pile:
        drawn_card = self.discard_pile.pop()
        action = "drew from discard pile"
    else:
        drawn_card = self.deck.draw_one()
        action = "drew from deck"

    player.add_card(drawn_card)

    # Check if player can knock or gin (Oklahoma rules)
    melds, deadwood = self.analyzer.find_sets_and_runs(player.hand)
    deadwood_value = sum(card.value for card in deadwood)

    if deadwood_value == 0:
        # Gin!
        self.winner = player
        self.game_over = True
        self.scores[player.name] += 25

        # Calculate other players' deadwood for scoring
        total_opponent_deadwood = 0
        for other_player in self.players:
            if other_player != player:
                opp_melds, opp_deadwood = self.analyzer.find_sets_and_runs(
                    other_player.hand
                )
                total_opponent_deadwood += sum(card.value for card in opp_deadwood)

        self.scores[player.name] += total_opponent_deadwood
        return f"{player.name} GIN! Winner!"
    elif deadwood_value <= self.knock_limit and self.knock_limit > 0:
        # Knock (if allowed) - check all other players
        self.winner = player
        self.game_over = True

        # Calculate if knockout is successful
        for other_player in self.players:
            if other_player != player:
                opp_melds, opp_deadwood = self.analyzer.find_sets_and_runs(
                    other_player.hand
                )
                opp_deadwood_value = sum(card.value for card in opp_deadwood)

                if opp_deadwood_value <= deadwood_value:
                    # Undercut - other player wins this round
                    self.scores[other_player.name] += 20
                    return f"{player.name} knocked with {deadwood_value} deadwood, but was undercut by {other_player.name}!"

        # Successful knock
        knock_points = 10
        total_opponent_deadwood = 0
        for other_player in self.players:
            if other_player != player:
                opp_melds, opp_deadwood = self.analyzer.find_sets_and_runs(
                    other_player.hand
                )
                opp_deadwood_value = sum(card.value for card in opp_deadwood)
                total_opponent_deadwood += max(0, opp_deadwood_value - deadwood_value)

        self.scores[player.name] += knock_points + total_opponent_deadwood
        return f"{player.name} knocks with {deadwood_value} deadwood (limit: {self.knock_limit})!"

    # Discard random card
    discard_card = random.choice(player.hand)
    player.remove_card(discard_card)
    self.discard_pile.append(discard_card)

    # Advance to next player
    if player_num is None:
        self.get_next_player()

    return f"{player.name} {action} and discarded {discard_card}"


def play_game(self, max_turns: int = 50):
    """Play a complete Oklahoma Gin game"""
    self.setup_game()

    print(f"Starting Oklahoma Gin with {self.num_players} players")
    print(f"Using {self.num_decks} deck(s) ({self.deck.total_cards()} total cards)")
    print(
        f"Knock limit set to: {self.knock_limit} (from first discard: {self.discard_pile[0]})"
    )

    for i, player in enumerate(self.players):
        print(f"{player.name}: {[str(card) for card in player.hand]}")
    print("-" * 50)

    turn = 0
    while turn < max_turns and not self.game_over:
        result = self.simulate_turn()
        print(f"Turn {turn + 1}: {result}")

        if self.game_over:
            print(f"\nGame Over! Winner: {self.winner.name}")
            print(f"Final scores: {self.scores}")
            return self.winner

        turn += 1

    print(f"\nGame ended after {turn} turns (no knock/gin)")
    return None


import random


def simulate_oklahoma_game():
    """Run a single Oklahoma Gin game"""
    game = OklahomaGin()
    return game.play_game()


if __name__ == "__main__":
    simulate_oklahoma_game()
