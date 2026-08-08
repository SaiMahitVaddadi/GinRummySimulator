#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from game_logic import Game, Player, Card, MeldAnalyzer


class ClassicGinRummy(Game):
    def __init__(self, num_players: int = 2):
        super().__init__(num_players)
        self.knock_limit = 10  # Classic Gin has 10-point knock limit

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

        # Check if player can knock or gin
        melds, deadwood = self.analyzer.find_sets_and_runs(player.hand)
        deadwood_value = sum(card.value for card in deadwood)

        if deadwood_value == 0:
            # Gin!
            self.winner = player
            self.game_over = True
            self.scores[player.name] += 25 + deadwood_value

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
        elif deadwood_value <= self.knock_limit:
            # Knock - check all other players
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
                    total_opponent_deadwood += max(
                        0, opp_deadwood_value - deadwood_value
                    )

            self.scores[player.name] += knock_points + total_opponent_deadwood
            return f"{player.name} knocks with {deadwood_value} deadwood!"

        # Discard random card
        discard_card = random.choice(player.hand)
        player.remove_card(discard_card)
        self.discard_pile.append(discard_card)

        # Advance to next player
        if player_num is None:
            self.get_next_player()

        return f"{player.name} {action} and discarded {discard_card}"

    def play_game(self, max_turns: int = 50):
        """Play a complete game"""
        self.setup_game()

        print(f"Starting Classic Gin Rummy with {self.num_players} players")
        print(f"Using {self.num_decks} deck(s) ({self.deck.total_cards()} total cards)")

        for i, player in enumerate(self.players):
            print(f"{player.name}: {[str(card) for card in player.hand]}")
        print(f"Top of discard pile: {self.discard_pile[-1]}")
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


def simulate_classic_game(num_players: int = 2):
    """Run a single Classic Gin Rummy game"""
    game = ClassicGinRummy(num_players)
    return game.play_game()


if __name__ == "__main__":
    # Test with 4 players by default
    simulate_classic_game(4)
