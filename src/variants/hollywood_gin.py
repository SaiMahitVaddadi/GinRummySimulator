#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from game_logic import Game, Player, Card, MeldAnalyzer


class HollywoodGin:
    def __init__(self):
        self.deck = Deck()
        self.players = [Player("Player 1"), Player("Player 2")]
        self.discard_pile = []
        self.analyzer = MeldAnalyzer()
        self.current_player = 0
        self.game_over = False
        self.winner = None
        # Hollywood Gin uses a 3-game scoring system
        self.game_scores = {
            "Player 1": {"game1": 0, "game2": 0, "game3": 0, "total": 0},
            "Player 2": {"game1": 0, "game2": 0, "game3": 0, "total": 0},
        }
        self.current_game = 1

    def setup_game(self):
        """Deal 10 cards to each player"""
        for player in self.players:
            player.hand = []  # Clear hand for new game
            for _ in range(10):
                player.add_card(self.deck.draw_one())

        # Start discard pile with one card
        if not self.discard_pile:  # First game
            self.discard_pile.append(self.deck.draw_one())
        else:  # Subsequent games keep discard pile
            pass

    def simulate_turn(self, player_num: int = None):
        """Simulate one turn for specified player or current player"""
        if player_num is None:
            player_num = self.current_player

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

            # Score based on current game in the series
            game_key = f"game{self.current_game}"
            opponent = self.players[1 - self.current_player]

            # Calculate opponent's deadwood
            opp_melds, opp_deadwood = self.analyzer.find_sets_and_runs(opponent.hand)
            opp_deadwood_value = sum(card.value for card in opp_deadwood)

            gin_bonus = 25
            self.game_scores[player.name][game_key] += gin_bonus + opp_deadwood_value
            self.game_scores[player.name]["total"] += gin_bonus + opp_deadwood_value

            return f"{player.name} GIN! Winner of Game {self.current_game}!"
        elif deadwood_value <= 10:
            # Knock
            self.winner = player
            self.game_over = True

            # Score based on current game in the series
            game_key = f"game{self.current_game}"
            opponent = self.players[1 - self.current_player]

            # Calculate opponent's deadwood
            opp_melds, opp_deadwood = self.analyzer.find_sets_and_runs(opponent.hand)
            opp_deadwood_value = sum(card.value for card in opp_deadwood)

            if deadwood_value < opp_deadwood_value:
                # Successful knock
                score_diff = opp_deadwood_value - deadwood_value
                self.game_scores[player.name][game_key] += 10 + score_diff
                self.game_scores[player.name]["total"] += 10 + score_diff
            else:
                # Undercut
                self.game_scores[opponent.name][game_key] += 20
                self.game_scores[opponent.name]["total"] += 20

            return f"{player.name} knocks with {deadwood_value} deadwood!"

        # Discard random card
        discard_card = random.choice(player.hand)
        player.remove_card(discard_card)
        self.discard_pile.append(discard_card)

        return f"{player.name} {action} and discarded {discard_card}"

    def play_game(self, max_turns: int = 50):
        """Play one game in the Hollywood Gin series"""
        self.game_over = False
        self.winner = None
        self.setup_game()

        print(f"Playing Game {self.current_game} of Hollywood Gin series")
        print(f"Player 1: {[str(card) for card in self.players[0].hand]}")
        print(f"Player 2: {[str(card) for card in self.players[1].hand]}")
        print(f"Top of discard pile: {self.discard_pile[-1]}")
        print("-" * 50)

        turn = 0
        while turn < max_turns and not self.game_over:
            player = self.players[self.current_player]
            result = self.simulate_turn()
            print(f"Turn {turn + 1}: {result}")

            if self.game_over:
                print(f"\nGame {self.current_game} Over! Winner: {self.winner.name}")
                print(f"Series scores: {self.game_scores}")
                return self.winner

            self.current_player = 1 - self.current_player
            turn += 1

        print(f"\nGame {self.current_game} ended after {turn} turns (no knock/gin)")
        return None

    def play_series(self, max_turns_per_game: int = 50):
        """Play the full 3-game Hollywood Gin series"""
        print("=" * 60)
        print("STARTING HOLLYWOOD GIN SERIES (Best of 3 Games)")
        print("=" * 60)

        while self.current_game <= 3:
            self.play_game(max_turns_per_game)

            # Check for reset between games
            if self.current_game < 3:
                self.discard_pile = []  # Clear for next game
                self.deck.reset()

            self.current_game += 1

        # Determine final winner
        p1_total = self.game_scores["Player 1"]["total"]
        p2_total = self.game_scores["Player 2"]["total"]

        final_winner = "Player 1" if p1_total > p2_total else "Player 2"

        print("\n" + "=" * 60)
        print("HOLLYWOOD GIN SERIES COMPLETE")
        print("=" * 60)
        print("Final Scores:")
        print(f"Player 1: {p1_total} points")
        print(f"Player 2: {p2_total} points")
        print(f"Series Winner: {final_winner}")
        print("=" * 60)

        return final_winner


def simulate_hollywood_series():
    """Run a complete Hollywood Gin series"""
    game = HollywoodGin()
    return game.play_series()


if __name__ == "__main__":
    simulate_hollywood_series()
