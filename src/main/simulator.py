#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from variants.classic_gin import ClassicGinRummy
from variants.oklahoma_gin import OklahomaGin
from variants.hollywood_gin import HollywoodGin
from variants.indian_rummy import (
    ThreeCardIndianRummy,
    SevenCardIndianRummy,
    TenCardIndianRummy,
    FifteenCardIndianRummy,
)


def print_menu():
    print("\n" + "=" * 60)
    print("🎴 GIN RUMMY & INDIAN RUMMY SIMULATOR 🎴")
    print("=" * 60)
    print("Choose a variant to simulate:")
    print("\n🎯 GIN RUMMY VARIANTS:")
    print("1. Classic Gin Rummy")
    print("2. Oklahoma Gin")
    print("3. Hollywood Gin (3-game series)")
    print("\n🇮🇳 INDIAN RUMMY VARIANTS:")
    print("4. 3-Card Indian Rummy")
    print("5. 7-Card Indian Rummy")
    print("6. 10-Card Indian Rummy")
    print("7. 15-Card Indian Rummy")
    print("\n🎮 MULTI-PLAYER OPTIONS:")
    print("8. Run all variants")
    print("9. Run multiple simulations")
    print("0. Exit")
    print("-" * 60)


def simulate_classic():
    print("\n🎯 CLASSIC GIN RUMMY SIMULATION")
    print("=" * 40)
    game = ClassicGinRummy()
    return game.play_game()


def simulate_oklahoma():
    print("\n🎯 OKLAHOMA GIN SIMULATION")
    print("=" * 40)
    game = OklahomaGin()
    return game.play_game()


def simulate_hollywood():
    print("\n🎯 HOLLYWOOD GIN SIMULATION")
    print("=" * 40)
    game = HollywoodGin()
    return game.play_series()


def simulate_3card_indian():
    print("\n🇮🇳 3-CARD INDIAN RUMMY SIMULATION")
    print("=" * 40)
    game = ThreeCardIndianRummy(6)  # Default 6 players for fast game
    return game.play_game()


def simulate_7card_indian():
    print("\n🇮🇳 7-CARD INDIAN RUMMY SIMULATION")
    print("=" * 40)
    game = SevenCardIndianRummy(4)  # Default 4 players
    return game.play_game()


def simulate_10card_indian():
    print("\n🇮🇳 10-CARD INDIAN RUMMY SIMULATION")
    print("=" * 40)
    game = TenCardIndianRummy(4)  # Default 4 players
    return game.play_game()


def simulate_15card_indian():
    print("\n🇮🇳 15-CARD INDIAN RUMMY SIMULATION")
    print("=" * 40)
    game = FifteenCardIndianRummy(3)  # Default 3 players for longer game
    return game.play_game()


def run_all_variants():
    print("\n🎯 RUNNING ALL GIN RUMMY & INDIAN RUMMY VARIANTS")
    print("=" * 60)

    print("\n1️⃣  Classic Gin Rummy")
    classic_winner = simulate_classic()

    print("\n" + "=" * 50)
    print("\n2️⃣  Oklahoma Gin")
    oklahoma_winner = simulate_oklahoma()

    print("\n" + "=" * 50)
    print("\n3️⃣  Hollywood Gin")
    hollywood_winner = simulate_hollywood()

    print("\n" + "=" * 50)
    print("\n4️⃣  3-Card Indian Rummy")
    indian3_winner = simulate_3card_indian()

    print("\n" + "=" * 50)
    print("\n5️⃣  7-Card Indian Rummy")
    indian7_winner = simulate_7card_indian()

    print("\n" + "=" * 50)
    print("\n6️⃣  10-Card Indian Rummy")
    indian10_winner = simulate_10card_indian()

    print("\n" + "=" * 50)
    print("\n7️⃣  15-Card Indian Rummy")
    indian15_winner = simulate_15card_indian()

    print("\n" + "=" * 60)
    print("📊 SUMMARY OF ALL VARIANTS")
    print("=" * 60)
    print(
        f"Classic Gin Rummy Winner: {classic_winner.name if classic_winner else 'No winner'}"
    )
    print(
        f"Oklahoma Gin Winner: {oklahoma_winner.name if oklahoma_winner else 'No winner'}"
    )
    print(
        f"Hollywood Gin Winner: {hollywood_winner if hollywood_winner else 'No winner'}"
    )
    print(
        f"3-Card Indian Rummy Winner: {indian3_winner.name if indian3_winner else 'No winner'}"
    )
    print(
        f"7-Card Indian Rummy Winner: {indian7_winner.name if indian7_winner else 'No winner'}"
    )
    print(
        f"10-Card Indian Rummy Winner: {indian10_winner.name if indian10_winner else 'No winner'}"
    )
    print(
        f"15-Card Indian Rummy Winner: {indian15_winner.name if indian15_winner else 'No winner'}"
    )


def run_multiple_simulations():
    print("\n🎯 MULTIPLE SIMULATIONS")
    print("=" * 40)

try:
        num_simulations = int(input("Number of simulations to run (1-100): ") or "10")
        num_simulations = max(1, min(100, num_simulations))
        variant = input("Variant (1-7=Variants): ") or "1"
    except ValueError:
        print("Invalid input. Running 10 Classic Gin simulations.")
        num_simulations = 10
        variant = "1"
    
    variant_map = {
        "1": ("Classic Gin Rummy", lambda: ClassicGinRummy()),
        "2": ("Oklahoma Gin", lambda: OklahomaGin()),
        "3": ("Hollywood Gin", lambda: HollywoodGin()),
        "4": ("3-Card Indian Rummy", lambda: ThreeCardIndianRummy(4)),
        "5": ("7-Card Indian Rummy", lambda: SevenCardIndianRummy(4)),
        "6": ("10-Card Indian Rummy", lambda: TenCardIndianRummy(4)),
        "7": ("15-Card Indian Rummy", lambda: FifteenCardIndianRummy(3))
    }

    variant_name, game_class = variant_map.get(
        variant, ("Classic Gin Rummy", lambda: ClassicGinRummy())
    )

    print(f"\nRunning {num_simulations} simulations of {variant_name}...")
    print("=" * 60)

    p1_wins = 0
    p2_wins = 0
    no_winner = 0

    for i in range(num_simulations):
        print(f"\nSimulation {i + 1}/{num_simulations}")

        game = game_class()

        if variant == "3":  # Hollywood Gin
            winner = game.play_series()
            if winner:
                if "Player 1" in winner:
                    p1_wins += 1
                else:
                    p2_wins += 1
        else:  # Classic or Oklahoma
            winner = game.play_game()
            if winner:
                if winner.name == "Player 1":
                    p1_wins += 1
                else:
                    p2_wins += 1
            else:
                no_winner += 1

    print("\n" + "=" * 60)
    print("📈 SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total Simulations: {num_simulations}")
    print(f"Player 1 Wins: {p1_wins} ({p1_wins / num_simulations * 100:.1f}%)")
    print(f"Player 2 Wins: {p2_wins} ({p2_wins / num_simulations * 100:.1f}%)")
    if variant != "3":
        print(f"No Winner: {no_winner} ({no_winner / num_simulations * 100:.1f}%)")
    print("=" * 60)


def main():
    """Main simulator menu"""
    while True:
        print_menu()

        try:
            choice = input("Enter your choice (0-5): ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            simulate_classic()
        elif choice == "2":
            simulate_oklahoma()
        elif choice == "3":
            simulate_hollywood()
        elif choice == "4":
            run_all_variants()
        elif choice == "5":
            run_multiple_simulations()
        else:
            print("❌ Invalid choice. Please enter 0-5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # If run with arguments, handle command-line mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--classic":
            simulate_classic()
        elif sys.argv[1] == "--oklahoma":
            simulate_oklahoma()
        elif sys.argv[1] == "--hollywood":
            simulate_hollywood()
        elif sys.argv[1] == "--all":
            run_all_variants()
        else:
            print("Usage: python simulator.py [--classic|--oklahoma|--hollywood|--all]")
    else:
        main()
