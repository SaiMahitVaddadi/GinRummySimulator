# 🎴 Gin Rummy Simulator

A comprehensive Gin Rummy simulator that implements multiple variants of the classic card game with random card dealing and automated gameplay.

## 🎯 Features

- **Classic Gin Rummy** - Traditional 10-card hands, knock with ≤10 deadwood
- **Oklahoma Gin** - Variable knock limit based on first discard
- **Hollywood Gin** - 3-game series with cumulative scoring
- **Automated gameplay** - Random but strategic card play simulation
- **Multiple simulation modes** - Single games or batch simulations
- **Interactive and command-line interfaces**

## 📁 Project Structure

```
GinRummySimulator/
├── src/
│   ├── game_logic.py          # Core game mechanics (Card, Deck, Player, Game classes)
│   ├── main/
│   │   ├── __init__.py
│   │   └── simulator.py       # Main simulator interface
│   ├── variants/
│   │   ├── __init__.py
│   │   ├── classic_gin.py     # Classic Gin Rummy variant
│   │   ├── oklahoma_gin.py    # Oklahoma Gin variant
│   │   └── hollywood_gin.py   # Hollywood Gin variant
│   └── __init__.py
├── run.py                     # Main entry point script
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- No external dependencies required (uses only Python standard library)

### Running the Simulator

#### Main Entry Point (Recommended)
```bash
python run.py
```

#### Direct Access
```bash
# Interactive mode
python src/main/simulator.py

# Command line mode
python src/main/simulator.py --classic      # Classic Gin Rummy
python src/main/simulator.py --oklahoma     # Oklahoma Gin  
python src/main/simulator.py --hollywood    # Hollywood Gin
python src/main/simulator.py --all          # All variants sequentially
```

#### Run Individual Variants
```bash
python src/variants/classic_gin.py      # Classic Gin Rummy game
python src/variants/oklahoma_gin.py     # Oklahoma Gin game
python src/variants/hollywood_gin.py     # Hollywood Gin series
```

## 🎮 Game Variants

### Classic Gin Rummy
- 10 cards dealt to each player
- Knock allowed with ≤10 deadwood points
- Gin (0 deadwood) awards 25 bonus points
- Supports 2-12 players with adaptive deck scaling
- Standard scoring system

### Oklahoma Gin
- Knock limit determined by first discard:
  - Ace: Only gin allowed (0 deadwood)
  - Face cards: 10 points
  - Number cards: Face value
- More strategic and unpredictable gameplay
- Supports 2-12 players with adaptive deck scaling

### Hollywood Gin
- 3-game series with cumulative scoring
- Points from each game add to running total
- Higher strategic complexity over multiple games
- Supports 2-12 players with adaptive deck scaling
- Winner determined by highest total score

## 🎯 Multi-Player Features

- **Adaptive Deck Scaling**: Automatically uses optimal number of decks based on player count
- **Smart Card Distribution**: Ensures fair card allocation across all players
- **Multi-Player Scoring**: Handles knock/gin outcomes across all opponents
- **Player Rotation**: Automatic turn progression through all players

### Player Count & Deck Scaling
- 2 players: 1 deck (52 cards)
- 3-4 players: 2 decks (104 cards)  
- 5-6 players: 3 decks (156 cards)
- 7+ players: Optimal scaling (capped at 8 decks)

## 🎲 Simulation Features

### Random Card Dealing
- Standard 52-card deck with proper shuffling
- Random but realistic draw/discard decisions
- Strategic meld detection (sets and runs)

### Automated Gameplay
- Players automatically:
  - Draw from deck or discard pile
  - Form melds (sets of 3-4 same rank, runs of 3+ consecutive same suit)
  - Calculate deadwood points
  - Decide when to knock or go gin
  - Discard strategically

### Batch Simulations
- Run multiple games for statistical analysis
- Win rate tracking for both players
- Configurable simulation count (1-100 games)

## 📊 Example Output

```
Starting Classic Gin Rummy
Player 1: ['K♠', '7♥', '4♦', '3♣', '9♠', '2♥', 'J♦', '8♣', '5♥', 'Q♠']
Player 2: ['A♦', '6♣', '9♦', '3♥', 'K♥', '7♠', '10♣', '4♠', '2♦', 'J♥']
Top of discard pile: 8♦
--------------------------------------------------
Turn 1: Player 1 drew from deck and discarded 5♥
Turn 2: Player 2 drew from discard pile and discarded 6♣
...
Turn 7: Player 1 GIN! Winner!

Game Over! Winner: Player 1
Final scores: {'Player 1': 25, 'Player 2': 0}
```

## 🎯 Game Rules Summary

### Objective
Arrange your 10 cards into melds while minimizing deadwood (unmatched cards).

### Melds
- **Sets**: 3-4 cards of same rank (e.g., 7♠ 7♥ 7♦)
- **Runs**: 3+ consecutive cards of same suit (e.g., 4♣ 5♣ 6♣)

### Winning
- **Gin**: All 10 cards in melds (0 deadwood)
- **Knock**: Deadwood ≤ limit (varies by variant)
- **Undercut**: Opponent knocks with higher deadwood than you

### Scoring
- Gin: 25 bonus + opponent's deadwood
- Knock: 10 bonus + deadwood difference
- Undercut: 20 points to defender

## 🔧 Technical Details

### Core Classes
- `Card`: Individual playing card with rank, suit, and value
- `Deck`: 52-card deck with shuffling and dealing
- `Player`: Hand management and meld analysis
- `MeldAnalyzer`: Detects sets, runs, and calculates deadwood

### Simulation Logic
- Probabilistic draw decisions
- Automatic meld detection
- Strategic knock/gin evaluation
- Configurable game parameters

## 🎨 Customization

The simulator is designed to be easily extensible:
- Modify knock limits in variant files
- Adjust simulation parameters
- Add new Gin Rummy variants
- Implement different AI strategies

## 📈 Use Cases

- Learn Gin Rummy rules and strategies
- Test different game variant mechanics
- Statistical analysis of game outcomes
- Educational tool for card game theory
- Foundation for more advanced AI implementations

## 🤝 Contributing

Feel free to extend the simulator with:
- Additional Gin Rummy variants
- More sophisticated AI strategies
- GUI interface
- Network multiplayer support
- Advanced statistical analysis tools

---

**Enjoy your Gin Rummy simulations! 🎴**