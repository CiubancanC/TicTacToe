# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Tic Tac Toe game with multiple AI implementations. The project features:
- A playable Tic Tac Toe game with PyGame
- A Deep Q-Network (DQN) agent that learns to play through self-play
- A perfect Minimax AI that never loses
- Multiple gameplay modes: AI training, Human vs AI, and Human vs Human

## Project Structure

### Core Source Code
- `src/game/tictactoe.py`: Contains the core game logic and GUI implementation
  - `TicTacToe`: Game logic class that manages board state, moves, and win conditions
  - `TicTacToeGUI`: PyGame interface for the game
- `src/ai/agent.py`: Implements the reinforcement learning agent
  - `DQN`: Neural network model architecture 
  - `PrioritizedReplayBuffer`: Memory storage with prioritized experience replay
  - `DQNAgent`: Agent that learns using Double DQN with prioritized experience replay
- `src/ai/minimax.py`: Implements the perfect Minimax AI
  - `MinimaxAgent`: Agent that uses the minimax algorithm with alpha-beta pruning
  - Support functions for board evaluation and move selection
- `src/main.py`: Main script with game modes and training loop

### Scripts and Utilities
- `run_game.py`: **ENHANCED** Interactive showcase with AI vs AI battles
- `scripts/train_improved_ai.py`: Script to train the original DQN agent
- `scripts/train_enhanced_ai.py`: **NEW** Enhanced training with strategic rewards & self-play archives
- `scripts/train_strategic_showcase.py`: **NEW** Focused strategic training demonstration
- `scripts/test_enhanced_features.py`: Quick test script for enhanced features
- `scripts/ai_battle.py`: **NEW** Quick AI tournament and model comparison
- `scripts/evaluate_enhanced_ai.py`: **NEW** Strategic capability testing
- `scripts/play_vs_ai.py`: Script to play against the trained DQN AI
- `scripts/play_vs_perfect_ai.py`: Script to play against the perfect Minimax AI

### Data and Documentation
- `models/`: Directory where trained models are saved with **method annotations**
- `visualizations/`: Training plots and performance charts
- `docs/`: Additional documentation and README variants
  - `MODEL_ANNOTATIONS.md`: **NEW** Complete guide to training method naming system
- **🏷️ Model Naming**: Enhanced system clearly identifies training methods

## Common Commands

### Setup and Installation

Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
```

### Running the Game

There are several ways to run the game:

1. **Main script** (interactive menu):
```bash
python3 src/main.py
```

2. **Use the interactive run_game.py menu** (recommended):
```bash
python3 run_game.py
```
Features:
- 🤖 **Training Options**: Original DQN, Enhanced DQN, Quick test
- 🎯 **Play vs AI**: Select from trained models, Perfect Minimax AI  
- ⚔️ **AI vs AI Battles**: DQN tournaments, DQN vs Minimax competitions
- 👥 **Human Modes**: Human vs Human, Model comparison
- 📊 **Model Management**: List, select, and compare different trained models

Or use legacy command line modes:
```bash
python3 run_game.py [mode]
```
Legacy modes: 1 (Train), 2 (Play vs DQN), 3 (Human vs Human), 4 (Play vs Minimax)

3. **Train the DQN AI** (choose one):
```bash
# Original DQN training
python3 scripts/train_improved_ai.py

# Enhanced DQN with strategic rewards & self-play archives
python3 scripts/train_enhanced_ai.py

# Quick test of enhanced features
python3 scripts/test_enhanced_features.py

# AI vs AI battles
python3 scripts/ai_battle.py
```

4. **Play against the trained DQN AI**:
```bash
python3 scripts/play_vs_ai.py
```

5. **Play against the perfect Minimax AI**:
```bash
python3 scripts/play_vs_perfect_ai.py
```

### AI Implementations

#### DQN Reinforcement Learning Agent

The enhanced DQN agent includes:

**Core Hyperparameters:**
- Learning rate: 0.0005 with adaptive scheduling
- Batch size: 128
- Epsilon decay: 0.997 (slower exploration decay)
- Gamma (discount factor): 0.99
- Target network update frequency: Every 5 steps
- Prioritized replay with alpha=0.6, beta=0.4

**🆕 Enhanced Reward Engineering:**
- **Win rewards**: 10 (vs previous 1) - encourages aggressive winning
- **Draw rewards**: 1 (vs previous 0.5) - values defensive play
- **Strategic move bonuses**: 
  - Center control: +0.5
  - Corner control: +0.3
  - Fork creation: +2.0 (two ways to win)
  - Blocking opponent wins: +1.5
  - Creating win threats: +1.0

**🆕 Self-Play with Historical Archives:**
- Maintains up to 5 historical opponent models
- Gradually increases historical opponent usage (30% → 70%)
- Weighted selection favoring recent opponents
- Archives models every 1000 episodes

#### Minimax Perfect AI

The Minimax agent:
- Uses the minimax algorithm with alpha-beta pruning
- Searches the entire game tree to find optimal moves
- Will never lose and always wins when possible
- Does not require training (deterministic algorithm)

### Project Dependencies

- Python 3.8+
- PyGame
- PyTorch
- NumPy
- Matplotlib

## Key Concepts

- Reinforcement Learning (DQN):
  - The DQN agent learns through trial and error
  - State representation is the flattened 3x3 board as input to the neural network
  - Actions are the 9 possible move positions
  - Rewards: +1 for winning, 0.5 for a draw, -1 for losing, -10 for invalid moves
  - Double DQN to reduce overestimation bias in Q-values
  - Prioritized Experience Replay to focus learning on important transitions

- Minimax Algorithm:
  - A perfect algorithm for zero-sum games like Tic Tac Toe
  - Recursively evaluates all possible future states
  - Assumes optimal play from both players
  - Alpha-beta pruning optimizes the search by skipping irrelevant branches
  - Guarantees optimal play (will never lose in Tic Tac Toe)