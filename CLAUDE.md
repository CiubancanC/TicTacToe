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
- `run_game.py`: Main launcher script with mode selection
- `scripts/train_improved_ai.py`: Script to train the improved DQN agent
- `scripts/play_vs_ai.py`: Script to play against the trained DQN AI
- `scripts/play_vs_perfect_ai.py`: Script to play against the perfect Minimax AI

### Data and Documentation
- `models/`: Directory where trained models are saved (keeps latest 3 + final)
- `visualizations/`: Training plots and performance charts
- `docs/`: Additional documentation and README variants

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

2. **Use the run_game.py script with different modes**:
```bash
python3 run_game.py [mode]
```
Modes:
- 1: Train AI agent (DQN)
- 2: Play against DQN AI
- 3: Human vs Human
- 4: Play against Perfect AI (Minimax)

3. **Train the improved DQN AI**:
```bash
python3 scripts/train_improved_ai.py
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

The improved DQN agent uses these hyperparameters:
- Learning rate: 0.0005 with adaptive scheduling
- Batch size: 128
- Epsilon decay: 0.997 (slower exploration decay)
- Gamma (discount factor): 0.99
- Target network update frequency: Every 5 steps
- Prioritized replay with alpha=0.6, beta=0.4

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