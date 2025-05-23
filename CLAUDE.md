# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Playing Against Trained Models (Web Interface)
```bash
# Start the web server
python app.py

# Then open http://localhost:5000 in your browser
```

### Training New Models
```bash
python tictactoe.py
```

### Setting up the Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

This is a TicTacToe game implementation with Q-Learning agents. The main components are:

1. **TicTacToe Game Engine** (`TicTacToe` class): Core game logic with board management, move validation, and win detection

2. **Player Types**:
   - `HumanPlayer`: Interactive player for human input
   - `RandomPlayer`: Makes random moves
   - `SmartRandomPlayer`: Random player with probabilistic smart moves (can detect winning/blocking moves)
   - `MinimaxPlayer`: Perfect player using minimax algorithm
   - `QLearningAgent`: Reinforcement learning agent that trains via Q-learning

3. **Training System**: 
   - `diverse_training()`: Multi-phase training approach that trains Q-learning agents against various opponent types
   - Training phases include: self-play, playing against random/smart random opponents, and final polishing
   - Trained models are saved as pickle files (`diverse_q_table_X.pkl`, `diverse_q_table_O.pkl`)

4. **Evaluation System**:
   - `evaluate_against_variety()`: Tests agents against multiple opponent types
   - `evaluate_against_minimax()`: Specific testing against perfect minimax player

## Key Implementation Details

- Q-Learning agents use epsilon-greedy exploration during training
- Board state is represented as a tuple for Q-table indexing
- The main script includes training, evaluation, and interactive play modes
- Pre-trained Q-tables are loaded for human vs AI gameplay

## Dependencies

The project has minimal dependencies:
- **Flask** (3.0.0) - Required only for the web interface (`app.py`)
- **Python Standard Library** - The core game and training (`tictactoe.py`) only uses:
  - `random` - For random moves and epsilon-greedy exploration
  - `pickle` - For saving/loading Q-tables
  - `math` - For minimax calculations

Note: The console interfaces (`tictactoe.py` and `play_game.py`) work with just a standard Python installation.