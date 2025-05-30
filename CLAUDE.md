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

#### Original Implementation (Pure Python)
```bash
# Train the original Q-learning agents
python tictactoe_original.py

# Or use the current default version
python tictactoe.py
```

#### Improved Implementation (With Neural Network Best Practices)
```bash
# Train the enhanced Q-learning agents with experience replay, batch learning, etc.
python tictactoe_improved.py
```

### Benchmarking and Comparison
```bash
# Run full benchmark comparison (10,000 episodes)
python benchmark_comparison.py

# Run quick benchmark comparison (1,000 episodes)
python benchmark_comparison.py --quick

# Results will be saved to:
# - benchmark_results_<timestamp>.json (detailed metrics)
# - convergence_comparison.png (visualization)
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

This is a TicTacToe game implementation with Q-Learning agents. The project now includes two implementations:

### Original Implementation (`tictactoe_original.py`, `tictactoe.py`)
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

### Improved Implementation (`tictactoe_improved.py`)
Enhanced version incorporating neural network best practices:

1. **Numpy-based Board Representation**: Uses numpy arrays for vectorized operations and faster computation

2. **Enhanced Q-Learning Agent**:
   - **Experience Replay Buffer**: Stores past experiences and samples mini-batches for more stable learning
   - **Batch Learning**: Updates Q-values in batches rather than single steps
   - **Learning Rate Scheduling**: Supports exponential, step, and cosine annealing schedules
   - **Q-value Normalization**: Prevents numerical instability through clipping and normalization
   - **Efficient State Hashing**: Integer-based board hashing for memory efficiency

3. **Improved Training**:
   - More stable convergence through experience replay
   - Better sample efficiency
   - Configurable hyperparameters (batch size, buffer capacity, scheduler type)

### Benchmarking System (`benchmark_comparison.py`, `benchmark_framework.py`)
- Compares performance between original and improved implementations
- Measures training time, memory usage, convergence speed, and final performance
- Generates detailed reports and visualizations

## Key Implementation Details

- Q-Learning agents use epsilon-greedy exploration during training
- Board state is represented as a tuple for Q-table indexing
- The main script includes training, evaluation, and interactive play modes
- Pre-trained Q-tables are loaded for human vs AI gameplay

## Dependencies

The project has minimal dependencies:
- **Flask** (3.0.0) - Required only for the web interface (`app.py`)
- **NumPy** (1.24.3) - Required for the improved implementation (`tictactoe_improved.py`)
- **Python Standard Library** - The core game and training uses:
  - `random` - For random moves and epsilon-greedy exploration
  - `pickle` - For saving/loading Q-tables
  - `math` - For minimax calculations
  - `collections` - For deque in experience replay buffer
  - `time` - For benchmarking
  - `json` - For saving benchmark results

Optional dependencies:
- **matplotlib** - For generating convergence comparison plots (benchmark_comparison.py)

Note: The original implementation (`tictactoe_original.py`) works with just a standard Python installation.