# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-file TicTacToe AI implementation featuring:
- Complete TicTacToe game with PyTorch-based Deep Q-Network (DQN) AI
- Perfect Minimax AI that never loses 
- Position-aware neural network that can play as both X and O
- Interactive gameplay modes and AI training capabilities
- Performance analysis tools for evaluating AI strength

## Project Structure

**Core Files:**
- `tictactoe.py`: **CONSOLIDATED** complete implementation with working training, gameplay, and analysis
- `requirements.txt`: Python dependencies (PyTorch, NumPy, Matplotlib, Pygame)
- `optimized_tictactoe_ai.pt`: Pre-trained DQN model (best available)
- `working_tictactoe_ai.pt`: Model trained by current implementation
- `venv/`: Virtual environment

## Common Commands

### Setup and Installation

Create virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Game

**Interactive menu (recommended):**
```bash
python3 tictactoe.py
```

**Direct commands (all working):**
```bash
# Train a new AI (800 episodes by default, completes in ~1 minute)
python3 tictactoe.py train --episodes 800

# Play against the trained AI
python3 tictactoe.py play

# AI vs AI battle (100 games)
python3 tictactoe.py battle --games 100

# Analyze AI performance vs Minimax
python3 tictactoe.py analyze --games 100
```

## Architecture Overview

### Core Classes

- `TicTacToe`: Game logic with strategic reward shaping for training
- `ImprovedDQNAgent`: Position-aware DQN with 4-layer neural network (input includes player position)
- `MinimaxAgent`: Perfect AI using minimax algorithm with alpha-beta pruning
- `TrainingMonitor`: Tracks training progress and performance metrics

### Key Features

**Position-Aware Training:**
- Network receives board state + player indicator (X=1, O=-1)
- Single model handles both X and O positions effectively
- Alternating position training for balanced gameplay

**Strategic Reward Engineering:**
- Center control bonus: +0.1
- Corner control bonus: +0.05  
- Fork creation (multiple win paths): +0.3
- Blocking opponent wins: +0.2
- Creating win threats: +0.2

**Training Process:**
- Double DQN with experience replay
- Epsilon decay from 1.0 to 0.05
- Self-play with position alternation
- Regular validation against Minimax

## Performance Targets

**AI vs Minimax benchmarks:**
- Excellent: 60%+ non-loss rate (draws + wins) - very difficult against perfect play
- Good: 30%+ non-loss rate  
- Fair: 15%+ non-loss rate
- Baseline: 5%+ non-loss rate

**Realistic expectations:**
- **Current implementation**: ~20-35% vs Minimax (trains in ~1 minute)
- Quick training (200 episodes): ~25-35% non-loss rate
- Full training (800 episodes): ~30-40% non-loss rate

Note: Achieving high performance against perfect Minimax AI is extremely challenging. The AI learns strategic patterns and can achieve decent performance through progressive training against increasingly difficult opponents.

## Development Notes

**Model Loading/Saving:**
- Models are saved as PyTorch checkpoints with full training state
- Automatic model path defaults to `improved_tictactoe_ai.pt`
- Training automatically saves models upon completion

**Gameplay Interface:**
- Text-based coordinate system: row,col (0-2 range)
- Board display uses X, O, and . symbols
- Interactive prompts for human players

**Training Validation:**
- Built-in testing against Minimax every 2000 episodes
- Performance tracking for both X and O positions
- Training diagnostics show win/draw/loss rates

## Key Dependencies

- Python 3.8+
- PyTorch: Neural network training and inference
- NumPy: Array operations and game state management
- Matplotlib: Training progress visualization (optional)
- Pygame: Listed in requirements (unused in current implementation)