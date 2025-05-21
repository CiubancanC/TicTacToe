# Tic Tac Toe with Reinforcement Learning

This project implements a Tic Tac Toe game using PyGame and trains a deep reinforcement learning agent (DQN) that learns to play optimally.

## Features

- Playable Tic Tac Toe game with a graphical user interface
- Deep Q-Network (DQN) agent that learns through self-play
- Human vs AI gameplay mode
- Human vs Human gameplay mode
- Training progress visualization

## Requirements

- Python 3.8+
- PyGame
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone this repository or download the source code
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```
python src/main.py
```

You'll be presented with three options:

1. **Train AI agent**: Train the reinforcement learning agent from scratch
2. **Play against AI**: Play a game against a previously trained AI, or an untrained one if no model exists
3. **Human vs Human**: Play a local multiplayer game

## How the AI Works

The AI uses a Deep Q-Network (DQN) to learn optimal play through self-play. The neural network takes the board state as input and outputs Q-values for each possible move.

During training, the agent:
- Plays against itself to learn from experience
- Uses epsilon-greedy exploration strategy
- Stores experiences in a replay buffer
- Learns by minimizing the difference between predicted and target Q-values

With sufficient training, the agent will learn to:
- Never lose (either win or tie)
- Choose the optimal move in any given board state
- Play defensively when needed

## Project Structure

- `src/game/tictactoe.py`: Implements the Tic Tac Toe game logic and GUI
- `src/ai/agent.py`: Implements the DQN agent with replay buffer
- `src/main.py`: Main script to run training and gameplay
- `models/`: Directory to store trained models