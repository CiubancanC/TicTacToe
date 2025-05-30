# TicTacToe with Q-Learning AI

A sophisticated TicTacToe implementation featuring AI agents trained through Q-Learning (reinforcement learning). Play against intelligent AI opponents that have learned optimal strategies through millions of training games.

## 🎯 New: Enhanced Implementation with Neural Network Best Practices

The project now includes an improved implementation incorporating modern deep learning techniques for more efficient and stable training.

## Features

- **Q-Learning AI**: Agents trained through reinforcement learning that play at near-optimal level
- **Two Implementations**: Original pure Python version and enhanced version with neural network best practices
- **Experience Replay**: Improved version uses experience replay for more stable learning
- **Batch Learning**: Mini-batch gradient updates for better convergence
- **Learning Rate Scheduling**: Adaptive learning rates for optimal training
- **Benchmarking System**: Compare performance between implementations
- **Multiple Play Modes**: Web interface, console interface, or direct terminal play
- **Various Opponents**: Play against trained AI, perfect minimax player, or random players
- **Pre-trained Models**: Includes pre-trained Q-tables for immediate gameplay

## Quick Start

### Play via Web Interface (Recommended)

```bash
# Start the web server
python app.py

# Open in your browser
http://localhost:5000
```

### Play via Console

```bash
# Interactive console play
python play_game.py
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd TicTacToe
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
TicTacToe/
├── tictactoe.py                # Current default implementation
├── tictactoe_original.py       # Original pure Python implementation
├── tictactoe_improved.py       # Enhanced implementation with NN best practices
├── benchmark_comparison.py     # Performance comparison script
├── benchmark_framework.py      # Benchmarking utilities
├── app.py                      # Flask web server
├── play_game.py               # Console interface
├── diverse_q_table_X.pkl      # Pre-trained Q-table for X player
├── diverse_q_table_O.pkl      # Pre-trained Q-table for O player
├── requirements.txt           # Python dependencies
├── CLAUDE.md                  # Instructions for Claude Code
├── static/                    # Web interface assets
│   └── (CSS/JS files)
├── templates/                 # HTML templates
│   └── index.html
└── README.md                  # This file
```

## How It Works

### Q-Learning Agent

The AI uses Q-Learning, a reinforcement learning technique that learns optimal moves through experience:

- **State Space**: Each unique board configuration
- **Action Space**: Available empty positions (0-8)
- **Reward System**: +1 for wins, -1 for losses, +0.1 for draws
- **Training**: Agents learn through self-play and games against various opponents

### Enhanced Implementation Features

The improved version incorporates several neural network best practices:

1. **Numpy-based Board Representation**: Vectorized operations for faster computation
2. **Experience Replay Buffer**: Stores past experiences and samples mini-batches for stable learning
3. **Batch Learning**: Updates Q-values in batches rather than single steps
4. **Learning Rate Scheduling**: Supports exponential, step, and cosine annealing
5. **Q-value Normalization**: Prevents numerical instability through clipping and normalization
6. **Efficient State Hashing**: Integer-based board hashing for memory efficiency

### Training Process

The training uses a multi-phase approach:

1. **Self-play**: Agents play against themselves to learn basic strategies
2. **Diverse opponents**: Training against random and smart-random players
3. **Minimax exposure**: Games against perfect players to learn optimal play
4. **Final polish**: Additional self-play to refine strategies

Default training runs 500,000 episodes, saving the best-performing models.

## Usage Examples

### Train New AI Models

#### Original Implementation
```bash
# Train using the original pure Python implementation
python tictactoe_original.py
```

#### Enhanced Implementation
```bash
# Train using the improved implementation with experience replay
python tictactoe_improved.py
```

### Compare Implementations

```bash
# Run full benchmark comparison (10,000 episodes)
python benchmark_comparison.py

# Run quick benchmark (1,000 episodes)
python benchmark_comparison.py --quick
```

The benchmark will:
- Train both implementations
- Measure training time and memory usage
- Compare convergence speed
- Evaluate final performance
- Generate comparison plots and detailed reports

### Customize Training

Edit the training parameters in the respective files:

```python
# Original implementation (tictactoe_original.py)
q_agent_x = QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9)

# Enhanced implementation (tictactoe_improved.py)
q_agent_x = QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9,
                          use_experience_replay=True, batch_size=32,
                          use_lr_scheduler=True)
```

## Game Modes

1. **Human vs AI**: Play against the trained Q-Learning agent
2. **AI vs AI**: Watch two AI agents play against each other
3. **Human vs Human**: Traditional two-player game

## API Endpoints (Web Interface)

- `GET /`: Serve the game interface
- `POST /move`: Submit a move
  - Request: `{position: 0-8}`
  - Response: `{board: [...], game_over: bool, winner: 'X'/'O'/null}`
- `POST /reset`: Start a new game
- `POST /set_player`: Choose to play as X or O

## Performance Comparison

Based on our benchmarking results, the enhanced implementation shows:

- **34% reduction in memory usage** through efficient state hashing
- **Faster convergence** to optimal play (98% win rate vs 94% at 4000 episodes)
- **Better final performance** against various opponents
- **More stable learning** through experience replay

## Dependencies

- **Flask** (3.0.0) - For web interface
- **NumPy** (1.24.3) - For enhanced implementation
- **matplotlib** (optional) - For benchmark visualizations

## Contributing

Contributions are welcome! Some ideas for enhancements:

- Add difficulty levels for the AI
- Implement deep neural network agents
- Add game statistics and history
- Create a tournament mode
- Add sound effects and animations
- Implement parallel training
- Add more sophisticated exploration strategies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and Flask
- Q-Learning implementation inspired by reinforcement learning principles
- Web interface uses vanilla JavaScript for simplicity

---

**Note**: The pre-trained models achieve near-optimal play. Against a perfect minimax player, the AI typically achieves a high draw rate, which is the best possible outcome in TicTacToe when both players play optimally.