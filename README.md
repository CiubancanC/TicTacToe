# TicTacToe with Q-Learning AI

A sophisticated TicTacToe implementation featuring AI agents trained through Q-Learning (reinforcement learning). Play against intelligent AI opponents that have learned optimal strategies through millions of training games.

## Features

- **Q-Learning AI**: Agents trained through reinforcement learning that play at near-optimal level
- **Multiple Play Modes**: Web interface, console interface, or direct terminal play
- **Various Opponents**: Play against trained AI, perfect minimax player, or random players
- **Pre-trained Models**: Includes pre-trained Q-tables for immediate gameplay
- **Training System**: Train your own AI agents with customizable parameters

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
├── tictactoe.py           # Core game engine and AI implementations
├── app.py                 # Flask web server
├── play_game.py           # Console interface
├── diverse_q_table_X.pkl  # Pre-trained Q-table for X player
├── diverse_q_table_O.pkl  # Pre-trained Q-table for O player
├── requirements.txt       # Python dependencies
├── static/               # Web interface assets
│   └── (CSS/JS files)
├── templates/            # HTML templates
│   └── index.html
└── README.md             # This file
```

## How It Works

### Q-Learning Agent

The AI uses Q-Learning, a reinforcement learning technique that learns optimal moves through experience:

- **State Space**: Each unique board configuration
- **Action Space**: Available empty positions (0-8)
- **Reward System**: +1 for wins, -1 for losses, +0.5 for draws
- **Training**: Agents learn through self-play and games against various opponents

### Training Process

The training uses a multi-phase approach:

1. **Self-play**: Agents play against themselves to learn basic strategies
2. **Diverse opponents**: Training against random and smart-random players
3. **Minimax exposure**: Games against perfect players to learn optimal play
4. **Final polish**: Additional self-play to refine strategies

Default training runs 500,000 episodes, saving the best-performing models.

## Usage Examples

### Train New AI Models

```bash
python tictactoe.py
```

This will:
- Train new Q-Learning agents
- Evaluate performance against various opponents
- Save the trained models as pickle files

### Customize Training

Edit the training parameters in `tictactoe.py`:

```python
# In main():
episodes = 500000  # Number of training games
learning_rate = 0.1
discount_factor = 0.95
initial_epsilon = 1.0
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

## Contributing

Contributions are welcome! Some ideas for enhancements:

- Add difficulty levels for the AI
- Implement neural network-based agents
- Add game statistics and history
- Create a tournament mode
- Add sound effects and animations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python and Flask
- Q-Learning implementation inspired by reinforcement learning principles
- Web interface uses vanilla JavaScript for simplicity

---

**Note**: The pre-trained models achieve near-optimal play. Against a perfect minimax player, the AI typically achieves a high draw rate, which is the best possible outcome in TicTacToe when both players play optimally.