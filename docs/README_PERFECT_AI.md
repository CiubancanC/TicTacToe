# Perfect Tic Tac Toe AI: Minimax with Alpha-Beta Pruning

We've implemented a perfect Tic Tac Toe AI using the Minimax algorithm with Alpha-Beta pruning. This provides a guaranteed optimal strategy that will never lose and will win whenever possible.

## Why Minimax?

While our Deep Q-Learning implementation is impressive, it may not achieve perfect play even with extensive training. This is because:

1. Reinforcement learning can get stuck in local optima
2. The exploration-exploitation tradeoff can prevent discovering rare but crucial moves
3. Stochastic elements in training can lead to inconsistent play

For a solved game like Tic Tac Toe with a small state space (~3^9 = 19,683 possible states), minimax is a more suitable approach that guarantees optimal play.

## How Minimax Works

Minimax is a decision-making algorithm for turn-based, zero-sum games:

1. The algorithm recursively explores the game tree
2. It assumes the opponent will make the best possible move
3. It maximizes the score when it's the AI's turn
4. It minimizes the score when it's the opponent's turn

Our implementation includes alpha-beta pruning, an optimization technique that:
- Eliminates branches that can't affect the final decision
- Dramatically improves performance without sacrificing accuracy
- Enables the algorithm to search deeper within computational constraints

## Benefits of Minimax for Tic Tac Toe

- **Perfect play**: The AI will never lose, and will always win when possible
- **No training required**: Works immediately without needing to train
- **Deterministic**: Always makes the same (optimal) move in the same situation
- **Explainable**: The algorithm's decisions can be traced and understood
- **Computationally efficient**: Fast enough for real-time play in Tic Tac Toe

## Comparing to DQN

| Feature | Minimax | DQN |
|---------|---------|-----|
| Perfect play | ✓ Guaranteed | ✗ Approximated |
| Training time | ✓ None needed | ✗ Hours or days |
| Explainability | ✓ High | ✗ Low (black box) |
| Scalability to larger games | ✗ Poor | ✓ Better |
| Handling unseen states | ✓ Perfect | ✗ Depends on generalization |

## Using the Perfect AI

Try playing against the unbeatable AI:

```bash
python play_vs_perfect_ai.py
```

Or with the run_game.py script:

```bash
python run_game.py 4
```

Good luck trying to win - the best you can hope for is a draw!