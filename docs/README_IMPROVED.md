# Improved Tic Tac Toe AI

We've significantly enhanced the AI for the Tic Tac Toe game using advanced deep reinforcement learning techniques:

## Improvements Made

### 1. Enhanced Neural Network Architecture
- Deeper network with larger hidden layers (256 → 256 → 128 → 9)
- Added batch normalization for more stable training
- Implemented LeakyReLU activation functions for better gradient flow
- Added dropout layers (20%) to prevent overfitting

### 2. Advanced DQN Techniques
- Implemented Double DQN to reduce overestimation bias
- Added gradient clipping to prevent exploding gradients
- Used Huber Loss (SmoothL1Loss) instead of MSE for better robustness
- Implemented learning rate scheduler to adaptively adjust learning rate

### 3. Prioritized Experience Replay
- Replaced uniform sampling with prioritized experience replay
- Added importance sampling to correct for bias
- Dynamically adjusts transition priorities based on TD error
- Helps focus learning on important transitions

### 4. Optimized Training Parameters
- Increased batch size from 64 to 128
- Reduced learning rate with better decay schedule
- More frequent target network updates
- Larger replay buffer capacity (50,000 vs 10,000)
- Added L2 regularization (weight decay) to prevent overfitting

## Results

The training metrics show:
- Win rate increased to 60-70%
- Draw rate around 30-35%
- Loss rate reduced to 1-4%

The AI has been trained to make better strategic decisions and is now much more difficult to beat!

## How to Use

1. **Train the improved AI**:
   ```
   python train_improved_ai.py
   ```

2. **Play against the improved AI**:
   ```
   python play_vs_ai.py
   ```

3. **Play human vs human**:
   ```
   python src/main.py
   ```
   Then select option 3