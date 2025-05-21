import sys
sys.path.append('.')
from src.main import train_agent
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    print("Training improved AI agent...")
    start_time = time.time()
    
    # Train for 10000 episodes to test the improved agent
    agent = train_agent(num_episodes=10000, save_interval=200)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print("Model saved as models/dqn_agent_final.pt")
    
    # Plot the loss history if available
    if hasattr(agent, 'loss_history') and len(agent.loss_history) > 0:
        plt.figure(figsize=(10, 6))
        # Smooth the loss curve with a rolling average
        window_size = min(100, len(agent.loss_history))
        loss_smoothed = np.convolve(agent.loss_history, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(loss_smoothed)
        plt.title('Training Loss (Smoothed)')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.savefig('loss_history.png')
        print("Loss history plot saved as loss_history.png")
    
    print("To play against the trained AI, run: python src/main.py and select option 2")