import sys
sys.path.append('.')
from src.main import human_vs_ai

if __name__ == "__main__":
    print("Starting game against the improved AI agent...")
    print("You will play as X (first player)")
    print("The AI will play as O (second player)")
    print("Click on the board to make your move")
    
    # Use the trained model
    human_vs_ai("models/dqn_agent_final.pt")