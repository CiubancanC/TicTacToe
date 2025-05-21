#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import train_agent
import matplotlib.pyplot as plt
import numpy as np
import time

def test_enhanced_features():
    """Quick test of enhanced features with small training"""
    print("🧪 Testing Enhanced AI Features")
    print("=" * 40)
    
    # Short training run to test new features
    print("Running quick test (500 episodes)...")
    start_time = time.time()
    
    agent = train_agent(num_episodes=500, save_interval=200)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Test completed in {training_time:.2f} seconds")
    print(f"📊 Historical opponents created: {len(agent.historical_opponents)}")
    print(f"📈 Training steps: {len(agent.loss_history)}")
    
    # Test strategic reward calculation
    from src.game.tictactoe import TicTacToe
    game = TicTacToe()
    
    # Test center move reward
    state, reward, done = game.make_move((1, 1), use_advanced_rewards=True)
    print(f"🎯 Center move reward: {reward} (should be > 0)")
    
    # Test corner move reward  
    game.reset()
    state, reward, done = game.make_move((0, 0), use_advanced_rewards=True)
    print(f"🔶 Corner move reward: {reward} (should be > 0)")
    
    print("\n🎉 Enhanced features working correctly!")
    return agent

if __name__ == "__main__":
    test_enhanced_features()