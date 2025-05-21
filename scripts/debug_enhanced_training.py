#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.tictactoe import TicTacToe
from src.ai.agent import DQNAgent

def test_reward_system():
    """Test if the enhanced reward system is working"""
    print("🔍 Testing Enhanced Reward System")
    print("=" * 40)
    
    game = TicTacToe()
    
    # Test center move reward
    print("🎯 Testing center move (1,1):")
    state, reward, done = game.make_move((1, 1), use_advanced_rewards=True)
    print(f"   Reward: {reward} (should be > 0 for strategic bonus)")
    
    # Test corner move reward  
    game.reset()
    print("\n🔶 Testing corner move (0,0):")
    state, reward, done = game.make_move((0, 0), use_advanced_rewards=True)
    print(f"   Reward: {reward} (should be > 0 for strategic bonus)")
    
    # Test edge move (no bonus)
    game.reset()
    print("\n📏 Testing edge move (0,1):")
    state, reward, done = game.make_move((0, 1), use_advanced_rewards=True)
    print(f"   Reward: {reward} (should be 0 - no strategic bonus)")
    
    # Test blocking scenario
    print("\n🛡️ Testing blocking reward:")
    game.reset()
    game.board[0, 0] = -1  # Opponent
    game.board[0, 1] = -1  # Opponent  
    game.current_player = 1  # Our turn
    state, reward, done = game.make_move((0, 2), use_advanced_rewards=True)
    print(f"   Blocking reward: {reward} (should include +1.5 blocking bonus)")

def quick_strategic_training():
    """Quick training focused on strategic play"""
    print("\n🚀 Quick Strategic Training Test (100 episodes)")
    print("=" * 50)
    
    agent = DQNAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99)
    game = TicTacToe()
    
    center_choices = 0
    corner_choices = 0
    total_first_moves = 0
    
    for episode in range(100):
        state = game.reset()
        
        # First move analysis
        valid_moves = game.get_valid_moves()
        action = agent.choose_action(state, valid_moves)
        
        if action == (1, 1):  # Center
            center_choices += 1
        elif action in [(0,0), (0,2), (2,0), (2,2)]:  # Corners
            corner_choices += 1
        total_first_moves += 1
        
        # Play out the game with enhanced rewards
        done = False
        moves = 0
        while not done and moves < 9:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            
            action = agent.choose_action(state, valid_moves)
            next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
            
            # Store experience and learn
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 32:  # Start learning after some experiences
                agent.replay()
            
            state = next_state
            moves += 1
            
            # Switch to opponent (simplified self-play)
            if not done:
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    import random
                    opponent_action = random.choice(valid_moves)
                    next_state, reward, done = game.make_move(opponent_action, use_advanced_rewards=True)
                    agent.remember(state, opponent_action, -reward, next_state, done)
                    state = next_state
    
    print(f"📊 First Move Analysis (last 100 episodes):")
    print(f"   Center (1,1): {center_choices}/100 ({center_choices}%)")
    print(f"   Corners: {corner_choices}/100 ({corner_choices}%)")
    print(f"   Others: {total_first_moves - center_choices - corner_choices}/100")
    
    # Test final behavior
    print(f"\n🎯 Final Model Behavior:")
    agent.epsilon = 0  # No exploration
    game.reset()
    final_action = agent.choose_action(game.get_state(), game.get_valid_moves())
    print(f"   First move preference: {final_action}")
    
    if final_action == (1, 1):
        print("   ✅ Learned center preference!")
    elif final_action in [(0,0), (0,2), (2,0), (2,2)]:
        print("   ⚠️ Learned corner preference (also strategic)")  
    else:
        print("   ❌ No strategic preference learned")

def main():
    print("🔧 Enhanced Training Debug Suite")
    print("=" * 50)
    
    # Test reward system
    test_reward_system()
    
    # Quick training test
    quick_strategic_training()
    
    print("\n💡 If rewards are working but training isn't effective:")
    print("   - May need longer training (>1000 episodes)")
    print("   - Consider adjusting learning rate or epsilon decay")
    print("   - Historical opponents may need more episodes to be effective")

if __name__ == "__main__":
    main()