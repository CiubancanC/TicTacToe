#!/usr/bin/env python3
"""
Strategic Showcase Training - Demonstrates enhanced reward system effectiveness
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai.agent import DQNAgent
from src.game.tictactoe import TicTacToe
import matplotlib.pyplot as plt
import numpy as np

def train_strategic_agent(episodes=3000, archive_interval=500):
    """Train agent with strategic rewards and show improvement over time"""
    print("🎯 Strategic Showcase Training")
    print("=" * 50)
    print(f"Training for {episodes} episodes with enhanced rewards...")
    print("📊 Tracking strategic behavior improvements")
    print()
    
    agent = DQNAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
    game = TicTacToe()
    
    # Tracking metrics
    center_moves = []
    corner_moves = []  
    strategic_moves = []
    rewards_per_episode = []
    win_rates = []
    
    episode_batch = 100
    games_won = 0
    games_total = 0
    
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        first_move_made = False
        episode_center = 0
        episode_corner = 0
        episode_strategic = 0
        
        # Play one game
        moves = 0
        while not game.game_over and moves < 9:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
                
            action = agent.choose_action(state, valid_moves)
            
            # Track first move for strategic analysis
            if not first_move_made:
                if action == (1, 1):  # Center
                    episode_center = 1
                    episode_strategic = 1
                elif action in [(0,0), (0,2), (2,0), (2,2)]:  # Corners
                    episode_corner = 1
                    episode_strategic = 1
                first_move_made = True
            
            # Make move with enhanced rewards
            next_state, reward, done = game.make_move(action, use_advanced_rewards=True)
            total_reward += reward
            
            # Store experience and learn
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            moves += 1
            
            if done:
                break
            
            # Opponent move (simple random for this demo)
            valid_moves = game.get_valid_moves()
            if valid_moves:
                import random
                opp_action = random.choice(valid_moves)
                next_state, opp_reward, done = game.make_move(opp_action, use_advanced_rewards=True)
                # Store opponent experience (optional)
                agent.remember(state, opp_action, -opp_reward, next_state, done)
                state = next_state
                moves += 1
        
        # Record metrics
        rewards_per_episode.append(total_reward)
        center_moves.append(episode_center)
        corner_moves.append(episode_corner)
        strategic_moves.append(episode_strategic)
        
        # Track win rate
        result = game.get_result()
        games_total += 1
        if result == 1:  # Agent won
            games_won += 1
        
        # Archive model periodically
        if (episode + 1) % archive_interval == 0:
            agent.archive_current_model()
        
        # Progress reporting
        if (episode + 1) % episode_batch == 0:
            recent_strategic = sum(strategic_moves[-episode_batch:])
            recent_center = sum(center_moves[-episode_batch:])
            recent_corner = sum(corner_moves[-episode_batch:])
            win_rate = games_won / games_total if games_total > 0 else 0
            
            print(f"Episode {episode + 1:4d}: Strategic={recent_strategic:2d}% Center={recent_center:2d}% Corner={recent_corner:2d}% WinRate={win_rate:.2f} ε={agent.epsilon:.3f}")
            
            win_rates.append(win_rate)
            games_won = 0  # Reset for next batch
            games_total = 0
    
    # Final evaluation
    print(f"\n🏆 Training Complete!")
    print(f"📊 Historical opponents created: {len(agent.historical_opponents)}")
    
    # Test final strategic behavior
    print(f"\n🎯 Final Strategic Assessment:")
    agent.epsilon = 0  # No exploration
    strategic_tests = 0
    center_tests = 0
    
    for test in range(20):
        game.reset()
        action = agent.choose_action(game.get_state(), game.get_valid_moves())
        if action == (1, 1):
            center_tests += 1
            strategic_tests += 1
        elif action in [(0,0), (0,2), (2,0), (2,2)]:
            strategic_tests += 1
    
    print(f"   Center preference: {center_tests}/20 ({center_tests/20*100:.0f}%)")
    print(f"   Strategic moves: {strategic_tests}/20 ({strategic_tests/20*100:.0f}%)")
    
    # Save model with descriptive name
    agent.save(f'models/strategic_dqn_{episodes}ep.pt')
    print(f"\n💾 Strategic model saved as: models/strategic_dqn_{episodes}ep.pt")
    
    # Create visualization
    create_strategic_visualization(center_moves, corner_moves, strategic_moves, rewards_per_episode, win_rates)
    
    return agent

def create_strategic_visualization(center_moves, corner_moves, strategic_moves, rewards, win_rates):
    """Create comprehensive visualization of strategic learning"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate rolling averages
    window = 100
    episodes = len(strategic_moves)
    
    if episodes >= window:
        center_avg = [sum(center_moves[i:i+window])/window for i in range(0, episodes-window+1, window)]
        corner_avg = [sum(corner_moves[i:i+window])/window for i in range(0, episodes-window+1, window)]  
        strategic_avg = [sum(strategic_moves[i:i+window])/window for i in range(0, episodes-window+1, window)]
        reward_avg = [sum(rewards[i:i+window])/window for i in range(0, episodes-window+1, window)]
        
        x_axis = list(range(0, episodes-window+1, window))
        
        # Strategic move evolution
        ax1.plot(x_axis, [s*100 for s in strategic_avg], 'b-', linewidth=3, label='Strategic Moves')
        ax1.plot(x_axis, [c*100 for c in center_avg], 'r--', linewidth=2, label='Center Moves')  
        ax1.plot(x_axis, [c*100 for c in corner_avg], 'g--', linewidth=2, label='Corner Moves')
        ax1.set_title('Strategic Play Evolution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Percentage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Reward progression
        ax2.plot(x_axis, reward_avg, 'purple', linewidth=3)
        ax2.set_title('Average Reward per Episode', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        # Win rate evolution
        if win_rates:
            ax3.plot(range(0, len(win_rates)*100, 100), [w*100 for w in win_rates], 'orange', linewidth=3, marker='o')
            ax3.set_title('Win Rate Evolution', fontweight='bold', fontsize=14)
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Win Rate (%)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        # Performance summary
        final_strategic = strategic_avg[-1] * 100 if strategic_avg else 0
        final_reward = reward_avg[-1] if reward_avg else 0
        final_winrate = win_rates[-1] * 100 if win_rates else 0
        
        ax4.text(0.1, 0.8, "🎯 Final Performance", fontsize=16, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.65, f"Strategic Moves: {final_strategic:.0f}%", fontsize=14, transform=ax4.transAxes)
        ax4.text(0.1, 0.55, f"Avg Reward: {final_reward:.2f}", fontsize=14, transform=ax4.transAxes) 
        ax4.text(0.1, 0.45, f"Win Rate: {final_winrate:.0f}%", fontsize=14, transform=ax4.transAxes)
        
        improvement = final_strategic - (strategic_avg[0] * 100 if strategic_avg else 0)
        ax4.text(0.1, 0.3, f"Strategic Improvement: +{improvement:.0f}%", fontsize=14, fontweight='bold', transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/strategic_showcase_results.png', dpi=300, bbox_inches='tight')
    print("📈 Strategic visualization saved to: visualizations/strategic_showcase_results.png")

def main():
    print("🎮 Strategic DQN Showcase")
    print("Demonstrating enhanced reward engineering effectiveness")
    print("=" * 60)
    
    episodes = int(input("Enter training episodes (default 2000): ") or "2000")
    
    agent = train_strategic_agent(episodes)
    
    print(f"\n✨ Showcase Complete!")
    print(f"\n🎮 Test the strategic AI:")
    print(f"   python3 -c \"")
    print(f"   from src.ai.agent import DQNAgent")
    print(f"   from src.game.tictactoe import TicTacToe")
    print(f"   agent = DQNAgent(epsilon=0)")
    print(f"   agent.load('models/strategic_dqn_{episodes}ep.pt')")
    print(f"   game = TicTacToe()")
    print(f"   print('First move:', agent.choose_action(game.get_state(), game.get_valid_moves()))")
    print(f"   \"")

if __name__ == "__main__":
    main()