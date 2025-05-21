#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import train_agent
import matplotlib.pyplot as plt
import numpy as np
import time

def train_enhanced_ai():
    """Train AI with improved reward engineering and self-play archives"""
    print("🤖 Training Enhanced AI with Advanced Features")
    print("=" * 50)
    print("Features:")
    print("✓ Strategic reward shaping (center control, forks, blocking)")
    print("✓ Self-play with historical opponent archives")
    print("✓ Higher win rewards (10x) vs lower draw rewards")
    print("✓ Prioritized experience replay")
    print("✓ Double DQN with target network updates")
    print("=" * 50)
    
    start_time = time.time()
    
    # Train for more episodes with the enhanced system
    print("Starting training for 15,000 episodes...")
    agent = train_agent(num_episodes=15000, save_interval=1000)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n🎉 Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    print("📁 Model saved as models/dqn_agent_final.pt")
    
    # Enhanced visualization
    if hasattr(agent, 'loss_history') and len(agent.loss_history) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss history with better smoothing
        window_size = min(200, len(agent.loss_history) // 20)
        if window_size > 1:
            loss_smoothed = np.convolve(agent.loss_history, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(loss_smoothed, linewidth=2, color='darkblue')
            ax1.set_title('Training Loss (Enhanced DQN)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # Training metrics summary
        final_loss = np.mean(agent.loss_history[-100:]) if len(agent.loss_history) >= 100 else agent.loss_history[-1]
        ax2.text(0.1, 0.8, f"📊 Training Summary", fontsize=16, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.1, 0.7, f"Total Episodes: 15,000", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.6, f"Training Time: {training_time/60:.1f} minutes", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.5, f"Final Loss: {final_loss:.4f}", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.4, f"Historical Opponents: {len(agent.historical_opponents)}", fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.3, f"Final Epsilon: {agent.epsilon:.4f}", fontsize=12, transform=ax2.transAxes)
        
        # Reward improvements
        ax2.text(0.1, 0.15, "🎯 Enhanced Features:", fontsize=12, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.15, 0.07, "• Strategic move rewards", fontsize=10, transform=ax2.transAxes)
        ax2.text(0.15, 0.02, "• Self-play vs historical opponents", fontsize=10, transform=ax2.transAxes)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/enhanced_training_results.png', dpi=300, bbox_inches='tight')
        print("📈 Enhanced training visualization saved to visualizations/enhanced_training_results.png")
    
    print("\n🎮 Ready to Play!")
    print("To test the enhanced AI:")
    print("  python3 scripts/play_vs_ai.py")
    print("  python3 scripts/play_vs_perfect_ai.py")
    print("  python3 src/main.py (and select option 2)")
    
    return agent

if __name__ == "__main__":
    enhanced_agent = train_enhanced_ai()