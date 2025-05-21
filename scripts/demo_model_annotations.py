#!/usr/bin/env python3
"""
Demo script to create models with proper training method annotations
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_model_naming():
    """Train small demo models to show the new naming system"""
    print("🏷️  Model Annotation Demo")
    print("=" * 50)
    print("Creating demo models with proper training method annotations...")
    
    from src.main import train_agent
    
    # 1. Original DQN (small for demo)
    print("\n🔬 Training Original DQN (500 episodes)...")
    agent1 = train_agent(num_episodes=500, save_interval=200)
    agent1.save("models/original_dqn_500ep.pt")
    print("💾 Saved: models/original_dqn_500ep.pt")
    
    # 2. Enhanced DQN (small for demo) 
    print("\n🎯 Training Enhanced DQN (500 episodes)...")
    agent2 = train_agent(num_episodes=500, save_interval=200)
    agent2.save("models/enhanced_dqn_500ep.pt")
    print("💾 Saved: models/enhanced_dqn_500ep.pt")
    
    print("\n✅ Demo models created!")
    print("\n📋 Testing new model listing system...")
    
    # Test the listing system
    from run_game import list_available_models
    models = list_available_models()
    
    print("\n📊 Available Models (with training annotations):")
    print("-" * 55)
    for i, (name, path) in enumerate(models, 1):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {i:2d}. {name} ({size_kb:.1f} KB)")
    
    print(f"\n🎉 Model annotation system working!")
    print(f"💡 Now you can easily distinguish training methods:")
    print(f"   🔬 Original DQN = Standard self-play")
    print(f"   🎯 Enhanced DQN = Strategic rewards + archives")
    print(f"   🧠 Strategic DQN = Focused strategic training")
    print(f"   📦 Legacy = Old naming convention")

if __name__ == "__main__":
    demo_model_naming()