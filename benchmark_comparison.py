import sys
import time
import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Import both versions
import tictactoe_original as original
import tictactoe_improved as improved
from benchmark_framework import BenchmarkFramework, BenchmarkMetrics

class TicTacToeBenchmark:
    def __init__(self):
        self.framework = BenchmarkFramework()
        self.results = {}
        
    def create_original_agents(self):
        """Create agents using original implementation"""
        agent_x = original.QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9, train_mode=True)
        agent_o = original.QLearningAgent('O', epsilon=0.2, alpha=0.1, gamma=0.9, train_mode=True)
        return agent_x, agent_o
    
    def create_improved_agents(self):
        """Create agents using improved implementation"""
        agent_x = improved.QLearningAgent('X', epsilon=0.2, alpha=0.1, gamma=0.9, 
                                         train_mode=True, use_experience_replay=True,
                                         batch_size=32, use_lr_scheduler=True)
        agent_o = improved.QLearningAgent('O', epsilon=0.2, alpha=0.1, gamma=0.9,
                                         train_mode=True, use_experience_replay=True,
                                         batch_size=32, use_lr_scheduler=True)
        return agent_x, agent_o
    
    def benchmark_training_time(self, implementation_name, module, create_agents_func, episodes=10000):
        """Benchmark training time for an implementation"""
        print(f"\nBenchmarking {implementation_name} training time...")
        
        metrics = BenchmarkMetrics()
        
        # Create agents
        agent_x, agent_o = create_agents_func()
        
        # Measure training time
        start_time = time.time()
        module.diverse_training(agent_x, agent_o, num_episodes=episodes)
        end_time = time.time()
        
        metrics.training_time = end_time - start_time
        
        # Measure memory usage
        if hasattr(agent_x, 'q_table'):
            serialized = pickle.dumps(agent_x.q_table)
            metrics.memory_usage_mb = len(serialized) / (1024 * 1024)
            metrics.q_table_entries = len(agent_x.q_table)
        
        # Evaluate performance
        agent_x.train_mode = False
        agent_o.train_mode = False
        
        # Test against random player
        wins = 0
        test_games = 100
        random_opponent = module.RandomPlayer('O')
        
        for _ in range(test_games):
            game = module.TicTacToe()
            winner = module.play(game, agent_x, random_opponent, print_game=False)
            if winner == 'X':
                wins += 1
        
        metrics.final_win_rates['vs_random'] = wins / test_games
        
        # Test against smart random
        wins = 0
        smart_opponent = module.SmartRandomPlayer('O', smart_probability=0.5)
        
        for _ in range(test_games):
            game = module.TicTacToe()
            winner = module.play(game, agent_x, smart_opponent, print_game=False)
            if winner == 'X':
                wins += 1
        
        metrics.final_win_rates['vs_smart_random'] = wins / test_games
        
        return metrics
    
    def benchmark_convergence(self, implementation_name, module, create_agents_func, 
                            eval_interval=1000, max_episodes=20000):
        """Measure convergence speed"""
        print(f"\nMeasuring convergence for {implementation_name}...")
        
        convergence_history = []
        agent_x, agent_o = create_agents_func()
        
        for episode in range(0, max_episodes, eval_interval):
            # Train for interval
            if episode > 0:
                module.diverse_training(agent_x, agent_o, num_episodes=eval_interval)
            
            # Evaluate
            agent_x.train_mode = False
            wins = 0
            test_games = 50
            random_opponent = module.RandomPlayer('O')
            
            for _ in range(test_games):
                game = module.TicTacToe()
                winner = module.play(game, agent_x, random_opponent, print_game=False)
                if winner == 'X':
                    wins += 1
            
            win_rate = wins / test_games
            convergence_history.append({
                'episode': episode,
                'win_rate': win_rate
            })
            
            agent_x.train_mode = True
            
            print(f"  Episode {episode}: Win rate = {win_rate:.2f}")
        
        return convergence_history
    
    def run_full_benchmark(self, training_episodes=10000):
        """Run complete benchmark comparison"""
        print("="*80)
        print("TICTACTOE IMPLEMENTATION BENCHMARK COMPARISON")
        print("="*80)
        
        # Benchmark original implementation
        print("\n1. ORIGINAL IMPLEMENTATION")
        print("-"*40)
        original_metrics = self.benchmark_training_time(
            "Original", original, self.create_original_agents, training_episodes
        )
        self.results['original'] = original_metrics
        
        # Benchmark improved implementation
        print("\n2. IMPROVED IMPLEMENTATION")
        print("-"*40)
        improved_metrics = self.benchmark_training_time(
            "Improved", improved, self.create_improved_agents, training_episodes
        )
        self.results['improved'] = improved_metrics
        
        # Compare convergence
        print("\n3. CONVERGENCE COMPARISON")
        print("-"*40)
        original_convergence = self.benchmark_convergence(
            "Original", original, self.create_original_agents, eval_interval=1000, max_episodes=10000
        )
        improved_convergence = self.benchmark_convergence(
            "Improved", improved, self.create_improved_agents, eval_interval=1000, max_episodes=10000
        )
        
        # Generate comparison report
        self.generate_report(original_convergence, improved_convergence)
        
    def generate_report(self, original_convergence, improved_convergence):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        orig = self.results['original']
        impr = self.results['improved']
        
        # Training time comparison
        print("\n1. TRAINING TIME:")
        print(f"   Original: {orig.training_time:.2f} seconds")
        print(f"   Improved: {impr.training_time:.2f} seconds")
        time_improvement = ((orig.training_time - impr.training_time) / orig.training_time) * 100
        print(f"   Improvement: {time_improvement:+.1f}% {'faster' if time_improvement > 0 else 'slower'}")
        
        # Memory usage comparison
        print("\n2. MEMORY USAGE:")
        print(f"   Original: {orig.memory_usage_mb:.2f} MB ({orig.q_table_entries} entries)")
        print(f"   Improved: {impr.memory_usage_mb:.2f} MB ({impr.q_table_entries} entries)")
        memory_improvement = ((orig.memory_usage_mb - impr.memory_usage_mb) / orig.memory_usage_mb) * 100
        print(f"   Improvement: {memory_improvement:+.1f}% {'less' if memory_improvement > 0 else 'more'} memory")
        
        # Performance comparison
        print("\n3. FINAL PERFORMANCE:")
        print("   vs Random Player:")
        print(f"     Original: {orig.final_win_rates['vs_random']*100:.1f}% win rate")
        print(f"     Improved: {impr.final_win_rates['vs_random']*100:.1f}% win rate")
        
        print("   vs Smart Random:")
        print(f"     Original: {orig.final_win_rates['vs_smart_random']*100:.1f}% win rate")
        print(f"     Improved: {impr.final_win_rates['vs_smart_random']*100:.1f}% win rate")
        
        # Key improvements summary
        print("\n4. KEY IMPROVEMENTS IMPLEMENTED:")
        print("   ✓ Numpy-based board representation for vectorization")
        print("   ✓ Experience replay buffer for stable learning")
        print("   ✓ Batch learning updates")
        print("   ✓ Q-value normalization for numerical stability")
        print("   ✓ Learning rate scheduling")
        print("   ✓ Efficient state hashing")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'training_episodes': 10000,
                'results': {
                    'original': orig.to_dict(),
                    'improved': impr.to_dict()
                },
                'convergence': {
                    'original': original_convergence,
                    'improved': improved_convergence
                }
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Plot convergence comparison
        self.plot_convergence(original_convergence, improved_convergence)
        
    def plot_convergence(self, original_convergence, improved_convergence):
        """Create convergence comparison plot"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Extract data
            orig_episodes = [d['episode'] for d in original_convergence]
            orig_rates = [d['win_rate'] for d in original_convergence]
            
            impr_episodes = [d['episode'] for d in improved_convergence]
            impr_rates = [d['win_rate'] for d in improved_convergence]
            
            # Plot
            plt.plot(orig_episodes, orig_rates, 'b-', label='Original', linewidth=2)
            plt.plot(impr_episodes, impr_rates, 'r-', label='Improved', linewidth=2)
            
            plt.xlabel('Training Episodes')
            plt.ylabel('Win Rate vs Random Player')
            plt.title('Convergence Speed Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Save plot
            plt.savefig('convergence_comparison.png', dpi=150, bbox_inches='tight')
            print("\nConvergence plot saved to: convergence_comparison.png")
            
        except Exception as e:
            print(f"\nCould not create plot: {e}")
            print("(matplotlib may not be installed)")

if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Warning: matplotlib not installed. Plots will be skipped.")
    
    # Run benchmark
    benchmark = TicTacToeBenchmark()
    
    # Use fewer episodes for quick testing
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick benchmark (1000 episodes)...")
        benchmark.run_full_benchmark(training_episodes=1000)
    else:
        print("Running full benchmark (10000 episodes)...")
        print("Use '--quick' flag for faster testing")
        benchmark.run_full_benchmark(training_episodes=10000)