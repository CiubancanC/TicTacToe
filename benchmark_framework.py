import time
import pickle
import os
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

class BenchmarkMetrics:
    """Container for benchmark metrics"""
    def __init__(self):
        self.training_time = 0.0
        self.episodes_to_threshold = 0
        self.final_win_rates = {}
        self.memory_usage_mb = 0.0
        self.q_table_size = 0
        self.q_table_entries = 0
        self.convergence_history = []
        self.evaluation_results = {}
        
    def to_dict(self):
        return {
            'training_time': self.training_time,
            'episodes_to_threshold': self.episodes_to_threshold,
            'final_win_rates': self.final_win_rates,
            'memory_usage_mb': self.memory_usage_mb,
            'q_table_size': self.q_table_size,
            'q_table_entries': self.q_table_entries,
            'convergence_history': self.convergence_history,
            'evaluation_results': self.evaluation_results
        }

class BenchmarkFramework:
    """Framework for comparing different TicTacToe implementations"""
    
    def __init__(self, results_dir="benchmark_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def measure_training_time(self, train_func, *args, **kwargs):
        """Measure the time taken for training"""
        start_time = time.time()
        result = train_func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def measure_convergence(self, agent_x, agent_o, eval_interval=1000, 
                          threshold_win_rate=0.8, max_episodes=100000):
        """Measure how many episodes it takes to reach a performance threshold"""
        convergence_history = []
        episodes_to_threshold = max_episodes
        
        for episode in range(0, max_episodes, eval_interval):
            # Quick evaluation against random player
            wins = 0
            test_games = 50
            
            for _ in range(test_games):
                # Test game logic here (simplified)
                pass
            
            win_rate = wins / test_games
            convergence_history.append({
                'episode': episode,
                'win_rate': win_rate
            })
            
            if win_rate >= threshold_win_rate and episodes_to_threshold == max_episodes:
                episodes_to_threshold = episode
                
        return episodes_to_threshold, convergence_history
    
    def measure_memory_usage(self, agent):
        """Estimate memory usage of the agent"""
        if hasattr(agent, 'q_table'):
            # Serialize to measure size
            serialized = pickle.dumps(agent.q_table)
            size_bytes = len(serialized)
            size_mb = size_bytes / (1024 * 1024)
            entries = len(agent.q_table)
            return size_mb, entries
        return 0.0, 0
    
    def evaluate_performance(self, agent, opponent_types, games_per_opponent=100):
        """Evaluate agent against various opponents"""
        results = {}
        
        for opponent_name, opponent in opponent_types.items():
            wins = losses = draws = 0
            
            # Evaluation logic here (simplified)
            # Would play games and track results
            
            results[opponent_name] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': wins / games_per_opponent if games_per_opponent > 0 else 0
            }
            
        return results
    
    def run_benchmark(self, implementation_name, train_func, create_agents_func, 
                     training_episodes=10000, **kwargs):
        """Run a complete benchmark for one implementation"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {implementation_name}")
        print(f"{'='*60}")
        
        metrics = BenchmarkMetrics()
        
        # Create agents
        agent_x, agent_o = create_agents_func()
        
        # Measure training time
        print("1. Measuring training time...")
        _, training_time = self.measure_training_time(
            train_func, agent_x, agent_o, training_episodes, **kwargs
        )
        metrics.training_time = training_time
        print(f"   Training completed in {training_time:.2f} seconds")
        
        # Measure memory usage
        print("2. Measuring memory usage...")
        memory_mb, entries = self.measure_memory_usage(agent_x)
        metrics.memory_usage_mb = memory_mb
        metrics.q_table_entries = entries
        print(f"   Q-table size: {memory_mb:.2f} MB with {entries} entries")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir, f"{implementation_name}_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump({
                'implementation': implementation_name,
                'timestamp': timestamp,
                'metrics': metrics.to_dict()
            }, f, indent=2)
        
        print(f"\nResults saved to: {result_file}")
        
        return metrics
    
    def compare_implementations(self, results: Dict[str, BenchmarkMetrics]):
        """Generate comparison report between implementations"""
        print(f"\n{'='*60}")
        print("BENCHMARK COMPARISON REPORT")
        print(f"{'='*60}\n")
        
        # Training time comparison
        print("Training Time:")
        for name, metrics in results.items():
            print(f"  {name}: {metrics.training_time:.2f} seconds")
        
        # Memory usage comparison
        print("\nMemory Usage:")
        for name, metrics in results.items():
            print(f"  {name}: {metrics.memory_usage_mb:.2f} MB ({metrics.q_table_entries} entries)")
        
        # Performance comparison would go here
        
        # Calculate improvements
        if len(results) == 2 and 'original' in results and 'improved' in results:
            orig = results['original']
            impr = results['improved']
            
            print(f"\n{'='*60}")
            print("IMPROVEMENTS SUMMARY")
            print(f"{'='*60}")
            
            time_improvement = ((orig.training_time - impr.training_time) / orig.training_time) * 100
            print(f"Training time: {time_improvement:+.1f}% {'faster' if time_improvement > 0 else 'slower'}")
            
            memory_improvement = ((orig.memory_usage_mb - impr.memory_usage_mb) / orig.memory_usage_mb) * 100
            print(f"Memory usage: {memory_improvement:+.1f}% {'less' if memory_improvement > 0 else 'more'}")