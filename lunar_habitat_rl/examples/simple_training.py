"""Simple training examples for Lunar Habitat RL Suite - Generation 1 enhancements."""

import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class SimpleTrainingLoop:
    """Basic training loop for lightweight agents without heavy ML dependencies."""
    
    def __init__(self, env, agent, log_interval: int = 100):
        """Initialize training loop.
        
        Args:
            env: Environment instance
            agent: Agent instance
            log_interval: Steps between logging
        """
        self.env = env
        self.agent = agent
        self.log_interval = log_interval
        self.training_data = []
        self.episode_data = []
        
    def train(self, total_timesteps: int = 10000, max_episode_length: int = 1000) -> Dict[str, Any]:
        """Run simple training loop.
        
        Args:
            total_timesteps: Total number of timesteps to train
            max_episode_length: Maximum steps per episode
            
        Returns:
            Training statistics
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        total_steps = 0
        episode_count = 0
        start_time = time.time()
        
        while total_steps < total_timesteps:
            episode_stats = self._run_episode(max_episode_length, total_steps, total_timesteps)
            
            self.episode_data.append(episode_stats)
            total_steps += episode_stats['length']
            episode_count += 1
            
            # Log progress
            if episode_count % 10 == 0 or total_steps >= total_timesteps:
                avg_reward = sum(ep['reward'] for ep in self.episode_data[-10:]) / min(10, len(self.episode_data))
                avg_length = sum(ep['length'] for ep in self.episode_data[-10:]) / min(10, len(self.episode_data))
                
                print(f"Episode {episode_count}, Step {total_steps}/{total_timesteps}: "
                      f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
        
        training_time = time.time() - start_time
        
        # Compile training statistics
        stats = self._compile_training_stats(episode_count, total_steps, training_time)
        
        print(f"Training completed in {training_time:.2f}s")
        print(f"Episodes: {episode_count}, Average reward: {stats['avg_reward']:.2f}")
        
        return stats
    
    def _run_episode(self, max_length: int, current_step: int, total_steps: int) -> Dict[str, Any]:
        """Run a single episode."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False
        
        episode_start_time = time.time()
        
        while not (done or truncated) and episode_length < max_length and current_step + episode_length < total_steps:
            # Get action from agent
            action, agent_state = self.agent.predict(obs)
            
            # Take step in environment
            obs, reward, done, truncated, step_info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Store step data for learning (if agent supports it)
            step_data = {
                'obs': obs[:],  # Copy observation
                'action': action[:],  # Copy action
                'reward': reward,
                'done': done,
                'info': step_info
            }
            self.training_data.append(step_data)
            
            # Optional: Update agent with experience (for learning agents)
            if hasattr(self.agent, 'update') and len(self.training_data) >= 100:
                # Simple batch update every 100 steps
                recent_data = self.training_data[-100:]
                self.agent.update(recent_data)
        
        episode_time = time.time() - episode_start_time
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'time': episode_time,
            'final_status': step_info.get('status', 'unknown') if 'step_info' in locals() else 'unknown',
            'terminated': done,
            'truncated': truncated
        }
    
    def _compile_training_stats(self, episode_count: int, total_steps: int, training_time: float) -> Dict[str, Any]:
        """Compile training statistics."""
        if not self.episode_data:
            return {'error': 'No episodes completed'}
        
        rewards = [ep['reward'] for ep in self.episode_data]
        lengths = [ep['length'] for ep in self.episode_data]
        
        stats = {
            'episodes': episode_count,
            'total_steps': total_steps,
            'training_time': training_time,
            'avg_reward': sum(rewards) / len(rewards),
            'std_reward': self._std(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'avg_length': sum(lengths) / len(lengths),
            'std_length': self._std(lengths),
            'success_rate': len([ep for ep in self.episode_data if ep['final_status'] == 'nominal']) / len(self.episode_data),
            'steps_per_second': total_steps / training_time if training_time > 0 else 0
        }
        
        return stats
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def save_training_data(self, filepath: str):
        """Save training data to file."""
        data = {
            'episode_data': self.episode_data,
            'training_data': self.training_data[-1000:],  # Keep last 1000 steps
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class SimpleEvaluator:
    """Simple evaluation for trained agents."""
    
    @staticmethod
    def evaluate_agent(env, agent, num_episodes: int = 10, max_episode_length: int = 1000) -> Dict[str, Any]:
        """Evaluate agent performance.
        
        Args:
            env: Environment instance
            agent: Agent to evaluate
            num_episodes: Number of evaluation episodes
            max_episode_length: Maximum steps per episode
            
        Returns:
            Evaluation results
        """
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        episode_results = []
        
        for episode in range(num_episodes):
            obs, info = env.reset(seed=42 + episode)  # Reproducible evaluation
            episode_reward = 0.0
            episode_length = 0
            done = False
            truncated = False
            
            safety_violations = []
            
            while not (done or truncated) and episode_length < max_episode_length:
                action, _ = agent.predict(obs)
                obs, reward, done, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check for safety issues
                if step_info.get('status') in ['critical', 'warning']:
                    safety_violations.append(step_info.get('status'))
            
            episode_results.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'final_status': step_info.get('status', 'unknown') if 'step_info' in locals() else 'unknown',
                'safety_violations': len(safety_violations),
                'terminated': done,
                'truncated': truncated
            })
            
            if (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")
        
        # Compile evaluation statistics
        rewards = [ep['reward'] for ep in episode_results]
        lengths = [ep['length'] for ep in episode_results]
        
        stats = {
            'num_episodes': num_episodes,
            'mean_reward': sum(rewards) / len(rewards),
            'std_reward': SimpleTrainingLoop._std(None, rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'mean_length': sum(lengths) / len(lengths),
            'success_rate': len([ep for ep in episode_results if ep['final_status'] == 'nominal']) / len(episode_results),
            'safety_violation_rate': len([ep for ep in episode_results if ep['safety_violations'] > 0]) / len(episode_results),
            'episode_results': episode_results
        }
        
        print(f"Evaluation completed:")
        print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Safety violation rate: {stats['safety_violation_rate']*100:.1f}%")
        
        return stats


class TrainingManager:
    """Manager for running different training experiments."""
    
    def __init__(self, output_dir: str = "./training_results"):
        """Initialize training manager.
        
        Args:
            output_dir: Directory to save training results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_agent_comparison(self, env_creator, agent_configs: List[Dict[str, Any]], 
                           timesteps: int = 5000) -> Dict[str, Any]:
        """Compare multiple agents.
        
        Args:
            env_creator: Function that creates environment
            agent_configs: List of agent configurations
            timesteps: Training timesteps per agent
            
        Returns:
            Comparison results
        """
        print(f"Running agent comparison with {len(agent_configs)} agents...")
        
        results = {}
        
        for i, config in enumerate(agent_configs):
            agent_name = config['name']
            agent_class = config['class']
            agent_kwargs = config.get('kwargs', {})
            
            print(f"\n--- Training {agent_name} ({i+1}/{len(agent_configs)}) ---")
            
            # Create fresh environment and agent
            env = env_creator()
            agent = agent_class(**agent_kwargs)
            
            # Train agent
            trainer = SimpleTrainingLoop(env, agent)
            training_stats = trainer.train(total_timesteps=timesteps)
            
            # Evaluate agent
            evaluation_stats = SimpleEvaluator.evaluate_agent(env, agent, num_episodes=10)
            
            # Save results
            agent_results = {
                'config': config,
                'training_stats': training_stats,
                'evaluation_stats': evaluation_stats,
                'timestamp': time.time()
            }
            
            results[agent_name] = agent_results
            
            # Save individual agent results
            result_file = self.output_dir / f"{agent_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(agent_results, f, indent=2)
            
            env.close()
        
        # Save comparison results
        comparison_file = self.output_dir / "agent_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _print_comparison_summary(self, results: Dict[str, Any]):
        """Print summary of agent comparison."""
        print("\n" + "="*50)
        print("AGENT COMPARISON SUMMARY")
        print("="*50)
        
        for agent_name, agent_results in results.items():
            eval_stats = agent_results['evaluation_stats']
            train_stats = agent_results['training_stats']
            
            print(f"\n{agent_name}:")
            print(f"  Training Episodes: {train_stats.get('episodes', 'N/A')}")
            print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Success Rate: {eval_stats['success_rate']*100:.1f}%")
            print(f"  Safety Violations: {eval_stats['safety_violation_rate']*100:.1f}%")
        
        # Find best agent
        best_agent = max(results.items(), key=lambda x: x[1]['evaluation_stats']['mean_reward'])
        print(f"\nBest performing agent: {best_agent[0]}")


def run_basic_training_example():
    """Run a basic training example with lightweight agents."""
    print("Running basic training example...")
    
    try:
        from ..environments.lightweight_habitat import LunarHabitatEnv
        from ..algorithms.lightweight_baselines import get_baseline_agent
        
        # Create environment
        env = LunarHabitatEnv(crew_size=4)
        
        # Test different agents
        agents_to_test = ['random', 'heuristic', 'pid', 'greedy']
        
        print("\nTesting agents:")
        for agent_type in agents_to_test:
            print(f"\n--- Testing {agent_type} agent ---")
            
            try:
                agent = get_baseline_agent(agent_type)
                
                # Short training run
                trainer = SimpleTrainingLoop(env, agent, log_interval=50)
                stats = trainer.train(total_timesteps=500, max_episode_length=100)
                
                print(f"Training completed for {agent_type}:")
                print(f"  Episodes: {stats['episodes']}")
                print(f"  Average reward: {stats['avg_reward']:.2f}")
                print(f"  Success rate: {stats['success_rate']*100:.1f}%")
                
            except Exception as e:
                print(f"Error testing {agent_type} agent: {e}")
        
        env.close()
        print("\nBasic training example completed!")
        
    except Exception as e:
        print(f"Error in basic training example: {e}")


def run_comparative_study():
    """Run a comparative study of different baseline agents."""
    print("Running comparative study of baseline agents...")
    
    try:
        from ..environments.lightweight_habitat import LunarHabitatEnv
        from ..algorithms.lightweight_baselines import (
            RandomAgent, HeuristicAgent, PIDControllerAgent, GreedyAgent
        )
        
        # Define agent configurations
        agent_configs = [
            {
                'name': 'random',
                'class': RandomAgent,
                'kwargs': {'action_dims': 22}
            },
            {
                'name': 'heuristic',
                'class': HeuristicAgent,
                'kwargs': {'action_dims': 22}
            },
            {
                'name': 'pid',
                'class': PIDControllerAgent,
                'kwargs': {'action_dims': 22}
            },
            {
                'name': 'greedy',
                'class': GreedyAgent,
                'kwargs': {'action_dims': 22, 'n_samples': 5}
            }
        ]
        
        # Environment creator
        def create_env():
            return LunarHabitatEnv(crew_size=4)
        
        # Run comparison
        manager = TrainingManager(output_dir="./training_results")
        results = manager.run_agent_comparison(
            env_creator=create_env,
            agent_configs=agent_configs,
            timesteps=2000
        )
        
        print("\nComparative study completed!")
        print(f"Results saved to: {manager.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error in comparative study: {e}")
        return None


if __name__ == "__main__":
    print("Lunar Habitat RL - Simple Training Examples")
    print("=" * 50)
    
    # Run basic example
    run_basic_training_example()
    
    print("\n" + "=" * 50)
    
    # Run comparative study
    run_comparative_study()