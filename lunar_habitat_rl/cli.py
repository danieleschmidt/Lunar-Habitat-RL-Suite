"""Command-line interface for the Lunar Habitat RL Suite."""

import typer
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional, List
import json
import time

from .environments import make_lunar_env, LunarHabitatEnv
from .algorithms import RandomAgent, HeuristicAgent
from .core import HabitatConfig
from .utils import setup_logging, get_logger

app = typer.Typer(help="🌙 Lunar Habitat RL Suite - NASA TRL 6 Reinforcement Learning Environment")
console = Console()
logger = get_logger("cli")


@app.command()
def info():
    """Display information about the Lunar Habitat RL Suite."""
    console.print("\n🌙 [bold blue]Lunar Habitat RL Suite[/bold blue]", style="bold")
    console.print("NASA TRL 6 Reinforcement Learning Environment for Autonomous Life Support Systems\n")
    
    table = Table(title="Environment Specifications")
    table.add_column("Component", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row("🫁 Life Support", "O₂/CO₂/N₂ balance, pressure control")
    table.add_row("🌡️ Thermal", "Day/night cycles (-173°C to 127°C)")
    table.add_row("⚡ Power", "Solar + battery + fuel cell optimization")
    table.add_row("💧 Water", "Recycling and purification systems")
    table.add_row("🚨 Emergency", "Micrometeorite impacts, system failures")
    table.add_row("👥 Crew", "Up to 12 crew members with health/stress modeling")
    
    console.print(table)
    
    console.print("\n[bold green]✅ Ready for NASA Artemis missions![/bold green]\n")


@app.command()
def create_env(
    config: Optional[str] = typer.Option("nasa_reference", help="Configuration preset"),
    crew_size: int = typer.Option(4, help="Number of crew members"),
    scenario: str = typer.Option("nominal_operations", help="Mission scenario"),
    difficulty: str = typer.Option("nominal", help="Difficulty level"),
    output: Optional[str] = typer.Option(None, help="Output file for environment specs")
):
    """Create and validate a lunar habitat environment."""
    console.print(f"🏗️  Creating lunar habitat environment...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing environment...", total=None)
        
        try:
            # Create environment
            env = LunarHabitatEnv(
                config=config,
                crew_size=crew_size,
                scenario=scenario,
                difficulty=difficulty
            )
            progress.update(task, description="Environment created successfully!")
            time.sleep(0.5)
            
            # Test environment
            progress.update(task, description="Running validation tests...")
            obs, info = env.reset(seed=42)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            progress.update(task, description="Validation completed!")
            
        except Exception as e:
            console.print(f"❌ Error creating environment: {e}")
            raise typer.Exit(1)
    
    # Display environment information
    table = Table(title="Environment Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Config Preset", config)
    table.add_row("Crew Size", str(crew_size))
    table.add_row("Scenario", scenario)
    table.add_row("Difficulty", difficulty)
    table.add_row("Observation Space", str(env.observation_space.shape))
    table.add_row("Action Space", str(env.action_space.shape))
    table.add_row("Max Episode Steps", str(env.max_steps))
    
    console.print(table)
    
    if output:
        # Save environment specifications
        env_spec = {
            'config': config,
            'crew_size': crew_size,
            'scenario': scenario,
            'difficulty': difficulty,
            'observation_space_shape': env.observation_space.shape,
            'action_space_shape': env.action_space.shape,
            'max_steps': env.max_steps,
            'created_at': time.time()
        }
        
        with open(output, 'w') as f:
            json.dump(env_spec, f, indent=2)
        
        console.print(f"✅ Environment specifications saved to: {output}")
    
    env.close()
    console.print("🎉 Environment validation successful!")


@app.command()
def run_baseline(
    agent: str = typer.Option("random", help="Baseline agent type"),
    episodes: int = typer.Option(10, help="Number of episodes to run"),
    config: str = typer.Option("nasa_reference", help="Environment configuration"),
    crew_size: int = typer.Option(4, help="Number of crew members"),
    render: bool = typer.Option(False, help="Enable rendering"),
    output: Optional[str] = typer.Option(None, help="Output file for results")
):
    """Run baseline agent on lunar habitat environment."""
    console.print(f"🤖 Running {agent} agent for {episodes} episodes...")
    
    # Create environment
    env = LunarHabitatEnv(
        config=config,
        crew_size=crew_size,
        render_mode="human" if render else None
    )
    
    # Create agent
    if agent == "random":
        baseline_agent = RandomAgent(env.action_space, seed=42)
    elif agent == "heuristic":
        baseline_agent = HeuristicAgent(env.action_space)
    else:
        console.print(f"❌ Unknown agent type: {agent}")
        console.print("Available agents: random, heuristic")
        raise typer.Exit(1)
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_metrics = []
    
    with Progress(console=console) as progress:
        task = progress.add_task(f"Running {agent} agent...", total=episodes)
        
        for episode in range(episodes):
            obs, info = env.reset(seed=42 + episode)
            episode_reward = 0.0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = baseline_agent.predict(obs)
                obs, reward, done, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Extract episode metrics
            metrics = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'survival_time': step_info.get('survival_time', 0.0),
                'crew_health': step_info.get('crew_health', 'N/A'),
                'success': not done  # Not terminated = success
            }
            episode_metrics.append(metrics)
            
            progress.update(task, advance=1)
    
    env.close()
    
    # Display results
    results_table = Table(title=f"{agent.title()} Agent Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    
    import numpy as np
    results_table.add_row("Episodes", str(episodes))
    results_table.add_row("Mean Reward", f"{np.mean(episode_rewards):.2f}")
    results_table.add_row("Std Reward", f"{np.std(episode_rewards):.2f}")
    results_table.add_row("Mean Length", f"{np.mean(episode_lengths):.1f}")
    results_table.add_row("Success Rate", f"{sum(1 for m in episode_metrics if m['success']) / episodes * 100:.1f}%")
    
    console.print(results_table)
    
    if output:
        # Save detailed results
        results = {
            'agent': agent,
            'episodes': episodes,
            'config': config,
            'crew_size': crew_size,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_metrics': episode_metrics,
            'summary': {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'mean_length': float(np.mean(episode_lengths)),
                'success_rate': sum(1 for m in episode_metrics if m['success']) / episodes
            },
            'timestamp': time.time()
        }
        
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"✅ Results saved to: {output}")


@app.command()
def benchmark(
    agents: List[str] = typer.Option(["random", "heuristic"], help="Agent types to benchmark"),
    episodes: int = typer.Option(50, help="Episodes per agent"),
    config: str = typer.Option("nasa_reference", help="Environment configuration"),
    output: Optional[str] = typer.Option("benchmark_results.json", help="Output file")
):
    """Benchmark multiple baseline agents."""
    console.print(f"🏁 Benchmarking agents: {', '.join(agents)}")
    
    env = LunarHabitatEnv(config=config)
    benchmark_results = {}
    
    for agent_name in agents:
        console.print(f"\n🤖 Testing {agent_name} agent...")
        
        # Create agent
        if agent_name == "random":
            agent = RandomAgent(env.action_space, seed=42)
        elif agent_name == "heuristic":
            agent = HeuristicAgent(env.action_space)
        else:
            console.print(f"⚠️  Skipping unknown agent: {agent_name}")
            continue
        
        # Run episodes
        episode_rewards = []
        survival_times = []
        
        with Progress(console=console) as progress:
            task = progress.add_task(f"{agent_name}...", total=episodes)
            
            for episode in range(episodes):
                obs, info = env.reset(seed=42 + episode)
                episode_reward = 0.0
                done = False
                truncated = False
                
                while not (done or truncated):
                    action = agent.predict(obs)
                    obs, reward, done, truncated, step_info = env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                survival_times.append(step_info.get('survival_time', 0.0))
                progress.update(task, advance=1)
        
        # Store results
        import numpy as np
        benchmark_results[agent_name] = {
            'episodes': episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_survival_time': float(np.mean(survival_times)),
            'success_rate': sum(1 for t in survival_times if t > 1.0) / episodes,
            'episode_rewards': episode_rewards,
            'survival_times': survival_times
        }
    
    env.close()
    
    # Display benchmark table
    benchmark_table = Table(title="Benchmark Results")
    benchmark_table.add_column("Agent", style="cyan")
    benchmark_table.add_column("Mean Reward", style="white")
    benchmark_table.add_column("Success Rate", style="green")
    benchmark_table.add_column("Survival Time", style="yellow")
    
    for agent_name, results in benchmark_results.items():
        benchmark_table.add_row(
            agent_name.title(),
            f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
            f"{results['success_rate']*100:.1f}%",
            f"{results['mean_survival_time']:.1f} sols"
        )
    
    console.print(benchmark_table)
    
    if output:
        # Save benchmark results
        final_results = {
            'benchmark_config': {
                'agents': agents,
                'episodes': episodes,
                'env_config': config
            },
            'results': benchmark_results,
            'timestamp': time.time()
        }
        
        with open(output, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        console.print(f"✅ Benchmark results saved to: {output}")


@app.command()
def validate_config(
    config_file: str = typer.Argument(..., help="Path to configuration file"),
    strict: bool = typer.Option(False, help="Enable strict validation")
):
    """Validate a habitat configuration file."""
    console.print(f"🔍 Validating configuration: {config_file}")
    
    try:
        # Load configuration
        if config_file.endswith('.json'):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            config = HabitatConfig(**config_data)
        else:
            # Assume it's a preset name
            config = HabitatConfig.from_preset(config_file)
        
        # Validate configuration
        from .utils.validation import validate_config
        validated_config = validate_config(config)
        
        console.print("✅ Configuration is valid!")
        
        # Display configuration summary
        config_table = Table(title="Configuration Summary")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Habitat Volume", f"{validated_config.volume} m³")
        config_table.add_row("Nominal Pressure", f"{validated_config.pressure_nominal} kPa")
        config_table.add_row("O₂ Nominal", f"{validated_config.o2_nominal} kPa")
        config_table.add_row("CO₂ Limit", f"{validated_config.co2_limit} kPa")
        config_table.add_row("Crew Size", str(validated_config.crew.size))
        config_table.add_row("Mission Duration", f"{validated_config.scenario.duration_days} days")
        
        console.print(config_table)
        
    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}")
        raise typer.Exit(1)


@app.command()  
def train(
    algorithm: str = typer.Option("random", help="Training algorithm"),
    timesteps: int = typer.Option(10000, help="Total training timesteps"),
    config: str = typer.Option("nasa_reference", help="Environment configuration"),
    output_dir: str = typer.Option("./models", help="Output directory for trained models"),
    tensorboard_log: Optional[str] = typer.Option(None, help="Tensorboard log directory")
):
    """Train an RL agent on the lunar habitat environment."""
    console.print(f"🎓 Training {algorithm} agent for {timesteps} timesteps...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = LunarHabitatEnv(config=config)
    
    # Create and train agent
    if algorithm == "random":
        agent = RandomAgent(env.action_space, seed=42)
        
        with Progress(console=console) as progress:
            task = progress.add_task("Training (collecting data)...", total=None)
            
            stats = agent.train(env, total_timesteps=timesteps)
            
            progress.update(task, description="Training completed!")
        
        # Save agent
        model_path = output_path / f"{algorithm}_agent.npz"
        agent.save(str(model_path))
        
    elif algorithm == "heuristic":
        agent = HeuristicAgent(env.action_space)
        
        with Progress(console=console) as progress:
            task = progress.add_task("Training (collecting data)...", total=None)
            
            stats = agent.train(env, total_timesteps=timesteps)
            
            progress.update(task, description="Training completed!")
        
        # Save agent
        model_path = output_path / f"{algorithm}_agent.npz"
        agent.save(str(model_path))
        
    else:
        console.print(f"❌ Algorithm '{algorithm}' not implemented yet.")
        console.print("Available algorithms: random, heuristic")
        raise typer.Exit(1)
    
    env.close()
    
    # Display training results
    results_table = Table(title="Training Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    
    if 'episodes' in stats:
        results_table.add_row("Episodes", str(stats['episodes']))
    if 'mean_reward' in stats:
        results_table.add_row("Mean Reward", f"{stats['mean_reward']:.2f}")
    if 'mean_length' in stats:
        results_table.add_row("Mean Episode Length", f"{stats['mean_length']:.1f}")
    
    console.print(results_table)
    console.print(f"✅ Model saved to: {model_path}")


if __name__ == "__main__":
    app()