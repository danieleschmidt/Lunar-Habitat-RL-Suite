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

# Try imports with fallbacks for lightweight mode
try:
    from .environments import make_lunar_env, LunarHabitatEnv
except ImportError:
    from .environments.lightweight_habitat import make_lunar_env, LunarHabitatEnv

try:
    from .algorithms import RandomAgent, HeuristicAgent
except ImportError:
    from .algorithms.lightweight_baselines import RandomAgent, HeuristicAgent

try:
    from .core import HabitatConfig
except ImportError:
    from .core.lightweight_config import HabitatConfig

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
def health_check(
    config: str = typer.Option("nasa_reference", help="Configuration to validate"),
    crew_size: int = typer.Option(4, help="Number of crew members"),
    verbose: bool = typer.Option(False, help="Detailed health check output")
):
    """Perform comprehensive health check of the habitat environment."""
    console.print("🏥 [bold blue]Running Habitat Health Check[/bold blue]\n")
    
    from .utils.robust_validation import EnvironmentHealthChecker
    
    try:
        # Initialize health checker
        health_checker = EnvironmentHealthChecker()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running health checks...", total=None)
            
            # Run health checks
            results = health_checker.run_full_health_check(
                config=config,
                crew_size=crew_size,
                verbose=verbose
            )
            
            progress.update(task, description="Health check completed!")
        
        # Display results
        health_table = Table(title="Health Check Results")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="white")
        health_table.add_column("Details", style="white")
        
        for component, result in results.items():
            status_style = "green" if result['status'] == 'PASS' else "red" if result['status'] == 'FAIL' else "yellow"
            health_table.add_row(
                component.title(),
                f"[{status_style}]{result['status']}[/{status_style}]",
                result.get('details', 'N/A')
            )
        
        console.print(health_table)
        
        # Overall status
        failed_checks = [c for c, r in results.items() if r['status'] == 'FAIL']
        if failed_checks:
            console.print(f"\n❌ [red]Health check failed[/red] - {len(failed_checks)} component(s) failed")
            console.print(f"Failed components: {', '.join(failed_checks)}")
            raise typer.Exit(1)
        else:
            console.print("\n✅ [green]All health checks passed![/green]")
            
    except Exception as e:
        console.print(f"❌ Health check failed with error: {e}")
        raise typer.Exit(1)


@app.command()
def quick_test(
    config: str = typer.Option("nasa_reference", help="Environment configuration"),
    episodes: int = typer.Option(5, help="Number of test episodes"),
    agent: str = typer.Option("heuristic", help="Agent type to test"),
    seed: int = typer.Option(42, help="Random seed for reproducibility")
):
    """Quick test of environment with a baseline agent."""
    console.print(f"🚀 [bold blue]Quick Test - {agent} agent for {episodes} episodes[/bold blue]\n")
    
    try:
        from .algorithms.lightweight_baselines import get_baseline_agent
        from .environments.lightweight_habitat import LunarHabitatEnv
        
        # Create environment and agent
        env = LunarHabitatEnv(crew_size=4)
        test_agent = get_baseline_agent(agent)
        
        results = []
        
        with Progress(console=console) as progress:
            task = progress.add_task(f"Running {agent} test...", total=episodes)
            
            for episode in range(episodes):
                obs, info = env.reset(seed=seed + episode)
                episode_reward = 0.0
                episode_length = 0
                done = False
                truncated = False
                
                while not (done or truncated):
                    action, _ = test_agent.predict(obs)
                    obs, reward, done, truncated, step_info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if episode_length > 1000:  # Safety limit
                        truncated = True
                
                results.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'length': episode_length,
                    'status': step_info.get('status', 'unknown')
                })
                
                progress.update(task, advance=1)
        
        env.close()
        
        # Display results
        test_table = Table(title="Quick Test Results")
        test_table.add_column("Episode", style="cyan")
        test_table.add_column("Reward", style="white")
        test_table.add_column("Length", style="white")
        test_table.add_column("Status", style="white")
        
        for result in results:
            status_style = "green" if result['status'] == 'nominal' else "yellow" if result['status'] in ['caution', 'warning'] else "red"
            test_table.add_row(
                str(result['episode']),
                f"{result['reward']:.2f}",
                str(result['length']),
                f"[{status_style}]{result['status']}[/{status_style}]"
            )
        
        console.print(test_table)
        
        # Summary statistics
        avg_reward = sum(r['reward'] for r in results) / len(results)
        avg_length = sum(r['length'] for r in results) / len(results)
        success_rate = len([r for r in results if r['status'] == 'nominal']) / len(results) * 100
        
        summary_table = Table(title="Summary Statistics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Average Reward", f"{avg_reward:.2f}")
        summary_table.add_row("Average Length", f"{avg_length:.1f}")
        summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        console.print(summary_table)
        
        if success_rate >= 80:
            console.print("\n✅ [green]Test passed - Environment is functioning well![/green]")
        elif success_rate >= 60:
            console.print("\n⚠️  [yellow]Test passed with warnings - Some issues detected[/yellow]")
        else:
            console.print("\n❌ [red]Test failed - Significant issues detected[/red]")
            
    except Exception as e:
        console.print(f"❌ Quick test failed: {e}")
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


@app.command()
def demo(
    scenario: str = typer.Option("emergency", help="Demo scenario to run"),
    duration: int = typer.Option(100, help="Number of steps to run"),
    agent: str = typer.Option("heuristic", help="Agent to use for demo")
):
    """Run an interactive demo of the habitat environment."""
    console.print(f"🎪 [bold blue]Habitat Demo - {scenario} scenario[/bold blue]\n")
    
    try:
        from .algorithms.lightweight_baselines import get_baseline_agent
        from .environments.lightweight_habitat import LunarHabitatEnv
        
        # Create environment and agent
        env = LunarHabitatEnv()
        demo_agent = get_baseline_agent(agent)
        
        obs, info = env.reset(seed=42)
        
        # Demo state tracking
        step_data = []
        
        console.print("🏁 Starting demo...\n")
        
        for step in range(duration):
            # Get agent action
            action, _ = demo_agent.predict(obs)
            
            # Execute step
            obs, reward, done, truncated, step_info = env.step(action)
            
            # Store step data
            step_data.append({
                'step': step,
                'reward': reward,
                'status': step_info.get('status', 'unknown'),
                'o2_pressure': obs[0] if len(obs) > 0 else 0,
                'co2_pressure': obs[1] if len(obs) > 1 else 0,
                'battery_charge': obs[8] if len(obs) > 8 else 0
            })
            
            # Display periodic updates
            if step % 20 == 0 or done or truncated:
                status_style = "green" if step_info.get('status') == 'nominal' else "yellow" if step_info.get('status') in ['caution', 'warning'] else "red"
                console.print(
                    f"Step {step:3d}: "
                    f"O₂: {obs[0]:.1f} kPa, "
                    f"CO₂: {obs[1]:.1f} kPa, "
                    f"Battery: {obs[8]:.1f}%, "
                    f"Status: [{status_style}]{step_info.get('status', 'unknown')}[/{status_style}], "
                    f"Reward: {reward:.2f}"
                )
            
            if done or truncated:
                break
        
        env.close()
        
        # Demo summary
        console.print("\n📊 [bold]Demo Summary[/bold]")
        
        summary_table = Table(title="Final State")
        summary_table.add_column("Parameter", style="cyan")
        summary_table.add_column("Value", style="white")
        
        final_data = step_data[-1] if step_data else {}
        summary_table.add_row("Steps Completed", str(len(step_data)))
        summary_table.add_row("Final Status", final_data.get('status', 'unknown'))
        summary_table.add_row("Total Reward", f"{sum(s['reward'] for s in step_data):.2f}")
        summary_table.add_row("Final O₂ Pressure", f"{final_data.get('o2_pressure', 0):.1f} kPa")
        summary_table.add_row("Final CO₂ Pressure", f"{final_data.get('co2_pressure', 0):.1f} kPa")
        summary_table.add_row("Final Battery", f"{final_data.get('battery_charge', 0):.1f}%")
        
        console.print(summary_table)
        
        if done:
            console.print("\n🚨 [red]Demo ended due to critical failure![/red]")
        elif final_data.get('status') == 'nominal':
            console.print("\n✅ [green]Demo completed successfully![/green]")
        else:
            console.print("\n⚠️  [yellow]Demo completed with warnings[/yellow]")
            
    except Exception as e:
        console.print(f"❌ Demo failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_agents():
    """List all available baseline agents."""
    console.print("🤖 [bold blue]Available Baseline Agents[/bold blue]\n")
    
    try:
        from .algorithms.lightweight_baselines import BASELINE_AGENTS
        
        agents_table = Table(title="Baseline Agents")
        agents_table.add_column("Agent Type", style="cyan")
        agents_table.add_column("Description", style="white")
        agents_table.add_column("Use Case", style="green")
        
        agent_descriptions = {
            'random': ('Random action selection', 'Baseline comparison, environment testing'),
            'heuristic': ('Rule-based control logic', 'Practical baseline, demonstration'),
            'pid': ('PID controller-based', 'Precise control, engineering validation'),
            'greedy': ('Greedy action selection', 'Short-term optimization')
        }
        
        for agent_type in BASELINE_AGENTS.keys():
            desc, use_case = agent_descriptions.get(agent_type, ('Unknown', 'Unknown'))
            agents_table.add_row(agent_type, desc, use_case)
        
        console.print(agents_table)
        
    except Exception as e:
        console.print(f"❌ Failed to list agents: {e}")


@app.command()
def system_info():
    """Display system information and dependencies."""
    console.print("💻 [bold blue]System Information[/bold blue]\n")
    
    import sys
    import platform
    
    system_table = Table(title="System Details")
    system_table.add_column("Component", style="cyan")
    system_table.add_column("Value", style="white")
    
    system_table.add_row("Python Version", f"{sys.version.split()[0]}")
    system_table.add_row("Platform", platform.platform())
    system_table.add_row("Architecture", platform.machine())
    system_table.add_row("Processor", platform.processor() or "Unknown")
    
    console.print(system_table)
    
    # Check for optional dependencies
    console.print("\n📦 [bold blue]Dependency Status[/bold blue]")
    
    deps_table = Table(title="Dependencies")
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="white")
    deps_table.add_column("Notes", style="white")
    
    # Check common dependencies
    dependencies = [
        ('numpy', 'Core numerical operations'),
        ('gymnasium', 'RL environment interface'),
        ('stable-baselines3', 'RL algorithms'),
        ('torch', 'Deep learning'),
        ('matplotlib', 'Plotting and visualization'),
        ('scipy', 'Scientific computing')
    ]
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            status = "[green]✓ Available[/green]"
        except ImportError:
            status = "[yellow]✗ Not installed[/yellow]"
        
        deps_table.add_row(dep_name, status, description)
    
    console.print(deps_table)
    
    console.print("\n💡 [italic]Note: Lunar Habitat RL Suite can run in lightweight mode without heavy dependencies[/italic]")


if __name__ == "__main__":
    app()