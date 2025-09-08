import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import YamlConfig
import re
from collections import defaultdict

# Import from the main choice model analysis script
from scripts.choice_analysis.analyse_choices import get_unique_settings, format_setting_name


def discover_seed_runs(base_results_dir: str, partial_run_name: str) -> list:
    """
    Discover all seed runs matching the pattern <partial_run_name>_seed<number>
    
    Args:
        base_results_dir: Base directory containing all runs
        partial_run_name: Partial run name without seed suffix
        
    Returns:
        List of full run names matching the pattern
    """
    seed_runs = []
    pattern = re.compile(rf'^{re.escape(partial_run_name)}_seed\d+$')
    
    if os.path.exists(base_results_dir):
        for item in os.listdir(base_results_dir):
            if pattern.match(item):
                seed_runs.append(item)
    
    return sorted(seed_runs)


def discover_training_steps(base_dir: str) -> list:
    """
    Discover all lora_adapters_step_X directories and return sorted step numbers.
    """
    steps = []
    pattern = re.compile(r'lora_adapters_step_(\d+)')
    
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            match = pattern.match(item)
            if match:
                steps.append(int(match.group(1)))
    
    return sorted(steps)


def load_step_data_if_exists(results_dir: str, wandb_run_name: str, step: int, unique_settings: list) -> dict:
    """
    Load data for a specific training step if all files exist. Return None if any missing.
    Does NOT run analysis - just loads existing files or returns None.
    
    Returns:
        dict or None: {comparison_key: {'theta_mean': ..., 'alpha_mean': ..., etc.}} or None if missing files
    """
    
    if step == 0:
        # Initial choices
        step_dir = os.path.join(results_dir, "initial_choices")
    else:
        # LoRA step
        step_dir = os.path.join(results_dir, f"forward_sft_choices/{wandb_run_name}/lora_adapters_step_{step}")
    
    # Generate expected comparison files (same logic as original script)
    expected_files = []
    for i, setting1 in enumerate(unique_settings):
        for j, setting2 in enumerate(unique_settings):
            if i >= j:  # Skip self-comparisons and duplicates
                continue
            
            temp1, style1 = setting1
            temp2, style2 = setting2
            
            pos_name = format_setting_name(temp1, style1)
            neg_name = format_setting_name(temp2, style2)
            comparison_name = f"{pos_name}_vs_{neg_name}"
            
            pickle_file = os.path.join(step_dir, f"choice_model_mcmc_{comparison_name}.pkl")
            csv_file = os.path.join(step_dir, f"choice_model_data_{comparison_name}.csv")
            
            expected_files.append((pickle_file, csv_file, (setting1, setting2), pos_name, neg_name))
    
    # Check if all files exist
    missing_files = []
    for pickle_file, csv_file, comparison_key, pos_name, neg_name in expected_files:
        if not os.path.exists(pickle_file) or not os.path.exists(csv_file):
            missing_files.append((pickle_file, csv_file))
    
    # If any files are missing, return None
    if missing_files:
        return None
    
    # Load all the data
    step_data = {}
    for pickle_file, csv_file, comparison_key, pos_name, neg_name in expected_files:
        
        try:
            # Load pickle file for MCMC results
            with open(pickle_file, 'rb') as f:
                mcmc_results = pickle.load(f)
            
            step_data[comparison_key] = {
                'theta_mean': mcmc_results['theta_mean'],
                'theta_std': mcmc_results['theta_std'],
                'prob_theta_positive': mcmc_results['prob_theta_positive'],
                'alpha_mean': mcmc_results['alpha_mean'],
                'alpha_std': mcmc_results['alpha_std'],
                'prob_alpha_positive': mcmc_results['prob_alpha_positive'],
                'pos_name': pos_name,
                'neg_name': neg_name
            }
        except Exception as e:
            print(f"Warning: Error loading {pickle_file}: {e}")
            return None
    
    return step_data


def aggregate_across_seeds(all_seed_data: dict) -> dict:
    """
    Aggregate data across seeds for each timestep and comparison.
    
    Args:
        all_seed_data: {seed_name: {step: {comparison_key: data}}}
        
    Returns:
        dict: {step: {comparison_key: {'theta_mean_across_seeds': ..., 'theta_std_across_seeds': ..., 'n_seeds': ...}}}
    """
    
    # First, collect all timesteps and comparison keys
    all_steps = set()
    all_comparison_keys = set()
    
    for seed_name, seed_data in all_seed_data.items():
        all_steps.update(seed_data.keys())
        for step_data in seed_data.values():
            all_comparison_keys.update(step_data.keys())
    
    all_steps = sorted(all_steps)
    all_comparison_keys = sorted(all_comparison_keys)
    
    # Aggregate data
    aggregated_data = {}
    
    for step in all_steps:
        aggregated_data[step] = {}
        
        for comp_key in all_comparison_keys:
            # Collect values from all seeds that have this step and comparison
            theta_means = []
            alpha_means = []
            pos_names = []
            neg_names = []
            
            for seed_name, seed_data in all_seed_data.items():
                if step in seed_data and comp_key in seed_data[step]:
                    theta_means.append(seed_data[step][comp_key]['theta_mean'])
                    alpha_means.append(seed_data[step][comp_key]['alpha_mean'])
                    pos_names.append(seed_data[step][comp_key]['pos_name'])
                    neg_names.append(seed_data[step][comp_key]['neg_name'])
            
            # Only include if we have data from at least one seed
            if theta_means:
                aggregated_data[step][comp_key] = {
                    'theta_mean_across_seeds': np.mean(theta_means),
                    'theta_std_across_seeds': np.std(theta_means, ddof=1) if len(theta_means) > 1 else 0.0,
                    'alpha_mean_across_seeds': np.mean(alpha_means),
                    'alpha_std_across_seeds': np.std(alpha_means, ddof=1) if len(alpha_means) > 1 else 0.0,
                    'n_seeds': len(theta_means),
                    'pos_name': pos_names[0],  # Should be the same across seeds
                    'neg_name': neg_names[0]   # Should be the same across seeds
                }
    
    return aggregated_data


def plot_cross_seed_trajectories(aggregated_data: dict, output_dir: str, partial_run_name: str):
    """
    Plot theta and alpha trajectories aggregated across seeds.
    """
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Cross-Seed Choice Model Parameter Trajectories\n{partial_run_name}', 
                 fontsize=16, fontweight='bold')
    
    # Get all comparison keys and steps
    all_comparison_keys = set()
    for step_data in aggregated_data.values():
        all_comparison_keys.update(step_data.keys())
    all_comparison_keys = sorted(all_comparison_keys)
    
    steps = sorted(aggregated_data.keys())
    
    # Generate colors for each comparison
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_comparison_keys)))
    
    # Find global y-axis ranges for consistent scaling
    all_theta_vals = []
    all_alpha_vals = []
    
    for step in steps:
        for comp_key in all_comparison_keys:
            if comp_key in aggregated_data[step]:
                data = aggregated_data[step][comp_key]
                theta_mean = data['theta_mean_across_seeds']
                theta_std = data['theta_std_across_seeds']
                alpha_mean = data['alpha_mean_across_seeds']
                alpha_std = data['alpha_std_across_seeds']
                
                all_theta_vals.extend([theta_mean - theta_std, theta_mean + theta_std])
                all_alpha_vals.extend([alpha_mean - alpha_std, alpha_mean + alpha_std])
    
    # Calculate consistent y-axis ranges
    if all_theta_vals:
        theta_min, theta_max = min(all_theta_vals) * 1.1, max(all_theta_vals) * 1.1
    else:
        theta_min, theta_max = -1, 1
    
    if all_alpha_vals:
        alpha_min, alpha_max = min(all_alpha_vals) * 1.1, max(all_alpha_vals) * 1.1
    else:
        alpha_min, alpha_max = -1, 1
    
    for i, comp_key in enumerate(all_comparison_keys):
        color = colors[i]
        
        # Get pos_name and neg_name from the first available step
        pos_name = neg_name = None
        for step in steps:
            if comp_key in aggregated_data[step]:
                pos_name = aggregated_data[step][comp_key]['pos_name']
                neg_name = aggregated_data[step][comp_key]['neg_name']
                break
        
        label = f"Pos: {pos_name}\nNeg: {neg_name}"
        
        # Collect theta and alpha data
        theta_means = []
        theta_stds = []
        alpha_means = []
        alpha_stds = []
        valid_steps = []
        
        for step in steps:
            if comp_key in aggregated_data[step]:
                data = aggregated_data[step][comp_key]
                theta_means.append(data['theta_mean_across_seeds'])
                theta_stds.append(data['theta_std_across_seeds'])
                alpha_means.append(data['alpha_mean_across_seeds'])
                alpha_stds.append(data['alpha_std_across_seeds'])
                valid_steps.append(step)
        
        # Plot theta trajectory
        ax1.errorbar(valid_steps, theta_means, yerr=theta_stds, 
                    color=color, marker='o', label=label, linewidth=2, markersize=6,
                    capsize=4, capthick=2)
        
        # Plot alpha trajectory  
        ax2.errorbar(valid_steps, alpha_means, yerr=alpha_stds,
                    color=color, marker='o', label=label, linewidth=2, markersize=6,
                    capsize=4, capthick=2)
    
    # Format theta plot
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('θ (Setting Preference)', fontsize=12)
    ax1.set_title('Setting Preference Evolution (Cross-Seed)\nθ > 0: Prefers Positive Setting', fontsize=14)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No preference')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(theta_min, theta_max)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Format alpha plot
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('α (Position Bias)', fontsize=12)
    ax2.set_title('Position Bias Evolution (Cross-Seed)\nα > 0: First Position Advantage', fontsize=14)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No bias')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(alpha_min, alpha_max)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'cross_seed_trajectories_{partial_run_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved cross-seed trajectory plot: {plot_file}")


def plot_seed_coverage(aggregated_data: dict, output_dir: str, partial_run_name: str):
    """
    Plot number of seeds reporting at each timestep for each comparison.
    """
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'Seed Coverage Across Training Steps\n{partial_run_name}', 
                 fontsize=16, fontweight='bold')
    
    # Get all comparison keys and steps
    all_comparison_keys = set()
    for step_data in aggregated_data.values():
        all_comparison_keys.update(step_data.keys())
    all_comparison_keys = sorted(all_comparison_keys)
    
    steps = sorted(aggregated_data.keys())
    
    # Generate colors for each comparison
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_comparison_keys)))
    
    # Find global y-axis range
    max_seeds = 0
    for step in steps:
        for comp_key in all_comparison_keys:
            if comp_key in aggregated_data[step]:
                n_seeds = aggregated_data[step][comp_key]['n_seeds']
                max_seeds = max(max_seeds, n_seeds)
    
    y_max = max_seeds * 1.1 if max_seeds > 0 else 5
    
    for i, comp_key in enumerate(all_comparison_keys):
        color = colors[i]
        
        # Get pos_name and neg_name from the first available step
        pos_name = neg_name = None
        for step in steps:
            if comp_key in aggregated_data[step]:
                pos_name = aggregated_data[step][comp_key]['pos_name']
                neg_name = aggregated_data[step][comp_key]['neg_name']
                break
        
        label = f"Pos: {pos_name}\nNeg: {neg_name}"
        
        # Collect seed counts
        seed_counts = []
        valid_steps = []
        
        for step in steps:
            if comp_key in aggregated_data[step]:
                seed_counts.append(aggregated_data[step][comp_key]['n_seeds'])
                valid_steps.append(step)
        
        # Plot seed count trajectory
        ax.plot(valid_steps, seed_counts, color=color, marker='o', label=label, 
               linewidth=2, markersize=6)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Number of Seeds Reporting', fontsize=12)
    ax.set_title('Seed Coverage Over Training Steps', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, y_max)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set integer y-ticks
    ax.set_yticks(range(0, int(y_max) + 1))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'seed_coverage_{partial_run_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved seed coverage plot: {plot_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.choice_analysis.scripts.choice_analysis.analyse_choices_aggregation /path/to/config.yaml <partial_run_name>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    partial_run_name = sys.argv[2]
    
    print(f"Aggregating choice analysis across seeds:")
    print(f"  Config: {config_path}")
    print(f"  Partial run name: {partial_run_name}")
    
    # Load configuration
    args = YamlConfig(config_path)
    
    # Set up directories
    split_name = "test"
    base_results_dir = f"results_and_data/modal_results/results/main/{args.args_name}"
    seed_search_dir = os.path.join(base_results_dir, split_name, "forward_sft_choices")
    
    # Discover seed runs
    seed_runs = discover_seed_runs(seed_search_dir, partial_run_name)
    
    if not seed_runs:
        raise ValueError(f"No seed runs found matching pattern: {partial_run_name}_seed*")
    
    print(f"Found {len(seed_runs)} seed runs: {seed_runs}")
    
    # Load initial choices from first seed to discover unique settings
    initial_results_file = os.path.join(base_results_dir, split_name, "initial_choices/choice_results.csv")
    
    if not os.path.exists(initial_results_file):
        raise FileNotFoundError(f"Initial results file not found: {initial_results_file}")
    
    raw_results = pd.read_csv(initial_results_file)
    unique_settings = get_unique_settings(raw_results)
    
    print(f"Found {len(unique_settings)} unique settings:")
    for setting in unique_settings:
        temp, style = setting
        print(f"  Temperature: {temp}, Style: {style}")
    
    # Load data for all seeds and steps
    print("\nLoading data for all seeds...")
    all_seed_data = {}  # {seed_name: {step: {comparison_key: data}}}
    
    for seed_run in seed_runs:
        print(f"  Processing seed: {seed_run}")
        results_dir = os.path.join(base_results_dir, split_name)
        wandb_base_dir = os.path.join(results_dir, f"forward_sft_choices/{seed_run}")
        
        # Discover training steps for this seed
        training_steps = discover_training_steps(wandb_base_dir)
        all_steps = [0] + training_steps  # Include initial choices as step 0
        
        print(f"    Found steps: {all_steps}")
        
        seed_data = {}
        for step in all_steps:
            step_data = load_step_data_if_exists(results_dir, seed_run, step, unique_settings)
            if step_data is not None:
                seed_data[step] = step_data
                print(f"      Step {step}: Loaded {len(step_data)} comparisons")
            else:
                print(f"      Step {step}: Missing files, skipping")
        
        all_seed_data[seed_run] = seed_data
    
    # Aggregate across seeds
    print("\nAggregating data across seeds...")
    aggregated_data = aggregate_across_seeds(all_seed_data)
    
    # Print summary of aggregated data
    print("\nAggregation summary:")
    for step in sorted(aggregated_data.keys()):
        n_comparisons = len(aggregated_data[step])
        if n_comparisons > 0:
            n_seeds_min = min(data['n_seeds'] for data in aggregated_data[step].values())
            n_seeds_max = max(data['n_seeds'] for data in aggregated_data[step].values())
            print(f"  Step {step}: {n_comparisons} comparisons, {n_seeds_min}-{n_seeds_max} seeds per comparison")
    
    # Create output directory
    output_dir = os.path.join(results_dir, f"cross_seed_analysis_{partial_run_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    print("\nCreating plots...")
    plot_cross_seed_trajectories(aggregated_data, output_dir, partial_run_name)
    plot_seed_coverage(aggregated_data, output_dir, partial_run_name)
    
    print(f"\n{'='*70}")
    print("Cross-seed aggregation complete!")
    print(f"Plots saved to: {output_dir}")
    print(f"  - cross_seed_trajectories_{partial_run_name}.png")
    print(f"  - seed_coverage_{partial_run_name}.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()