import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import YamlConfig
from itertools import combinations
import re

# Import from the main choice model analysis script
from scripts.choice_analysis.analyse_choices import main as run_choice_analysis, get_unique_settings, format_setting_name


def check_data_completeness(choice_data_df: pd.DataFrame) -> tuple:
    """
    Check if the choice data has complete pairs (both o=1 and o=-1 for each document_idx).
    
    Returns:
        tuple: (num_datapoints, is_complete)
    """
    if len(choice_data_df) == 0:
        return 0, True
    
    # Group by document_idx and check if we have both o=1 and o=-1
    grouped = choice_data_df.groupby('document_idx')['o'].apply(set)
    
    # Count unique document indices
    num_datapoints = len(grouped)
    
    # Check if all documents have both directions
    is_complete = all({1, -1}.issubset(directions) for directions in grouped)
    
    return num_datapoints, is_complete


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


def check_and_load_step_data(results_dir: str, wandb_run_name: str, step: int, unique_settings: list, 
                            config_path: str) -> dict:
    """
    Check if all required pickle files exist for a step. If not, run the analysis.
    Load and return data for a specific training step.
    
    Returns:
        dict: {comparison_key: {'theta_mean': ..., 'alpha_mean': ..., 'data_count': ..., 'data_complete': ...}}
    """
    
    if step == 0:
        # Initial choices
        step_dir = os.path.join(results_dir, "initial_choices")
        use_lora = False
        lora_run_name = None
        artifact_name = None
    else:
        # LoRA step
        step_dir = os.path.join(results_dir, f"forward_sft_choices/{wandb_run_name}/lora_adapters_step_{step}")
        use_lora = True
        lora_run_name = wandb_run_name
        artifact_name = f"lora_adapters_step_{step}"
    
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
    
    # If any files are missing, run the analysis
    if missing_files:
        print(f"\nStep {step}: Found {len(missing_files)} missing files, running analysis...")
        if step == 0:
            print("  Running base model analysis...")
        else:
            print(f"  Running LoRA analysis for {artifact_name}...")
        
        # Run the analysis to generate missing files
        run_choice_analysis(config_path, use_lora=use_lora, 
                          lora_run_name=lora_run_name, artifact_name=artifact_name)
        print(f"  Analysis complete for step {step}")
    
    # Now load all the data
    step_data = {}
    for pickle_file, csv_file, comparison_key, pos_name, neg_name in expected_files:
        
        if not os.path.exists(pickle_file) or not os.path.exists(csv_file):
            raise FileNotFoundError(f"Expected file still missing after analysis: {pickle_file} or {csv_file}")
        
        # Load pickle file for MCMC results
        with open(pickle_file, 'rb') as f:
            mcmc_results = pickle.load(f)
        
        # Load CSV file for data completeness
        choice_data = pd.read_csv(csv_file)
        data_count, data_complete = check_data_completeness(choice_data)
        
        step_data[comparison_key] = {
            # Use pre-computed aggregated statistics from enriched pickle
            'theta_mean': mcmc_results['theta_mean'],
            'theta_std': mcmc_results['theta_std'],
            'prob_theta_positive': mcmc_results['prob_theta_positive'],
            'alpha_mean': mcmc_results['alpha_mean'],
            'alpha_std': mcmc_results['alpha_std'],
            'prob_alpha_positive': mcmc_results['prob_alpha_positive'],
            # Data completeness from CSV
            'data_count': data_count,
            'data_complete': data_complete,
            'pos_name': pos_name,
            'neg_name': neg_name
        }
    
    return step_data


def plot_trajectories(all_data: dict, output_dir: str):
    """
    Plot theta and alpha trajectories over training steps.
    """
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Choice Model Parameter Trajectories During Training', fontsize=16, fontweight='bold')
    
    # Get all comparison keys and steps
    comparison_keys = list(next(iter(all_data.values())).keys())
    steps = sorted(all_data.keys())
    
    # Generate colors for each comparison
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_keys)))
    
    # Find global y-axis ranges for consistent scaling
    all_theta_means = []
    all_theta_stds = []
    all_alpha_means = []
    all_alpha_stds = []
    
    for step in steps:
        for comp_key in comparison_keys:
            if comp_key in all_data[step]:
                theta_mean = all_data[step][comp_key]['theta_mean']
                theta_std = all_data[step][comp_key]['theta_std']
                alpha_mean = all_data[step][comp_key]['alpha_mean']
                alpha_std = all_data[step][comp_key]['alpha_std']
                
                all_theta_means.append(theta_mean)
                all_theta_stds.append(theta_std)
                all_alpha_means.append(alpha_mean)
                all_alpha_stds.append(alpha_std)
    
    # Calculate consistent y-axis ranges
    theta_min = min(np.array(all_theta_means) - np.array(all_theta_stds)) * 1.1
    theta_max = max(np.array(all_theta_means) + np.array(all_theta_stds)) * 1.1
    alpha_min = min(np.array(all_alpha_means) - np.array(all_alpha_stds)) * 1.1
    alpha_max = max(np.array(all_alpha_means) + np.array(all_alpha_stds)) * 1.1
    
    for i, comp_key in enumerate(comparison_keys):
        color = colors[i]
        pos_name = all_data[steps[0]][comp_key]['pos_name']
        neg_name = all_data[steps[0]][comp_key]['neg_name']
        
        label = f"Positive: {pos_name}\nNegative: {neg_name}"
        
        # Collect theta and alpha data (using pre-computed values)
        theta_means = []
        theta_stds = []
        alpha_means = []
        alpha_stds = []
        valid_steps = []
        
        for step in steps:
            if comp_key in all_data[step]:
                theta_means.append(all_data[step][comp_key]['theta_mean'])
                theta_stds.append(all_data[step][comp_key]['theta_std'])
                alpha_means.append(all_data[step][comp_key]['alpha_mean'])
                alpha_stds.append(all_data[step][comp_key]['alpha_std'])
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
    ax1.set_title('Setting Preference Evolution\nθ > 0: Prefers Positive Setting', fontsize=14)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No preference')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(theta_min, theta_max)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Format alpha plot
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('α (Position Bias)', fontsize=12)
    ax2.set_title('Position Bias Evolution\nα > 0: First Position Advantage', fontsize=14)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='No bias')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(alpha_min, alpha_max)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'pairwise_forward_sft_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved trajectory plot: {plot_file}")


def plot_diagnostics(all_data: dict, output_dir: str):
    """
    Plot data completeness diagnostics over training steps.
    """
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Data Completeness Diagnostics During Training', fontsize=16, fontweight='bold')
    
    # Get all comparison keys and steps
    comparison_keys = list(next(iter(all_data.values())).keys())
    steps = sorted(all_data.keys())
    
    # Generate colors for each comparison
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_keys)))
    
    # Find global y-axis range
    all_counts = []
    for step in steps:
        for comp_key in comparison_keys:
            if comp_key in all_data[step]:
                all_counts.append(all_data[step][comp_key]['data_count'])
    
    y_max = max(all_counts) * 1.1 if all_counts else 10
    
    for i, comp_key in enumerate(comparison_keys):
        color = colors[i]
        pos_name = all_data[steps[0]][comp_key]['pos_name']
        neg_name = all_data[steps[0]][comp_key]['neg_name']
        
        label = f"Positive: {pos_name}\nNegative: {neg_name}"
        
        # Collect data counts and completeness
        data_counts = []
        complete_steps = []
        incomplete_steps = []
        incomplete_counts = []
        valid_steps = []
        
        for step in steps:
            if comp_key in all_data[step]:
                data_count = all_data[step][comp_key]['data_count']
                data_complete = all_data[step][comp_key]['data_complete']
                
                data_counts.append(data_count)
                valid_steps.append(step)
                
                if not data_complete:
                    incomplete_steps.append(step)
                    incomplete_counts.append(data_count)
        
        # Plot line for data counts
        ax.plot(valid_steps, data_counts, color=color, marker='o', label=label, 
               linewidth=2, markersize=6)
        
        # Add red X marks for incomplete data
        if incomplete_steps:
            ax.scatter(incomplete_steps, incomplete_counts, color='red', marker='x', 
                      s=200, linewidth=4, zorder=10)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Number of Document Pairs', fontsize=12)
    ax.set_title('Data Completeness Over Training\nRed X = Incomplete Pairs (Missing Forward/Backward)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, y_max)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add custom legend entry for red X
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='x', color='red', linewidth=0, 
                          markersize=12, markeredgewidth=3)]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + custom_lines,
             labels=ax.get_legend_handles_labels()[1] + ['Incomplete Data'],
             bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'diagnostic_pairwise_forward_sft_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved diagnostic plot: {plot_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m summarise_choice_analysis /path/to/config.yaml <wandb_run_name>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    wandb_run_name = sys.argv[2]
    
    print(f"Summarizing choice analysis trajectories:")
    print(f"  Config: {config_path}")
    print(f"  WandB Run: {wandb_run_name}")
    
    # Load configuration
    args = YamlConfig(config_path)
    
    # Set up directories
    split_name = "test"
    results_dir = f"results_and_data/modal_results/results/main/{args.args_name}/{split_name}"
    wandb_base_dir = os.path.join(results_dir, f"forward_sft_choices/{wandb_run_name}")
    
    # Discover training steps
    training_steps = discover_training_steps(wandb_base_dir)
    all_steps = [0] + training_steps  # Include initial choices as step 0
    
    print(f"Found training steps: {training_steps}")
    print(f"All steps (including initial): {all_steps}")
    
    if not training_steps:
        raise ValueError("No training steps found! Make sure LoRA results exist.")
    
    # Load initial choices to discover unique settings
    initial_results_file = os.path.join(results_dir, "initial_choices/choice_results.csv")
    if not os.path.exists(initial_results_file):
        raise FileNotFoundError(f"Initial results file not found: {initial_results_file}")
    
    raw_results = pd.read_csv(initial_results_file)
    unique_settings = get_unique_settings(raw_results)
    
    print(f"Found {len(unique_settings)} unique settings:")
    for setting in unique_settings:
        temp, style = setting
        print(f"  Temperature: {temp}, Style: {style}")
    
    # Load data for all steps (with auto-analysis if missing)
    print("\nLoading/checking data for all steps...")
    all_data = {}
    for step in all_steps:
        print(f"  Checking step {step}...")
        step_data = check_and_load_step_data(results_dir, wandb_run_name, step, unique_settings, config_path)
        all_data[step] = step_data
        print(f"    Loaded {len(step_data)} comparisons")
    
    # Create plots
    print("\nCreating plots...")
    plot_trajectories(all_data, wandb_base_dir)
    plot_diagnostics(all_data, wandb_base_dir)
    
    print(f"\n{'='*50}")
    print("Summary complete!")
    print(f"Plots saved to: {wandb_base_dir}")
    print(f"  - pairwise_forward_sft_summary.png")
    print(f"  - diagnostic_pairwise_forward_sft_summary.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()