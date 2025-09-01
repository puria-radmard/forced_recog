import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from utils.util import YamlConfig

# Import Bradley-Terry functions from the original script
from analyse_choices import run_bradley_terry_mcmc


def generate_synthetic_bt_data(theta_true: float, alpha_true: float, 
                              theta_std: float, alpha_std: float,
                              num_data: int) -> pd.DataFrame:
    """
    Generate synthetic Bradley-Terry data with parameter noise.
    
    Args:
        theta_true: True setting preference parameter
        alpha_true: True position bias parameter  
        theta_std: Standard deviation of noise added to theta
        alpha_std: Standard deviation of noise added to alpha
        num_data: Number of comparison pairs to generate
        
    Returns:
        DataFrame with columns [document_idx, o, y] matching BT format
    """
    synthetic_rows = []
    
    for i in range(num_data):
        
        # Generate both orders (o=+1, o=-1) with same noise
        for o in [+1, -1]:

            # Generate shared noise for this comparison pair
            theta_noise = np.random.normal(0, theta_std)
            alpha_noise = np.random.normal(0, alpha_std)
            
            # Noisy parameters
            theta_noisy = theta_true + theta_noise
            alpha_noisy = alpha_true + alpha_noise

            # Calculate probability using Bradley-Terry model
            logit = theta_noisy + 2 * o * alpha_noisy
            p = 1 / (1 + np.exp(-logit))  # sigmoid
            
            synthetic_rows.append({
                'document_idx': f"doc_{i}_{o}",  # Unique identifier
                'o': o,
                'y': p
            })
    
    return pd.DataFrame(synthetic_rows)


def plot_recovery_analysis(csv_file: str, output_dir: str, experiment_count: int) -> None:
    """
    Generate parameter recovery plots from current validation results.
    
    Args:
        csv_file: Path to CSV file with validation results
        output_dir: Directory to save plots
        experiment_count: Number of experiments completed (for filename)
    """
    
    # Read current results
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        print("No data available for plotting yet")
        return
        
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            print("No data rows available for plotting yet")
            return
    except Exception as e:
        print(f"Error reading CSV for plotting: {e}")
        return
    
    print(f"Generating recovery plots with {len(df)} data points...")
    
    # Set up plot style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Bradley-Terry Parameter Recovery Analysis\n'
                 f'({experiment_count} experiments completed)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: P(θ > 0) vs true θ, colored by true α
    scatter1 = axes[0, 0].scatter(df['theta_true_mean'], df['theta_p_gt_0'], 
                                 c=df['alpha_true_mean'], alpha=0.7, s=20, cmap='viridis')
    axes[0, 0].plot([-3, 3], [0.5, 0.5], 'r--', alpha=0.5, label='P = 0.5')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5, label='θ = 0')
    axes[0, 0].set_xlabel('True θ (Setting Preference)')
    axes[0, 0].set_ylabel('P(θ > 0) from Posterior')
    axes[0, 0].set_title('Theta Direction Recovery')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('True α')
    
    # Plot 2: P(α > 0) vs true α, colored by true θ  
    scatter2 = axes[0, 1].scatter(df['alpha_true_mean'], df['alpha_p_gt_0'],
                                 c=df['theta_true_mean'], alpha=0.7, s=20, cmap='viridis')
    axes[0, 1].plot([-2, 2], [0.5, 0.5], 'r--', alpha=0.5, label='P = 0.5')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.5, label='α = 0')
    axes[0, 1].set_xlabel('True α (Position Bias)')
    axes[0, 1].set_ylabel('P(α > 0) from Posterior')
    axes[0, 1].set_title('Alpha Direction Recovery')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('True θ')
    
    # Plot 3: E[θ] vs true θ, colored by true α
    scatter3 = axes[1, 0].scatter(df['theta_true_mean'], df['theta_est_mean'],
                                 c=df['alpha_true_mean'], alpha=0.7, s=20, cmap='viridis')
    # Perfect recovery line
    theta_range = [df['theta_true_mean'].min(), df['theta_true_mean'].max()]
    axes[1, 0].plot(theta_range, theta_range, 'r-', alpha=0.8, label='Perfect Recovery')
    axes[1, 0].set_xlabel('True θ (Setting Preference)')
    axes[1, 0].set_ylabel('E[θ] from Posterior')
    axes[1, 0].set_title('Theta Point Estimate Recovery')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
    cbar3.set_label('True α')
    
    # Plot 4: E[α] vs true α, colored by true θ
    scatter4 = axes[1, 1].scatter(df['alpha_true_mean'], df['alpha_est_mean'],
                                 c=df['theta_true_mean'], alpha=0.7, s=20, cmap='viridis')
    # Perfect recovery line
    alpha_range = [df['alpha_true_mean'].min(), df['alpha_true_mean'].max()]
    axes[1, 1].plot(alpha_range, alpha_range, 'r-', alpha=0.8, label='Perfect Recovery')
    axes[1, 1].set_xlabel('True α (Position Bias)')
    axes[1, 1].set_ylabel('E[α] from Posterior')
    axes[1, 1].set_title('Alpha Point Estimate Recovery')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 1])
    cbar4.set_label('True θ')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'recovery_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved recovery plot: {plot_file}")


def validate_bradley_terry_recovery(config_path: str) -> None:
    """
    Run parameter recovery validation for Bradley-Terry model.
    
    Args:
        config_path: Path to YAML configuration file
    """
    
    # Load configuration
    args = YamlConfig(config_path)
    
    # Extract parameters
    theta_min = args.theta_min
    theta_max = args.theta_max
    theta_num = args.theta_num
    theta_std = args.theta_std
    
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    alpha_num = args.alpha_num
    alpha_std = args.alpha_std
    
    num_data = args.num_data
    num_repeats = args.num_repeats
    
    # MCMC parameters (use defaults if not specified)
    theta_prior_sigma = getattr(args, 'theta_prior_sigma', 1.0)
    alpha_prior_sigma = getattr(args, 'alpha_prior_sigma', 0.5)
    n_samples = getattr(args, 'n_samples', 2000)
    n_warmup = getattr(args, 'n_warmup', 1000)
    
    print(f"Bradley-Terry Parameter Recovery Validation")
    print(f"==========================================")
    print(f"Configuration: {args.args_name}")
    print(f"Theta grid: [{theta_min}, {theta_max}] x {theta_num} points, noise σ={theta_std}")
    print(f"Alpha grid: [{alpha_min}, {alpha_max}] x {alpha_num} points, noise σ={alpha_std}")
    print(f"Data per gridpoint: {num_data} pairs ({2*num_data} observations)")
    print(f"Repeats per gridpoint: {num_repeats}")
    print(f"MCMC: {n_samples} samples, {n_warmup} warmup")
    print(f"Priors: θ~N(0,{theta_prior_sigma}), α~N(0,{alpha_prior_sigma})")
    
    # Create parameter grid
    theta_values = np.linspace(theta_min, theta_max, theta_num)
    alpha_values = np.linspace(alpha_min, alpha_max, alpha_num)
    grid_points = list(product(theta_values, alpha_values))
    
    print(f"Total grid points: {len(grid_points)}")
    print(f"Total experiments: {len(grid_points) * num_repeats}")
    
    # Setup output directory and file
    output_dir = f"results_and_data/results/bt_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{args.args_name}.csv")
    
    # Initialize CSV file with headers
    headers = [
        'theta_true_mean', 'theta_est_mean', 'theta_est_std', 'theta_p_gt_0',
        'alpha_true_mean', 'alpha_est_mean', 'alpha_est_std', 'alpha_p_gt_0',
        'repeat_idx'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    print(f"Initialized output file: {output_file}")
    
    # Run validation experiments
    total_experiments = len(grid_points) * num_repeats
    experiment_count = 0
    
    # Outer loop: repeats
    for repeat_idx in range(num_repeats):
        print(f"\n--- Repeat {repeat_idx + 1}/{num_repeats} ---")
        
        # Randomize order of grid points for this repeat
        shuffled_grid = grid_points.copy()
        np.random.shuffle(shuffled_grid)
        
        # Inner loop: grid points
        for grid_idx, (theta_true, alpha_true) in enumerate(shuffled_grid):
            experiment_count += 1
            
            print(f"Experiment {experiment_count}/{total_experiments}: "
                  f"θ={theta_true:.3f}, α={alpha_true:.3f}")
            
            try:
                # Generate synthetic data
                synthetic_data = generate_synthetic_bt_data(
                    theta_true, alpha_true, theta_std, alpha_std, num_data
                )
                
                # Run Bradley-Terry MCMC
                mcmc_results = run_bradley_terry_mcmc(
                    synthetic_data, 
                    theta_prior_sigma=theta_prior_sigma,
                    alpha_prior_sigma=alpha_prior_sigma,
                    n_samples=n_samples,
                    n_warmup=n_warmup
                )
                
                # Extract summary statistics
                theta_samples = mcmc_results['theta_samples']
                alpha_samples = mcmc_results['alpha_samples']
                
                result_row = {
                    'theta_true_mean': theta_true,
                    'theta_est_mean': theta_samples.mean(),
                    'theta_est_std': theta_samples.std(),
                    'theta_p_gt_0': (theta_samples > 0).mean(),
                    'alpha_true_mean': alpha_true,
                    'alpha_est_mean': alpha_samples.mean(),
                    'alpha_est_std': alpha_samples.std(),
                    'alpha_p_gt_0': (alpha_samples > 0).mean(),
                    'repeat_idx': repeat_idx
                }
                
                # Append to CSV file
                with open(output_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(result_row)
                
                print(f"  θ: {theta_true:.3f} → {result_row['theta_est_mean']:.3f} ± {result_row['theta_est_std']:.3f}")
                print(f"  α: {alpha_true:.3f} → {result_row['alpha_est_mean']:.3f} ± {result_row['alpha_est_std']:.3f}")
                
                # Generate recovery plots every 10 experiments
                if experiment_count % 10 == 0:
                    print(f"  Generating recovery plots at {experiment_count} experiments...")
                    plot_recovery_analysis(output_file, output_dir, experiment_count)
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                print("  Skipping this experiment...")
                continue
    
    print(f"\n{'='*60}")
    print(f"Parameter Recovery Validation Complete!")
    print(f"Results saved to: {output_file}")
    print(f"Successfully completed {experiment_count} experiments")
    
    # Generate final recovery plots
    if experiment_count > 0:
        print("Generating final recovery plots...")
        plot_recovery_analysis(output_file, output_dir, experiment_count)
    
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bt_validation.py /path/to/config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    validate_bradley_terry_recovery(config_path)