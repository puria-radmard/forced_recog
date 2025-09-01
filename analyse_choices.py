import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import YamlConfig
from itertools import combinations

import pymc as pm
import arviz as az
import pytensor.tensor as pt


def process_to_bradley_terry_format(results_df: pd.DataFrame, 
                                  positive_temp: float, positive_style: str,
                                  negative_temp: float, negative_style: str) -> pd.DataFrame:
    """
    Convert raw choice results to Bradley-Terry format for specific (temp, style) comparison.
    
    For each comparison, we want:
    - y: probability of choosing the positive setting (positive_temp, positive_style)
    - o: +1 if positive setting first, -1 if positive setting second
    
    Args:
        results_df: Raw results with [document_idx, summary1_temp, summary1_trial, summary1_style,
                   summary2_temp, summary2_trial, summary2_style, prob_choice_1, prob_choice_2]
        positive_temp: Temperature of the "positive" setting in Bradley-Terry model
        positive_style: Style of the "positive" setting in Bradley-Terry model
        negative_temp: Temperature of the "negative" setting in Bradley-Terry model  
        negative_style: Style of the "negative" setting in Bradley-Terry model
    
    Returns:
        DataFrame with [document_idx, o, y]
    """
    processed_rows = []
    
    for idx, row in results_df.iterrows():
        # Check if this row is a comparison between our target settings
        setting1 = (row['summary1_temp'], row['summary1_style'])
        setting2 = (row['summary2_temp'], row['summary2_style'])
        positive_setting = (positive_temp, positive_style)
        negative_setting = (negative_temp, negative_style)
        
        # Skip if this isn't a comparison between our target settings
        if not ((setting1 == positive_setting and setting2 == negative_setting) or
                (setting1 == negative_setting and setting2 == positive_setting)):
            continue
        
        # Normalize probabilities to sum to 1
        total_prob = row['prob_choice_1'] + row['prob_choice_2']
        if total_prob == 0:
            print(f"Warning: Zero total probability for row {idx}, skipping")
            continue
            
        prob_choice_1_norm = row['prob_choice_1'] / total_prob
        prob_choice_2_norm = row['prob_choice_2'] / total_prob
        
        # Convert to Bradley-Terry format
        if setting1 == positive_setting and setting2 == negative_setting:
            # Forward: positive setting first, negative setting second
            o = +1  # positive setting is in first position
            y = prob_choice_1_norm  # Probability of choosing positive setting
            
        elif setting1 == negative_setting and setting2 == positive_setting:
            # Backward: negative setting first, positive setting second  
            o = -1  # positive setting is in second position
            y = prob_choice_2_norm  # Probability of choosing positive setting
            
        else:
            # This shouldn't happen given our filtering above, but just in case
            raise Exception(f"Not expecting settings {setting1} and {setting2}")
        
        processed_rows.append({
            'document_idx': row['document_idx'],
            'o': o,
            'y': y
        })
    
    return pd.DataFrame(processed_rows)


def run_bradley_terry_mcmc(bt_data: pd.DataFrame, 
                          theta_prior_sigma: float = 1.0,
                          alpha_prior_sigma: float = 0.5,
                          n_samples: int = 2000,
                          n_warmup: int = 1000) -> dict:
    """
    Run MCMC inference on Bradley-Terry model with position bias.
    
    Model:
    θ ~ N(0, theta_prior_sigma)    # Setting preference  
    α ~ N(0, alpha_prior_sigma)    # Position bias
    y_n ~ Bernoulli(sigmoid(θ + 2*o_n*α))
    
    Args:
        bt_data: DataFrame with [document_idx, o, y]
        theta_prior_sigma: Prior std for setting preference
        alpha_prior_sigma: Prior std for position bias  
        n_samples: Number of posterior samples
        n_warmup: Number of warmup samples
        
    Returns:
        Dictionary with posterior samples and diagnostics
    """
    
    print(f"\nBradley-Terry MCMC Setup:")
    print(f"  Data: {len(bt_data)} comparisons")
    print(f"  Order counts: {bt_data['o'].value_counts().to_dict()}")
    print(f"  Mean y (prob choose positive): {bt_data['y'].mean():.4f}")
    print(f"  Priors: θ~N(0,{theta_prior_sigma}), α~N(0,{alpha_prior_sigma})")
    print(f"  Sampling: {n_samples} samples, {n_warmup} warmup")

    y_obs = bt_data['y'].values
    o_obs = bt_data['o'].values
    
    with pm.Model() as model:
        # Priors
        theta = pm.Normal('theta', mu=0.0, sigma=theta_prior_sigma)
        alpha = pm.Normal('alpha', mu=0.0, sigma=alpha_prior_sigma)
        
        # Linear predictor: θ + 2*o*α
        logit_p = theta + 2 * o_obs * alpha
        predicted_p = pm.math.sigmoid(logit_p)
        
        # Binary cross-entropy likelihood
        pm.Potential('likelihood', 
                 pt.sum(y_obs * pt.log(predicted_p) + 
                        (1 - y_obs) * pt.log(1 - predicted_p)))
        
        # Sample
        print("\nRunning MCMC with NUTS sampler...")
        trace = pm.sample(draws=n_samples, tune=n_warmup, cores=1, 
                        return_inferencedata=True, progressbar=True)
    
    # Extract results
    theta_samples = trace.posterior['theta'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()
    summary = az.summary(trace)
    
    # Diagnostics
    r_hat_theta = float(summary.loc['theta', 'r_hat'])
    r_hat_alpha = float(summary.loc['alpha', 'r_hat'])
    converged = r_hat_theta < 1.01 and r_hat_alpha < 1.01
    
    # Interpretations
    prob_prefer_positive = (theta_samples > 0).mean()
    prob_first_bias = (alpha_samples > 0).mean()
    
    print(f"\nMCMC Results:")
    print(f"  θ (setting preference): {theta_samples.mean():.4f} ± {theta_samples.std():.4f}")
    print(f"  α (position bias): {alpha_samples.mean():.4f} ± {alpha_samples.std():.4f}")
    print(f"  R-hat: θ={r_hat_theta:.3f}, α={r_hat_alpha:.3f}")
    print(f"  Converged: {converged}")
    print(f"\nInterpretations:")
    print(f"  P(prefers positive): {prob_prefer_positive:.3f}")
    print(f"  P(first position bias): {prob_first_bias:.3f}")
    
    return {
        'theta_samples': theta_samples,
        'alpha_samples': alpha_samples,
        'trace': trace,
        'summary': summary,
        'diagnostics': {
            'r_hat_theta': r_hat_theta,
            'r_hat_alpha': r_hat_alpha,
            'converged': converged
        },
        'interpretations': {
            'prob_prefer_positive': prob_prefer_positive,
            'prob_first_bias': prob_first_bias
        },
        'model_spec': {
            'theta_prior_sigma': theta_prior_sigma,
            'alpha_prior_sigma': alpha_prior_sigma,
            'n_samples': n_samples,
            'n_warmup': n_warmup
        }
    }
        

def plot_bradley_terry_posteriors(mcmc_results: dict, output_dir: str, comparison_name: str, pos_name: str, neg_name: str) -> None:
    """
    Create visualizations of Bradley-Terry posterior distributions.
    
    Args:
        mcmc_results: Results from run_bradley_terry_mcmc
        output_dir: Directory to save plots
        comparison_name: Name of the comparison for plot titles
    """
    
    if mcmc_results.get('placeholder', False):
        print("Skipping visualization for placeholder results")
        return
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Bradley-Terry Posterior Analysis\npositive: {pos_name}\nnegative: {neg_name}', 
                 fontsize=16, fontweight='bold')
    
    theta_samples = mcmc_results['theta_samples']
    alpha_samples = mcmc_results['alpha_samples']
    
    # Theta (setting preference) histogram
    axes[0, 0].hist(theta_samples, bins=50, alpha=0.7, color='steelblue', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.8, label='No preference')
    axes[0, 0].axvline(theta_samples.mean(), color='orange', linestyle='-', alpha=0.8, 
                      label=f'Mean = {theta_samples.mean():.3f}')
    axes[0, 0].set_xlabel('θ (Setting Preference)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Setting Preference Distribution\nθ > 0: Prefers Positive Setting')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha (position bias) histogram  
    axes[0, 1].hist(alpha_samples, bins=50, alpha=0.7, color='forestgreen', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8, label='No bias')
    axes[0, 1].axvline(alpha_samples.mean(), color='orange', linestyle='-', alpha=0.8,
                      label=f'Mean = {alpha_samples.mean():.3f}')
    axes[0, 1].set_xlabel('α (Position Bias)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Position Bias Distribution\nα > 0: First Position Advantage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Joint scatter plot
    axes[1, 0].scatter(theta_samples[::10], alpha_samples[::10], alpha=0.3, s=10, color='purple')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('θ (Setting Preference)')
    axes[1, 0].set_ylabel('α (Position Bias)')
    axes[1, 0].set_title('Joint Posterior Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics and interpretations
    axes[1, 1].axis('off')
    
    # Calculate key statistics
    prob_prefer_positive = mcmc_results['interpretations']['prob_prefer_positive']
    prob_first_bias = mcmc_results['interpretations']['prob_first_bias']
    
    theta_ci = np.percentile(theta_samples, [2.5, 97.5])
    alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])
    
    r_hat_theta = mcmc_results['diagnostics']['r_hat_theta']
    r_hat_alpha = mcmc_results['diagnostics']['r_hat_alpha']
    converged = mcmc_results['diagnostics']['converged']
    
    summary_text = f"""
    MCMC Summary
    ──────────────────────
    
    Setting Preference (θ):
    Mean: {theta_samples.mean():.4f}
    Std:  {theta_samples.std():.4f}
    95% CI: [{theta_ci[0]:.4f}, {theta_ci[1]:.4f}]
    
    Position Bias (α):
    Mean: {alpha_samples.mean():.4f}
    Std:  {alpha_samples.std():.4f}
    95% CI: [{alpha_ci[0]:.4f}, {alpha_ci[1]:.4f}]
    
    Interpretations:
    ──────────────────────
    P(prefers positive): {prob_prefer_positive:.3f}
    P(1st pos. bias): {prob_first_bias:.3f}
    
    Diagnostics:
    ──────────────────────
    R̂ (θ): {r_hat_theta:.3f}
    R̂ (α): {r_hat_alpha:.3f}
    Converged: {converged}
    
    Model: y ~ Bernoulli(σ(θ + 2oα))
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'bradley_terry_posteriors_{comparison_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved posterior visualization: {plot_file}")


def format_setting_name(temp: float, style: str) -> str:
    """Format a (temperature, style) setting into a filename-safe string."""
    style_str = "none" if style is None else str(style)
    return f"temp{temp}_{style_str}"


def analyze_bradley_terry(run_name: str, split_name: str,
                         positive_temp: float, positive_style: str,
                         negative_temp: float, negative_style: str,
                         theta_prior_sigma: float, alpha_prior_sigma: float,
                         n_samples: int, n_warmup: int,
                         use_lora: bool = False,
                         lora_run_name: str = None,
                         artifact_name: str = None) -> tuple:
    """
    Complete Bradley-Terry analysis for a specific (temp, style) comparison.
    
    Args:
        run_name: Name of the run
        split_name: Name of the split
        positive_temp: Temperature of positive setting
        positive_style: Style of positive setting
        negative_temp: Temperature of negative setting
        negative_style: Style of negative setting
        theta_prior_sigma: Prior std for setting preference
        alpha_prior_sigma: Prior std for position bias
        n_samples: Number of MCMC samples
        n_warmup: Number of MCMC warmup samples
        use_lora: Whether using LoRA results
        lora_run_name: WandB run name for LoRA
        artifact_name: Artifact name for LoRA
        
    Returns:
        Tuple of (bt_data, mcmc_results)
    """
    
    # Create comparison name for file naming
    pos_name = format_setting_name(positive_temp, positive_style)
    neg_name = format_setting_name(negative_temp, negative_style)
    comparison_name = f"{pos_name}_vs_{neg_name}"
    
    print(f"\n{'='*70}")
    print(f"Bradley-Terry Analysis: {split_name}")
    print(f"Comparison: {comparison_name}")
    if use_lora:
        print(f"Using LoRA: {lora_run_name}/{artifact_name}")
    else:
        print("Using base model results")
    print(f"{'='*70}")
    
    # Setup directories based on whether using LoRA
    results_dir = f"results_and_data/results/main/{run_name}/{split_name}"
    
    if use_lora:
        # Use LoRA-specific directory
        choice_dir = f"forward_sft_choices/{lora_run_name}/{artifact_name}"
        results_file = os.path.join(results_dir, choice_dir, "choice_results.csv")
        output_dir = os.path.join(results_dir, choice_dir)
    else:
        # Use base model directory
        choice_dir = "initial_choices"
        results_file = os.path.join(results_dir, choice_dir, "choice_results.csv")
        output_dir = os.path.join(results_dir, choice_dir)
    
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        return None, None
    
    # Load and process data
    print(f"Loading results from: {results_file}")
    raw_results = pd.read_csv(results_file)
    print(f"Loaded {len(raw_results)} raw comparisons")
    
    bt_data = process_to_bradley_terry_format(
        raw_results, positive_temp, positive_style, negative_temp, negative_style
    )
    print(f"Processed to {len(bt_data)} Bradley-Terry format comparisons")
    
    if len(bt_data) == 0:
        print(f"WARNING: No comparisons found for {comparison_name}")
        return bt_data, None
    
    # Save processed data
    bt_file = os.path.join(output_dir, f"bradley_terry_data_{comparison_name}.csv")
    bt_data.to_csv(bt_file, index=False)
    print(f"Saved Bradley-Terry data: {bt_file}")
    
    # Run MCMC
    mcmc_results = run_bradley_terry_mcmc(
        bt_data, theta_prior_sigma, alpha_prior_sigma, n_samples, n_warmup
    )
    
    # Create visualizations
    plot_bradley_terry_posteriors(mcmc_results, output_dir, comparison_name, pos_name, neg_name)
    
    # Save results
    results_file = os.path.join(output_dir, f"bradley_terry_mcmc_{comparison_name}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(mcmc_results, f)
    print(f"Saved MCMC results: {results_file}")
    
    return bt_data, mcmc_results


def get_unique_settings(results_df: pd.DataFrame) -> list:
    """
    Get all unique (temperature, style) combinations from the choice results.
    
    Args:
        results_df: DataFrame with choice results
        
    Returns:
        List of unique (temp, style) tuples
    """
    settings = set()
    
    # Get settings from both summary1 and summary2 columns
    for _, row in results_df.iterrows():
        settings.add((row['summary1_temp'], row['summary1_style']))
        settings.add((row['summary2_temp'], row['summary2_style']))
    
    return sorted(list(settings))


if __name__ == "__main__":
    
    if len(sys.argv) not in [2, 4]:
        print("Usage:")
        print("  Base model: python bradley_terry_analysis.py /path/to/config.yaml")
        print("  With LoRA:  python bradley_terry_analysis.py /path/to/config.yaml <wandb_run_name> <artifact_name>")
        sys.exit(1)
    
    # Load configuration
    config_path = sys.argv[1]
    use_lora = len(sys.argv) == 4
    
    if use_lora:
        lora_run_name = sys.argv[2]
        artifact_name = sys.argv[3]
        print(f"Running Bradley-Terry analysis on LoRA results:")
        print(f"  WandB Run: {lora_run_name}")
        print(f"  Artifact: {artifact_name}")
    else:
        lora_run_name = None
        artifact_name = None
        print("Running Bradley-Terry analysis on base model results")
    
    args = YamlConfig(config_path)
    
    # Extract MCMC parameters with defaults
    theta_prior_sigma = getattr(args, 'theta_prior_sigma', 1.0)
    alpha_prior_sigma = getattr(args, 'alpha_prior_sigma', 1.0) 
    n_samples = getattr(args, 'n_samples', 2000)
    n_warmup = getattr(args, 'n_warmup', 1000)
    
    print(f"Bradley-Terry Analysis Configuration:")
    print(f"  Run: {args.args_name}")
    print(f"  Splits: {args.splits}")
    print(f"  Priors: θ~N(0,{theta_prior_sigma}), α~N(0,{alpha_prior_sigma})")
    print(f"  MCMC: {n_samples} samples, {n_warmup} warmup")
    
    # Run analysis for each split
    all_results = {}
    for split_name in args.splits:
        
        # Determine path based on LoRA usage
        results_dir = f"results_and_data/results/main/{args.args_name}/{split_name}"
        
        if use_lora:
            results_file = os.path.join(results_dir, f"forward_sft_choices/{lora_run_name}/{artifact_name}/choice_results.csv")
        else:
            results_file = os.path.join(results_dir, "initial_choices/choice_results.csv")
        
        if not os.path.exists(results_file):
            print(f"ERROR: Results file not found: {results_file}")
            continue
            
        raw_results = pd.read_csv(results_file)
        unique_settings = get_unique_settings(raw_results)
        
        print(f"\nFound {len(unique_settings)} unique settings in {split_name}:")
        for setting in unique_settings:
            temp, style = setting
            print(f"  Temperature: {temp}, Style: {style}")
        
        # Run pairwise comparisons for all combinations
        split_results = {}
        for i, setting1 in enumerate(unique_settings):
            for j, setting2 in enumerate(unique_settings):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                    
                temp1, style1 = setting1
                temp2, style2 = setting2
                
                print(f"\nComparing {format_setting_name(temp1, style1)} vs {format_setting_name(temp2, style2)}")
                
                bt_data, mcmc_results = analyze_bradley_terry(
                    args.args_name, split_name, 
                    temp1, style1, temp2, style2,
                    theta_prior_sigma, alpha_prior_sigma, 
                    n_samples, n_warmup,
                    use_lora, lora_run_name, artifact_name
                )
                
                comparison_key = (setting1, setting2)
                split_results[comparison_key] = (bt_data, mcmc_results)
        
        all_results[split_name] = split_results
    
    print(f"\n{'='*70}")
    print("Bradley-Terry Analysis Complete!")
    print(f"{'='*70}")