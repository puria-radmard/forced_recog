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


def process_to_choice_format(results_df: pd.DataFrame, 
                            positive_temp: float, positive_style: str,
                            negative_temp: float, negative_style: str) -> pd.DataFrame:
    """
    Convert raw choice results to choice model format for specific (temp, style) comparison.
    
    For each comparison, we want:
    - y: probability of choosing the positive setting (positive_temp, positive_style)
    - o: +1 if positive setting first, -1 if positive setting second
    
    Args:
        results_df: Raw results with [document_idx, summary1_temp, summary1_trial, summary1_style,
                   summary2_temp, summary2_trial, summary2_style, prob_choice_1, prob_choice_2]
        positive_temp: Temperature of the "positive" setting in choice model
        positive_style: Style of the "positive" setting in choice model
        negative_temp: Temperature of the "negative" setting in choice model  
        negative_style: Style of the "negative" setting in choice model
    
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
            raise Exception(f"Warning: Zero total probability for row {idx}, skipping")
            
        prob_choice_1_norm = row['prob_choice_1'] / total_prob
        prob_choice_2_norm = row['prob_choice_2'] / total_prob
        
        # Convert to choice model format
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


def run_choice_model_mcmc(choice_data: pd.DataFrame, 
                         theta_prior_sigma: float = 1.0,
                         alpha_prior_sigma: float = 0.5,
                         kappa_prior_alpha: float = 2.0,
                         kappa_prior_beta: float = 1.0,
                         n_samples: int = 2000,
                         n_warmup: int = 1000) -> dict:
    """
    Run MCMC inference on choice model with Beta likelihood and position bias.
    
    Model:
    θ ~ N(0, theta_prior_sigma)           # Setting preference  
    α ~ N(0, alpha_prior_sigma)           # Position bias
    κ ~ Gamma(kappa_prior_alpha, kappa_prior_beta)  # Concentration parameter
    p_n = sigmoid(θ + 2*o_n*α)            # Expected positive choice probability
    y_n ~ Beta(κ*p_n, κ*(1-p_n))         # Positive choice probability with Beta noise
    
    Args:
        choice_data: DataFrame with [document_idx, o, y]
        theta_prior_sigma: Prior std for setting preference
        alpha_prior_sigma: Prior std for position bias  
        kappa_prior_alpha: Prior shape parameter for concentration
        kappa_prior_beta: Prior rate parameter for concentration
        n_samples: Number of posterior samples
        n_warmup: Number of warmup samples
        
    Returns:
        Dictionary with posterior samples, diagnostics, and aggregated statistics for plotting
    """
    
    print(f"\nChoice Model MCMC Setup:")
    print(f"  Data: {len(choice_data)} comparisons")
    print(f"  Order counts: {choice_data['o'].value_counts().to_dict()}")
    print(f"  Mean y (prob choose positive): {choice_data['y'].mean():.4f}")
    print(f"  Priors: θ~N(0,{theta_prior_sigma}), α~N(0,{alpha_prior_sigma}), κ~Gamma({kappa_prior_alpha},{kappa_prior_beta})")
    print(f"  Sampling: {n_samples} samples, {n_warmup} warmup")

    y_obs = choice_data['y'].values
    o_obs = choice_data['o'].values
    
    with pm.Model() as model:
        # Priors
        theta = pm.Normal('theta', mu=0.0, sigma=theta_prior_sigma)
        alpha = pm.Normal('alpha', mu=0.0, sigma=alpha_prior_sigma)
        kappa = pm.Gamma('kappa', alpha=kappa_prior_alpha, beta=kappa_prior_beta)
        
        # Linear predictor: θ + 2*o*α
        logit_p = theta + 2 * o_obs * alpha
        predicted_p = pm.math.sigmoid(logit_p)
        
        # Beta parameters: α = κ*p, β = κ*(1-p)
        alpha_beta = kappa * predicted_p
        beta_beta = kappa * (1 - predicted_p)
        
        # Beta likelihood
        pm.Beta('y', alpha=alpha_beta, beta=beta_beta, observed=y_obs)
        
        # Sample
        print("\nRunning MCMC with NUTS sampler...")
        trace = pm.sample(draws=n_samples, tune=n_warmup, cores=1, 
                        return_inferencedata=True, progressbar=True)
    
    # Extract results
    theta_samples = trace.posterior['theta'].values.flatten()
    alpha_samples = trace.posterior['alpha'].values.flatten()
    kappa_samples = trace.posterior['kappa'].values.flatten()
    summary = az.summary(trace)
    
    # Calculate aggregated statistics for plotting (NEW)
    theta_mean = np.mean(theta_samples)
    theta_std = np.std(theta_samples)
    prob_theta_positive = (theta_samples > 0).mean()
    
    alpha_mean = np.mean(alpha_samples)
    alpha_std = np.std(alpha_samples)
    prob_alpha_positive = (alpha_samples > 0).mean()
    
    # Diagnostics
    r_hat_theta = float(summary.loc['theta', 'r_hat'])
    r_hat_alpha = float(summary.loc['alpha', 'r_hat'])
    r_hat_kappa = float(summary.loc['kappa', 'r_hat'])
    converged = r_hat_theta < 1.01 and r_hat_alpha < 1.01 and r_hat_kappa < 1.01
    
    # Interpretations
    prob_prefer_positive = (theta_samples > 0).mean()
    prob_first_bias = (alpha_samples > 0).mean()
    
    print(f"\nMCMC Results:")
    print(f"  θ (setting preference): {theta_mean:.4f} ± {theta_std:.4f}")
    print(f"  α (position bias): {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"  κ (concentration): {kappa_samples.mean():.4f} ± {kappa_samples.std():.4f}")
    print(f"  R-hat: θ={r_hat_theta:.3f}, α={r_hat_alpha:.3f}, κ={r_hat_kappa:.3f}")
    print(f"  Converged: {converged}")
    print(f"\nInterpretations:")
    print(f"  P(prefers positive): {prob_prefer_positive:.3f}")
    print(f"  P(first position bias): {prob_first_bias:.3f}")
    print(f"  Mean concentration: {kappa_samples.mean():.3f} (higher = less probability variability)")
    
    return {
        'theta_samples': theta_samples,
        'alpha_samples': alpha_samples,
        'kappa_samples': kappa_samples,
        'trace': trace,
        'summary': summary,
        # Aggregated statistics for plotting (NEW)
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'prob_theta_positive': prob_theta_positive,
        'alpha_mean': alpha_mean,
        'alpha_std': alpha_std,
        'prob_alpha_positive': prob_alpha_positive,
        'diagnostics': {
            'r_hat_theta': r_hat_theta,
            'r_hat_alpha': r_hat_alpha,
            'r_hat_kappa': r_hat_kappa,
            'converged': converged
        },
        'interpretations': {
            'prob_prefer_positive': prob_prefer_positive,
            'prob_first_bias': prob_first_bias
        },
        'model_spec': {
            'theta_prior_sigma': theta_prior_sigma,
            'alpha_prior_sigma': alpha_prior_sigma,
            'kappa_prior_alpha': kappa_prior_alpha,
            'kappa_prior_beta': kappa_prior_beta,
            'n_samples': n_samples,
            'n_warmup': n_warmup
        }
    }
        

def plot_choice_model_posteriors(mcmc_results: dict, output_dir: str, comparison_name: str, pos_name: str, neg_name: str) -> None:
    """
    Create visualizations of choice model posterior distributions.
    
    Args:
        mcmc_results: Results from run_choice_model_mcmc
        output_dir: Directory to save plots
        comparison_name: Name of the comparison for plot titles
    """
    
    if mcmc_results.get('placeholder', False):
        print("Skipping visualization for placeholder results")
        return
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Choice Model Posterior Analysis\npositive: {pos_name}\nnegative: {neg_name}', 
                 fontsize=16, fontweight='bold')
    
    theta_samples = mcmc_results['theta_samples']
    alpha_samples = mcmc_results['alpha_samples']
    kappa_samples = mcmc_results['kappa_samples']
    
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
    
    # Kappa (concentration) histogram
    axes[0, 2].hist(kappa_samples, bins=50, alpha=0.7, color='darkorange', density=True)
    axes[0, 2].axvline(kappa_samples.mean(), color='red', linestyle='-', alpha=0.8,
                      label=f'Mean = {kappa_samples.mean():.3f}')
    axes[0, 2].set_xlabel('κ (Concentration)')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Concentration Parameter\nHigher κ = Lower Recongition Probability Variability')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Joint scatter plot (theta vs alpha)
    axes[1, 0].scatter(theta_samples[::10], alpha_samples[::10], alpha=0.3, s=10, color='purple')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('θ (Setting Preference)')
    axes[1, 0].set_ylabel('α (Position Bias)')
    axes[1, 0].set_title('Joint θ-α Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Joint scatter plot (theta vs kappa)
    axes[1, 1].scatter(theta_samples[::10], kappa_samples[::10], alpha=0.3, s=10, color='brown')
    axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('θ (Setting Preference)')
    axes[1, 1].set_ylabel('κ (Concentration)')
    axes[1, 1].set_title('Joint θ-κ Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics and interpretations
    axes[1, 2].axis('off')
    
    # Calculate key statistics
    prob_prefer_positive = mcmc_results['interpretations']['prob_prefer_positive']
    prob_first_bias = mcmc_results['interpretations']['prob_first_bias']
    
    theta_ci = np.percentile(theta_samples, [2.5, 97.5])
    alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])
    kappa_ci = np.percentile(kappa_samples, [2.5, 97.5])
    
    r_hat_theta = mcmc_results['diagnostics']['r_hat_theta']
    r_hat_alpha = mcmc_results['diagnostics']['r_hat_alpha']
    r_hat_kappa = mcmc_results['diagnostics']['r_hat_kappa']
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
    
    Concentration (κ):
    Mean: {kappa_samples.mean():.4f}
    Std:  {kappa_samples.std():.4f}
    95% CI: [{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]
    
    Interpretations:
    ──────────────────────
    P(prefers positive): {prob_prefer_positive:.3f}
    P(1st pos. bias): {prob_first_bias:.3f}
    
    Diagnostics:
    ──────────────────────
    R̂ (θ): {r_hat_theta:.3f}
    R̂ (α): {r_hat_alpha:.3f}
    R̂ (κ): {r_hat_kappa:.3f}
    Converged: {converged}
    
    Model: 
    p = σ(θ + 2oα)
    y ~ Beta(κp, κ(1-p))
    """
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'choice_model_posteriors_{comparison_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved posterior visualization: {plot_file}")


def format_setting_name(temp: float, style: str) -> str:
    """Format a (temperature, style) setting into a filename-safe string."""
    style_str = "none" if style is None else str(style)
    return f"temp{temp}_{style_str}"


def analyze_choice_model(run_name: str, split_name: str,
                        positive_temp: float, positive_style: str,
                        negative_temp: float, negative_style: str,
                        theta_prior_sigma: float, alpha_prior_sigma: float,
                        kappa_prior_alpha: float, kappa_prior_beta: float,
                        n_samples: int, n_warmup: int,
                        use_lora: bool = False,
                        lora_run_name: str = None,
                        artifact_name: str = None) -> tuple:
    """
    Complete choice model analysis for a specific (temp, style) comparison.
    
    Args:
        run_name: Name of the run
        split_name: Name of the split
        positive_temp: Temperature of positive setting
        positive_style: Style of positive setting
        negative_temp: Temperature of negative setting
        negative_style: Style of negative setting
        theta_prior_sigma: Prior std for setting preference
        alpha_prior_sigma: Prior std for position bias
        kappa_prior_alpha: Prior alpha for concentration parameter
        kappa_prior_beta: Prior beta for concentration parameter
        n_samples: Number of MCMC samples
        n_warmup: Number of MCMC warmup samples
        use_lora: Whether using LoRA results
        lora_run_name: WandB run name for LoRA
        artifact_name: Artifact name for LoRA
        
    Returns:
        Tuple of (choice_data, mcmc_results)
    """
    
    # Create comparison name for file naming
    pos_name = format_setting_name(positive_temp, positive_style)
    neg_name = format_setting_name(negative_temp, negative_style)
    comparison_name = f"{pos_name}_vs_{neg_name}"
    
    print(f"\n{'='*70}")
    print(f"Choice Model Analysis: {split_name}")
    print(f"Comparison: {comparison_name}")
    if use_lora:
        print(f"Using LoRA: {lora_run_name}/{artifact_name}")
    else:
        print("Using base model results")
    print(f"{'='*70}")
    
    # Setup directories based on whether using LoRA
    results_dir = f"results_and_data/modal_results/results/main/{run_name}/{split_name}"
    
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
    
    choice_data = process_to_choice_format(
        raw_results, positive_temp, positive_style, negative_temp, negative_style
    )
    print(f"Processed to {len(choice_data)} choice model format comparisons")
    
    if len(choice_data) == 0:
        print(f"WARNING: No comparisons found for {comparison_name}")
        return choice_data, None
    
    # Save processed data
    choice_file = os.path.join(output_dir, f"choice_model_data_{comparison_name}.csv")
    choice_data.to_csv(choice_file, index=False)
    print(f"Saved choice model data: {choice_file}")
    
    # Run MCMC
    mcmc_results = run_choice_model_mcmc(
        choice_data, theta_prior_sigma, alpha_prior_sigma, 
        kappa_prior_alpha, kappa_prior_beta, n_samples, n_warmup
    )
    
    # Create visualizations
    plot_choice_model_posteriors(mcmc_results, output_dir, comparison_name, pos_name, neg_name)
    
    # Save results
    results_file = os.path.join(output_dir, f"choice_model_mcmc_{comparison_name}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(mcmc_results, f)
    print(f"Saved MCMC results: {results_file}")
    
    return choice_data, mcmc_results


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


def main(config_path: str, use_lora: bool = False, lora_run_name: str = None, artifact_name: str = None):
    """
    Main function for choice model analysis - can be called from other scripts.
    
    Args:
        config_path: Path to configuration YAML file
        use_lora: Whether to use LoRA results
        lora_run_name: WandB run name for LoRA (if use_lora=True)
        artifact_name: Artifact name for LoRA (if use_lora=True)
    """
    
    if use_lora and (lora_run_name is None or artifact_name is None):
        raise ValueError("lora_run_name and artifact_name must be provided when use_lora=True")
    
    if use_lora:
        print(f"Running choice model analysis on LoRA results:")
        print(f"  WandB Run: {lora_run_name}")
        print(f"  Artifact: {artifact_name}")
    else:
        print("Running choice model analysis on base model results")
    
    args = YamlConfig(config_path)
    
    # Extract MCMC parameters with defaults
    theta_prior_sigma = getattr(args, 'theta_prior_sigma', 1.0)
    alpha_prior_sigma = getattr(args, 'alpha_prior_sigma', 1.0)
    kappa_prior_alpha = getattr(args, 'kappa_prior_alpha', 2.0) 
    kappa_prior_beta = getattr(args, 'kappa_prior_beta', 1.0)
    n_samples = getattr(args, 'n_samples', 2000)
    n_warmup = getattr(args, 'n_warmup', 1000)
    
    print(f"Choice Model Analysis Configuration:")
    print(f"  Run: {args.args_name}")
    print(f"  Priors: θ~N(0,{theta_prior_sigma}), α~N(0,{alpha_prior_sigma}), κ~Gamma({kappa_prior_alpha},{kappa_prior_beta})")
    print(f"  MCMC: {n_samples} samples, {n_warmup} warmup")

    # Determine path based on LoRA usage
    split_name = "test"
    results_dir = f"results_and_data/modal_results/results/main/{args.args_name}/{split_name}"
    
    if use_lora:
        results_file = os.path.join(results_dir, f"forward_sft_choices/{lora_run_name}/{artifact_name}/choice_results.csv")
    else:
        results_file = os.path.join(results_dir, "initial_choices/choice_results.csv")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"ERROR: Results file not found: {results_file}")
        
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
            
            choice_data, mcmc_results = analyze_choice_model(
                args.args_name, split_name, 
                temp1, style1, temp2, style2,
                theta_prior_sigma, alpha_prior_sigma, 
                kappa_prior_alpha, kappa_prior_beta,
                n_samples, n_warmup,
                use_lora, lora_run_name, artifact_name
            )
            
            comparison_key = (setting1, setting2)
            split_results[comparison_key] = (choice_data, mcmc_results)
    
    print(f"\n{'='*70}")
    print("Choice Model Analysis Complete!")
    print(f"{'='*70}")
    
    return split_results


if __name__ == "__main__":
    
    if len(sys.argv) not in [2, 4]:
        print("Usage:")
        print("  Base model: python -m scripts.choice_analysis.analyse_choices /path/to/config.yaml")
        print("  With LoRA:  python -m scripts.choice_analysis.analyse_choices /path/to/config.yaml <wandb_run_name> <artifact_name>")
        sys.exit(1)
    
    # Load configuration
    config_path = sys.argv[1]
    use_lora = len(sys.argv) == 4
    
    if use_lora:
        lora_run_name = sys.argv[2]
        artifact_name = sys.argv[3]
        main(config_path, use_lora=True, lora_run_name=lora_run_name, artifact_name=artifact_name)
    else:
        main(config_path, use_lora=False)