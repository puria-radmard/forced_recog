import pandas as pd
import numpy as np
import os
import sys
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

import pymc as pm
import arviz as az
import pytensor.tensor as pt


def process_fj_to_choice_format(results_df: pd.DataFrame, 
                                        positive_run_name: str, positive_lora_name: str,
                                        negative_run_name: str, negative_lora_name: str) -> pd.DataFrame:
    """
    Convert false justification choice results to choice model format.
    
    For each comparison, we want:
    - y: probability of choosing the positive setting (positive_run_name, positive_lora_name)
    - o: +1 if positive setting first, -1 if positive setting second
    
    Args:
        results_df: Raw results with [document_idx, summary1_run_name, summary1_lora_name,
                   summary2_run_name, summary2_lora_name, prob_choice_1, prob_choice_2]
        positive_run_name: Run name of the "positive" setting
        positive_lora_name: LoRA name of the "positive" setting
        negative_run_name: Run name of the "negative" setting  
        negative_lora_name: LoRA name of the "negative" setting
    
    Returns:
        DataFrame with [document_idx, o, y]
    """
    processed_rows = []
    
    for idx, row in results_df.iterrows():
        # Check if this row is a comparison between our target settings
        setting1 = (row['summary1_run_name'], row['summary1_lora_name'])
        setting2 = (row['summary2_run_name'], row['summary2_lora_name'])
        positive_setting = (positive_run_name, positive_lora_name)
        negative_setting = (negative_run_name, negative_lora_name)
        
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
    
    # Calculate aggregated statistics for plotting
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


def plot_fj_choice_model_posteriors(mcmc_results: dict, output_dir: str, 
                                           pos_name: str, neg_name: str) -> None:
    """
    Create visualizations of false justification choice model posterior distributions.
    
    Args:
        mcmc_results: Results from run_choice_model_mcmc
        output_dir: Directory to save plots
        pos_name: Name of positive setting
        neg_name: Name of negative setting
    """
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Contrastive Choice Model Posterior Analysis\nPositive: {pos_name}\nNegative: {neg_name}', 
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
    axes[0, 2].set_title('Concentration Parameter\nHigher κ = Lower Recognition Probability Variability')
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
    plot_file = os.path.join(output_dir, 'fj_choice_model_posteriors.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved posterior visualization: {plot_file}")


def analyze_fj_choice_directory(directory_path: str,
                                       theta_prior_sigma: float = 1.0,
                                       alpha_prior_sigma: float = 0.5,
                                       kappa_prior_alpha: float = 2.0,
                                       kappa_prior_beta: float = 1.0,
                                       n_samples: int = 2000,
                                       n_warmup: int = 1000) -> tuple:
    """
    Analyze fj choice results from a directory containing both
    fj_choice_results.csv and choice_args.yaml.
    
    Args:
        directory_path: Path to directory containing files
        theta_prior_sigma: Prior std for setting preference
        alpha_prior_sigma: Prior std for position bias
        kappa_prior_alpha: Prior alpha for concentration parameter
        kappa_prior_beta: Prior beta for concentration parameter
        n_samples: Number of MCMC samples
        n_warmup: Number of MCMC warmup samples
        
    Returns:
        Tuple of (choice_data, mcmc_results)
    """
    
    print(f"\n{'='*70}")
    print(f"False Justification Choice Model Analysis")
    print(f"Directory: {directory_path}")
    print(f"{'='*70}")
    
    # Load lora args to determine positive/negative settings
    lora_args_file = os.path.join(directory_path, "choice_args.yaml")
    if not os.path.exists(lora_args_file):
        raise FileNotFoundError(f"ERROR: choice_args.yaml not found: {lora_args_file}")
    
    with open(lora_args_file, 'r') as f:
        lora_args = yaml.safe_load(f)
    
    # Validate required keys
    required_keys = ['positive_run_name', 'positive_artifact_name', 'negative_run_name', 'negative_artifact_name']
    for key in required_keys:
        if key not in lora_args:
            raise ValueError(f"Missing required key '{key}' in choice_args.yaml")
    
    positive_run_name = lora_args['positive_run_name']
    positive_artifact_name = lora_args['positive_artifact_name']
    negative_run_name = lora_args['negative_run_name']
    negative_artifact_name = lora_args['negative_artifact_name']
    
    print(f"Settings from choice_args.yaml:")
    print(f"  Positive: {positive_run_name}/{positive_artifact_name}")
    print(f"  Negative: {negative_run_name}/{negative_artifact_name}")
    
    # Load fj choice results
    results_file = os.path.join(directory_path, "fj_choice_results.csv")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"ERROR: Results file not found: {results_file}")
    
    print(f"Loading results from: {results_file}")
    raw_results = pd.read_csv(results_file)
    print(f"Loaded {len(raw_results)} raw comparisons")
    
    # Verify the data contains our expected settings
    unique_settings = set()
    for _, row in raw_results.iterrows():
        unique_settings.add((row['summary1_run_name'], row['summary1_lora_name']))
        unique_settings.add((row['summary2_run_name'], row['summary2_lora_name']))
    
    expected_positive = (positive_run_name, positive_artifact_name)
    expected_negative = (negative_run_name, negative_artifact_name)
    
    if expected_positive not in unique_settings:
        raise ValueError(f"Positive setting {expected_positive} not found in data. Available: {unique_settings}")
    if expected_negative not in unique_settings:
        raise ValueError(f"Negative setting {expected_negative} not found in data. Available: {unique_settings}")
    
    print(f"Verified that both settings are present in the data")
    
    # Process data
    choice_data = process_fj_to_choice_format(
        raw_results, positive_run_name, positive_artifact_name, 
        negative_run_name, negative_artifact_name
    )
    print(f"Processed to {len(choice_data)} choice model format comparisons")
    
    if len(choice_data) == 0:
        print(f"WARNING: No valid comparisons found")
        return choice_data, None
    
    # Save processed data
    choice_file = os.path.join(directory_path, "fj_choice_model_data.csv")
    choice_data.to_csv(choice_file, index=False)
    print(f"Saved choice model data: {choice_file}")
    
    # Run MCMC
    mcmc_results = run_choice_model_mcmc(
        choice_data, theta_prior_sigma, alpha_prior_sigma, 
        kappa_prior_alpha, kappa_prior_beta, n_samples, n_warmup
    )
    
    # Create visualizations
    pos_name = f"{positive_run_name}/{positive_artifact_name}"
    neg_name = f"{negative_run_name}/{negative_artifact_name}"
    plot_fj_choice_model_posteriors(mcmc_results, directory_path, pos_name, neg_name)
    
    # Save results
    results_file = os.path.join(directory_path, "choice_model_mcmc.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(mcmc_results, f)
    print(f"Saved MCMC results: {results_file}")
    
    return choice_data, mcmc_results


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python -m scripts_summary.choice_analysis.analyse_fj_choices /path/to/lora/directory")
        print("  (Directory must contain both fj_choice_results.csv and choice_args.yaml)")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    if not os.path.exists(directory_path):
        print(f"ERROR: Directory does not exist: {directory_path}")
        sys.exit(1)
    
    if not os.path.isdir(directory_path):
        print(f"ERROR: Path is not a directory: {directory_path}")
        sys.exit(1)
    
    print(f"Analyzing fj choices in directory: {directory_path}")
    
    try:
        choice_data, mcmc_results = analyze_fj_choice_directory(
            directory_path=directory_path,
            theta_prior_sigma=1.0,
            alpha_prior_sigma=0.5,
            kappa_prior_alpha=2.0,
            kappa_prior_beta=1.0,
            n_samples=2000,
            n_warmup=1000
        )
        
        print(f"\n{'='*70}")
        print("False Justification Choice Model Analysis Complete!")
        print(f"All results saved to: {directory_path}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        sys.exit(1)