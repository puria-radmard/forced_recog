import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from utils.util import YamlConfig


def compute_simple_averaging(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute position-bias-corrected temperature preferences using simple averaging.
    
    For each document, we average across forward/backward orders to get:
    - Position-bias-corrected P(choose T=0.0)  
    - Position bias estimate
    - Temperature preference delta
    
    Args:
        results_df: Raw results with [document_idx, order, prob_choice_1, prob_choice_2, ...]
        
    Returns:
        DataFrame with [document_idx, p_choose_t0_corrected, position_bias, temp_delta]
    """
    
    # Group by document to pair forward/backward comparisons
    doc_results = []
    
    for doc_idx in results_df['document_idx'].unique():
        doc_data = results_df[results_df['document_idx'] == doc_idx]
        
        # Should have exactly 2 rows: forward and backward
        if len(doc_data) != 2:
            print(f"Warning: Document {doc_idx} has {len(doc_data)} comparisons, expected 2")
            continue
            
        forward_row = doc_data[doc_data['order'] == 'forward']
        backward_row = doc_data[doc_data['order'] == 'backward']
        
        if len(forward_row) != 1 or len(backward_row) != 1:
            print(f"Warning: Document {doc_idx} missing forward or backward comparison")
            continue
            
        forward_row = forward_row.iloc[0]
        backward_row = backward_row.iloc[0]
        
        # Normalize probabilities
        def normalize_probs(row):
            total = row['prob_choice_1'] + row['prob_choice_2']
            if total == 0:
                return 0.5, 0.5  # Default to equal if both zero
            return row['prob_choice_1'] / total, row['prob_choice_2'] / total
        
        f_p1, f_p2 = normalize_probs(forward_row)
        b_p1, b_p2 = normalize_probs(backward_row)
        
        # Extract P(choose T=0.0) in each order
        # Forward: T=0.0 first, so P(choose T=0.0) = P(choose first) = f_p1
        # Backward: T=0.0 second, so P(choose T=0.0) = P(choose second) = b_p2
        p_choose_t0_forward = f_p1
        p_choose_t0_backward = b_p2
        
        # Position-bias-corrected estimate
        p_choose_t0_corrected = (p_choose_t0_forward + p_choose_t0_backward) / 2
        
        # Temperature preference delta: 2 * P(T=0.0) - 1
        temp_delta = 2 * p_choose_t0_corrected - 1
        
        # Position bias: average first-position advantage - 0.5
        p_choose_first_forward = f_p1  # Forward: first = T=0.0
        p_choose_first_backward = b_p1  # Backward: first = T=1.0
        position_bias = (p_choose_first_forward + p_choose_first_backward) / 2 - 0.5
        
        doc_results.append({
            'document_idx': doc_idx,
            'p_choose_t0_forward': p_choose_t0_forward,
            'p_choose_t0_backward': p_choose_t0_backward, 
            'p_choose_t0_corrected': p_choose_t0_corrected,
            'temp_delta': temp_delta,
            'position_bias': position_bias
        })
    
    return pd.DataFrame(doc_results)


def plot_simple_averaging_results(results_df: pd.DataFrame, output_dir: str, split_name: str) -> None:
    """
    Create visualizations for simple averaging results.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Simple Order-Averaging Analysis - {split_name.title()} Split', 
                 fontsize=16, fontweight='bold')
    
    # Temperature delta histogram
    axes[0, 0].hist(results_df['temp_delta'], bins=30, alpha=0.7, color='steelblue', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.8, label='No preference')
    axes[0, 0].axvline(results_df['temp_delta'].mean(), color='orange', linestyle='-', alpha=0.8,
                      label=f'Mean = {results_df["temp_delta"].mean():.3f}')
    axes[0, 0].set_xlabel('Temperature Preference Delta')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Temperature Preference Distribution\n> 0: Prefers T=0.0 (Low Temp)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Position bias histogram
    axes[0, 1].hist(results_df['position_bias'], bins=30, alpha=0.7, color='forestgreen', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8, label='No bias')
    axes[0, 1].axvline(results_df['position_bias'].mean(), color='orange', linestyle='-', alpha=0.8,
                      label=f'Mean = {results_df["position_bias"].mean():.3f}')
    axes[0, 1].set_xlabel('Position Bias')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Position Bias Distribution\n> 0: First Position Advantage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot: position bias vs temperature preference
    axes[1, 0].scatter(results_df['position_bias'], results_df['temp_delta'], alpha=0.6, s=20)
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Position Bias')
    axes[1, 0].set_ylabel('Temperature Preference Delta')
    axes[1, 0].set_title('Position Bias vs Temperature Preference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    
    mean_temp_delta = results_df['temp_delta'].mean()
    std_temp_delta = results_df['temp_delta'].std()
    mean_pos_bias = results_df['position_bias'].mean()
    std_pos_bias = results_df['position_bias'].std()
    
    # Proportion preferring each temperature
    prop_prefer_t0 = (results_df['temp_delta'] > 0).mean()
    prop_prefer_t1 = (results_df['temp_delta'] < 0).mean()
    prop_first_bias = (results_df['position_bias'] > 0).mean()
    
    temp_delta_ci = np.percentile(results_df['temp_delta'], [2.5, 97.5])
    pos_bias_ci = np.percentile(results_df['position_bias'], [2.5, 97.5])
    
    summary_text = f"""
    Simple Averaging Results
    ────────────────────────────
    
    Temperature Preference:
    Mean Δ: {mean_temp_delta:.4f} ± {std_temp_delta:.4f}
    95% CI: [{temp_delta_ci[0]:.3f}, {temp_delta_ci[1]:.3f}]
    
    Position Bias:
    Mean: {mean_pos_bias:.4f} ± {std_pos_bias:.4f}
    95% CI: [{pos_bias_ci[0]:.3f}, {pos_bias_ci[1]:.3f}]
    
    Proportions:
    ────────────────────────────
    Prefers T=0.0: {prop_prefer_t0:.3f}
    Prefers T=1.0: {prop_prefer_t1:.3f}
    First pos. bias: {prop_first_bias:.3f}
    
    Sample Size: {len(results_df)} documents
    
    Method: Position-bias-corrected
    averaging across forward/backward
    presentation orders.
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'simple_averaging_results_{split_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved simple averaging visualization: {plot_file}")


def analyze_simple_averaging(run_name: str, split_name: str) -> None:
    """
    Complete simple averaging analysis for a dataset split.
    """
    
    print(f"\n{'='*70}")
    print(f"Simple Order-Averaging Analysis: {split_name}")
    print(f"{'='*70}")
    
    # Setup paths
    results_dir = f"results_and_data/results/e1_temperature_comparison/{run_name}/{split_name}"
    results_file = os.path.join(results_dir, "choice_results.csv")
    
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        return
    
    # Load data
    print(f"Loading results from: {results_file}")
    raw_results = pd.read_csv(results_file)
    print(f"Loaded {len(raw_results)} raw comparisons")
    
    # Compute simple averaging
    results_df = compute_simple_averaging(raw_results)
    print(f"Processed {len(results_df)} document pairs")
    
    # Print summary
    mean_delta = results_df['temp_delta'].mean()
    mean_bias = results_df['position_bias'].mean()
    prop_prefer_t0 = (results_df['temp_delta'] > 0).mean()
    
    print(f"\nSimple Averaging Results:")
    print(f"  Mean temperature delta: {mean_delta:.4f} ± {results_df['temp_delta'].std():.4f}")
    print(f"  Mean position bias: {mean_bias:.4f} ± {results_df['position_bias'].std():.4f}")
    print(f"  Proportion preferring T=0.0: {prop_prefer_t0:.3f}")
    print(f"  Proportion preferring T=1.0: {1-prop_prefer_t0:.3f}")
    
    # Save processed data
    output_file = os.path.join(results_dir, "simple_averaging_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved results: {output_file}")
    
    # Create visualization
    plot_simple_averaging_results(results_df, results_dir, split_name)
    
    return results_df


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python simple_averaging_analysis.py /path/to/config.yaml")
        sys.exit(1)
    
    # Load configuration
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    print(f"Simple Order-Averaging Analysis:")
    print(f"  Run: {args.args_name}")
    print(f"  Splits: {args.splits}")
    
    # Run analysis for each split
    all_results = {}
    for split_name in args.splits:
        results_df = analyze_simple_averaging(args.args_name, split_name)
        all_results[split_name] = results_df
    
    print(f"\n{'='*70}")
    print("Simple Averaging Analysis Complete!")
    print(f"{'='*70}")