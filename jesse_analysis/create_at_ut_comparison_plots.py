#!/usr/bin/env python3
"""
Create AT vs UT comparison scatter plots from analyze_results_2 CSV files.

This script compares Assistant Tags (AT) vs User Tags (UT) accuracy rates
with preference data in red and recognition data in blue. Creates two plots:
1. Model comparison treatments
2. Caps & Typos treatments
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List, Tuple
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


def extract_model_name(column_name: str) -> str:
    """
    Extract model name from column name (e.g., 'pref_claude-3-5-haiku-20241022' -> 'claude-3-5-haiku').
    
    Args:
        column_name: Column name with pref/rec prefix
        
    Returns:
        Cleaned model name
    """
    # Remove pref_ or rec_ prefix
    if column_name.startswith('pref_'):
        model_part = column_name[5:]  # Remove 'pref_'
    elif column_name.startswith('rec_'):
        model_part = column_name[4:]  # Remove 'rec_'
    else:
        return column_name
    
    # Remove date suffix
    model_part = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_part)
    model_part = re.sub(r'-\d{8}$', '', model_part)
    
    return model_part


def get_model_size_order(model_name: str) -> int:
    """
    Get size order for model (smaller number = smaller model).
    
    Args:
        model_name: Model name
        
    Returns:
        Size order number
    """
    # Claude models (smaller to larger)
    if 'claude-3-5-haiku' in model_name:
        return 1
    elif 'claude-sonnet-4' in model_name:
        return 2
    # Gemini models (smaller to larger)
    elif 'gemini-2.5-flash' in model_name:
        return 3
    elif 'gemini-2.5-pro' in model_name:
        return 4
    # GPT models (smaller to larger)
    elif 'gpt-4o-mini' in model_name:
        return 5
    elif 'gpt-4.1-mini' in model_name:
        return 6
    elif 'gpt-4.1' in model_name and 'mini' not in model_name:
        return 7
    else:
        return 999  # Unknown models at the end


# Removed get_model_marker_map function - using simple dots now


def clean_treatment_name(treatment: str) -> str:
    """
    Clean treatment name for display.
    
    Args:
        treatment: Original treatment name
        
    Returns:
        Cleaned treatment name
    """
    # Remove model_comparison_ prefix
    cleaned = treatment.replace('model_comparison_', '')
    
    # Remove company prefixes
    cleaned = cleaned.replace('anthropic_', '')
    cleaned = cleaned.replace('google_', '')
    
    # Remove date suffixes
    cleaned = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', cleaned)
    cleaned = re.sub(r'-\d{8}$', '', cleaned)
    
    return cleaned


def create_at_ut_comparison_plot(at_df: pd.DataFrame, ut_df: pd.DataFrame, title: str, output_path: str) -> None:
    """
    Create AT vs UT comparison scatter plot.
    
    Args:
        at_df: AT accuracy DataFrame
        ut_df: UT accuracy DataFrame
        title: Plot title
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Collect data for trendlines
    pref_ut_data = []
    pref_at_data = []
    rec_ut_data = []
    rec_at_data = []
    
    # Plot each row-column combination
    for row_idx in at_df.index:
        for col in at_df.columns:
            if col.startswith('pref_') or col.startswith('rec_'):
                at_val = at_df.loc[row_idx, col]
                ut_val = ut_df.loc[row_idx, col]
                
                # Skip NaN values
                if pd.isna(at_val) or pd.isna(ut_val):
                    continue
                
                # Determine color based on pref/rec
                if col.startswith('pref_'):
                    color = '#d62728'  # Red for preference
                    pref_ut_data.append(ut_val)
                    pref_at_data.append(at_val)
                else:
                    color = '#1f77b4'  # Blue for recognition
                    rec_ut_data.append(ut_val)
                    rec_at_data.append(at_val)
                
                # Plot point as simple dot
                plt.scatter(ut_val, at_val, 
                           marker='o', 
                           color=color, 
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
    
    # Add trendlines with correlation coefficients
    if len(pref_ut_data) > 1 and len(pref_at_data) > 1:
        # Preference trendline
        pref_ut_array = np.array(pref_ut_data).reshape(-1, 1)
        pref_at_array = np.array(pref_at_data)
        pref_model = LinearRegression().fit(pref_ut_array, pref_at_array)
        # Calculate Pearson correlation
        pref_corr, pref_p_value = pearsonr(pref_ut_data, pref_at_data)
        # Extend trendline across full chart width (0 to 1)
        pref_ut_range = np.linspace(0, 1, 100)
        pref_at_pred = pref_model.predict(pref_ut_range.reshape(-1, 1))
        plt.plot(pref_ut_range, pref_at_pred, color='#d62728', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Preference Trend (r={pref_corr:.3f})')
    
    if len(rec_ut_data) > 1 and len(rec_at_data) > 1:
        # Recognition trendline
        rec_ut_array = np.array(rec_ut_data).reshape(-1, 1)
        rec_at_array = np.array(rec_at_data)
        rec_model = LinearRegression().fit(rec_ut_array, rec_at_array)
        # Calculate Pearson correlation
        rec_corr, rec_p_value = pearsonr(rec_ut_data, rec_at_data)
        # Extend trendline across full chart width (0 to 1)
        rec_ut_range = np.linspace(0, 1, 100)
        rec_at_pred = rec_model.predict(rec_ut_range.reshape(-1, 1))
        plt.plot(rec_ut_range, rec_at_pred, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Recognition Trend (r={rec_corr:.3f})')
    
    # Customize plot
    plt.xlabel('User Tags (UT) Accuracy', fontsize=12)
    plt.ylabel('Assistant Tags (AT) Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to exactly 0-1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add diagonal line (AT = UT)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Create legend for data types (colors, trendlines, and diagonal)
    # Get correlation values for legend labels
    pref_corr_text = ""
    rec_corr_text = ""
    if len(pref_ut_data) > 1 and len(pref_at_data) > 1:
        pref_corr, _ = pearsonr(pref_ut_data, pref_at_data)
        pref_corr_text = f" (r={pref_corr:.3f})"
    if len(rec_ut_data) > 1 and len(rec_at_data) > 1:
        rec_corr, _ = pearsonr(rec_ut_data, rec_at_data)
        rec_corr_text = f" (r={rec_corr:.3f})"
    
    data_type_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#d62728', linestyle='None', markersize=8, label='Preference'),
        plt.Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None', markersize=8, label='Recognition'),
        plt.Line2D([0], [0], marker='', color='#d62728', linestyle='-', linewidth=2, label=f'Preference Trend{pref_corr_text}'),
        plt.Line2D([0], [0], marker='', color='#1f77b4', linestyle='-', linewidth=2, label=f'Recognition Trend{rec_corr_text}'),
        plt.Line2D([0], [0], marker='', color='black', linestyle='--', linewidth=1, alpha=0.5, label='AT = UT')
    ]
    
    # Add legend
    if data_type_legend_elements:
        plt.legend(handles=data_type_legend_elements, 
                 title='Data Type', 
                 loc='upper left',
                 bbox_to_anchor=(1.02, 1))
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created AT vs UT comparison plot: {output_path}")


def process_at_ut_comparison(input_dir: str, output_dir: str) -> None:
    """
    Process AT and UT CSV files and create comparison plots.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save scatter plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load AT and UT data from both model comparison and caps/typo files
    at_model_comps_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_AT_model_comps.csv')
    ut_model_comps_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_UT_model_comps.csv')
    at_caps_typo_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_AT_caps_typo.csv')
    ut_caps_typo_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_UT_caps_typo.csv')
    
    if not all(os.path.exists(f) for f in [at_model_comps_file, ut_model_comps_file, at_caps_typo_file, ut_caps_typo_file]):
        print("❌ Required CSV files not found!")
        print(f"Looking for: {at_model_comps_file}")
        print(f"Looking for: {ut_model_comps_file}")
        print(f"Looking for: {at_caps_typo_file}")
        print(f"Looking for: {ut_caps_typo_file}")
        return
    
    try:
        # Read CSV files
        at_model_comps_df = pd.read_csv(at_model_comps_file, index_col=0)
        ut_model_comps_df = pd.read_csv(ut_model_comps_file, index_col=0)
        at_caps_typo_df = pd.read_csv(at_caps_typo_file, index_col=0)
        ut_caps_typo_df = pd.read_csv(ut_caps_typo_file, index_col=0)
        
        print(f"📊 Loaded AT model comps data: {at_model_comps_df.shape}")
        print(f"📊 Loaded UT model comps data: {ut_model_comps_df.shape}")
        print(f"📊 Loaded AT caps/typo data: {at_caps_typo_df.shape}")
        print(f"📊 Loaded UT caps/typo data: {ut_caps_typo_df.shape}")
        
        # Create two comparison plots
        
        # 1. Model comparisons
        create_at_ut_comparison_plot(at_model_comps_df, ut_model_comps_df, 
                                   "AT vs UT Accuracy Comparison (Model Comparisons)", 
                                   os.path.join(output_dir, "AT_vs_UT_model_comps.png"))
        
        # 2. Caps/typos
        create_at_ut_comparison_plot(at_caps_typo_df, ut_caps_typo_df, 
                                   "AT vs UT Accuracy Comparison (Caps & Typos)", 
                                   os.path.join(output_dir, "AT_vs_UT_caps_typo.png"))
        
        print(f"\n🎉 AT vs UT comparison plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create AT vs UT comparison scatter plots')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2',
                       help='Directory containing CSV files (default: results_and_data/analysis/analyze_results_2)')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/at_ut_comparison',
                       help='Directory to save comparison plots (default: results_and_data/analysis/at_ut_comparison)')
    
    args = parser.parse_args()
    
    print("🎯 Creating AT vs UT Comparison Scatter Plots")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process AT vs UT comparison
    process_at_ut_comparison(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
