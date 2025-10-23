#!/usr/bin/env python3
"""
Create preference vs recognition comparison scatter plots from analyze_results_2 CSV files.

This script compares preference vs recognition accuracy rates
with AT data in red and UT data in blue. Creates three plots:
1. Model comparison treatments only
2. Caps & Typos treatments only
3. All treatments combined (model comparisons + caps/typos)
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


def create_pref_rec_comparison_plot(at_df: pd.DataFrame, ut_df: pd.DataFrame, title: str, output_path: str) -> None:
    """
    Create preference vs recognition comparison scatter plot.
    
    Args:
        at_df: AT accuracy DataFrame
        ut_df: UT accuracy DataFrame
        title: Plot title
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Collect data for trendlines
    at_pref_data = []
    at_rec_data = []
    ut_pref_data = []
    ut_rec_data = []
    
    # Plot each row-column combination
    for row_idx in at_df.index:
        for col in at_df.columns:
            if col.startswith('pref_'):
                # Get preference and recognition values for this model
                pref_col = col
                rec_col = col.replace('pref_', 'rec_')
                
                if rec_col not in at_df.columns:
                    continue
                
                at_pref_val = at_df.loc[row_idx, pref_col]
                at_rec_val = at_df.loc[row_idx, rec_col]
                ut_pref_val = ut_df.loc[row_idx, pref_col]
                ut_rec_val = ut_df.loc[row_idx, rec_col]
                
                # Skip NaN values
                if pd.isna(at_pref_val) or pd.isna(at_rec_val):
                    continue
                
                # Collect AT data (x=rec, y=pref)
                at_pref_data.append(at_pref_val)
                at_rec_data.append(at_rec_val)
                # Plot AT point
                plt.scatter(at_rec_val, at_pref_val, 
                           marker='o', 
                           color='#d62728',  # Red for AT
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
                
                # Skip NaN values for UT
                if not pd.isna(ut_pref_val) and not pd.isna(ut_rec_val):
                    # Collect UT data (x=rec, y=pref)
                    ut_pref_data.append(ut_pref_val)
                    ut_rec_data.append(ut_rec_val)
                    # Plot UT point
                    plt.scatter(ut_rec_val, ut_pref_val, 
                               marker='o', 
                               color='#1f77b4',  # Blue for UT
                               s=100, 
                               alpha=0.7,
                               edgecolors='black',
                               linewidth=0.5)
    
    # Add trendlines with correlation coefficients
    if len(at_pref_data) > 1 and len(at_rec_data) > 1:
        # AT trendline
        at_rec_array = np.array(at_rec_data).reshape(-1, 1)
        at_pref_array = np.array(at_pref_data)
        at_model = LinearRegression().fit(at_rec_array, at_pref_array)
        # Calculate Pearson correlation
        at_corr, at_p_value = pearsonr(at_rec_data, at_pref_data)
        at_rec_range = np.linspace(0, 1.02, 100)
        at_pref_pred = at_model.predict(at_rec_range.reshape(-1, 1))
        # Keep original predictions (no clipping)
        plt.plot(at_rec_range, at_pref_pred, color='#d62728', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'AT Trend (r={at_corr:.3f})')
    
    if len(ut_pref_data) > 1 and len(ut_rec_data) > 1:
        # UT trendline
        ut_rec_array = np.array(ut_rec_data).reshape(-1, 1)
        ut_pref_array = np.array(ut_pref_data)
        ut_model = LinearRegression().fit(ut_rec_array, ut_pref_array)
        # Calculate Pearson correlation
        ut_corr, ut_p_value = pearsonr(ut_rec_data, ut_pref_data)
        ut_rec_range = np.linspace(0, 1.02, 100)
        ut_pref_pred = ut_model.predict(ut_rec_range.reshape(-1, 1))
        # Keep original predictions (no clipping)
        plt.plot(ut_rec_range, ut_pref_pred, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'UT Trend (r={ut_corr:.3f})')
    
    # Customize plot
    plt.xlabel('Recognition Accuracy', fontsize=12)
    plt.ylabel('Preference Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to 0-1.02
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    
    # Add diagonal line (Pref = Rec)
    plt.plot([0, 1.02], [0, 1.02], 'k--', alpha=0.5, linewidth=1)
    
    # Create legend for data types (colors, trendlines, and diagonal)
    # Get correlation values for legend labels
    at_corr_text = ""
    ut_corr_text = ""
    if len(at_pref_data) > 1 and len(at_rec_data) > 1:
        at_corr, _ = pearsonr(at_rec_data, at_pref_data)
        at_corr_text = f" (r={at_corr:.3f})"
    if len(ut_pref_data) > 1 and len(ut_rec_data) > 1:
        ut_corr, _ = pearsonr(ut_rec_data, ut_pref_data)
        ut_corr_text = f" (r={ut_corr:.3f})"
    
    data_type_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#d62728', linestyle='None', markersize=8, label='AT'),
        plt.Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None', markersize=8, label='UT'),
        plt.Line2D([0], [0], marker='', color='#d62728', linestyle='-', linewidth=2, label=f'AT Trend{at_corr_text}'),
        plt.Line2D([0], [0], marker='', color='#1f77b4', linestyle='-', linewidth=2, label=f'UT Trend{ut_corr_text}'),
        plt.Line2D([0], [0], marker='', color='black', linestyle='--', linewidth=1, alpha=0.5, label='Pref = Rec')
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
    
    print(f"✅ Created preference vs recognition comparison plot: {output_path}")


def process_pref_rec_comparison(input_dir: str, output_dir: str) -> None:
    """
    Process AT and UT CSV files and create preference vs recognition comparison plots.
    
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
        
        # Create three comparison plots
        
        # 1. Model comparisons only
        create_pref_rec_comparison_plot(at_model_comps_df, ut_model_comps_df, 
                                      "Preference vs Recognition Accuracy (Model Comparisons)", 
                                      os.path.join(output_dir, "Pref_vs_Rec_model_comps.png"))
        
        # 2. Caps/typos only
        create_pref_rec_comparison_plot(at_caps_typo_df, ut_caps_typo_df, 
                                      "Preference vs Recognition Accuracy (Caps & Typos)", 
                                      os.path.join(output_dir, "Pref_vs_Rec_caps_typo.png"))
        
        # 3. Combined (model comparisons + caps/typos)
        at_combined_df = pd.concat([at_model_comps_df, at_caps_typo_df], axis=0)
        ut_combined_df = pd.concat([ut_model_comps_df, ut_caps_typo_df], axis=0)
        
        print(f"📊 Combined AT data: {at_combined_df.shape}")
        print(f"📊 Combined UT data: {ut_combined_df.shape}")
        
        create_pref_rec_comparison_plot(at_combined_df, ut_combined_df, 
                                      "Preference vs Recognition Accuracy (All Treatments)", 
                                      os.path.join(output_dir, "Pref_vs_Rec_all_treatments.png"))
        
        print(f"\n🎉 Preference vs Recognition comparison plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create preference vs recognition comparison scatter plots')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2',
                       help='Directory containing CSV files (default: results_and_data/analysis/analyze_results_2)')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/pref_rec_comparison',
                       help='Directory to save comparison plots (default: results_and_data/analysis/pref_rec_comparison)')
    
    args = parser.parse_args()
    
    print("🎯 Creating Preference vs Recognition Comparison Scatter Plots")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process preference vs recognition comparison
    process_pref_rec_comparison(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
