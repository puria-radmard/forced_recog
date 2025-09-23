#!/usr/bin/env python3
"""
Create AT vs UT recognition comparison scatter plots from analyze_results_2 CSV files.

This script compares Assistant Tags (AT) vs User Tags (UT) recognition accuracy rates
across different treatment categories, excluding preference data. Creates four plots:
1. Caps & Typos treatments only
2. Model comparison treatments only  
3. All treatments combined (single trendline)
4. All treatments combined (dual trendlines: caps/typo vs model comparison)
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


def clean_treatment_name(treatment: str) -> str:
    """
    Clean treatment name for display.
    
    Args:
        treatment: Original treatment name
        
    Returns:
        Cleaned treatment name
    """
    # Handle capitalization treatments
    if 'capitalization' in treatment:
        if 'S2' in treatment:
            return "cApItALizaTiOn"
        elif 'S4' in treatment:
            return "CAPITALIZATION"
        else:
            return "Capitalization"
    # Handle typo treatments
    elif 'typo' in treatment:
        if 'S2' in treatment:
            return "Few Typos (Ex@mple)"
        elif 'S4' in treatment:
            return "Many Typos (Exx@mpl)"
        else:
            return "Typos"
    else:
        return treatment


def create_at_ut_recognition_plot(at_df: pd.DataFrame, ut_df: pd.DataFrame, title: str, output_path: str) -> None:
    """
    Create AT vs UT recognition comparison scatter plot.
    
    Args:
        at_df: AT recognition DataFrame
        ut_df: UT recognition DataFrame
        title: Plot title
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Collect data for trendline (only recognition data)
    rec_ut_data = []
    rec_at_data = []
    
    # Plot each row-column combination
    for row_idx in at_df.index:
        for col in at_df.columns:
            # Only process recognition columns
            if col.startswith('rec_'):
                at_val = at_df.loc[row_idx, col]
                ut_val = ut_df.loc[row_idx, col]
                
                # Skip NaN values
                if pd.isna(at_val) or pd.isna(ut_val):
                    continue
                
                # Collect data for trendline
                rec_ut_data.append(ut_val)
                rec_at_data.append(at_val)
                
                # Plot point as simple dot (blue for recognition)
                plt.scatter(ut_val, at_val, 
                           marker='o', 
                           color='#1f77b4',  # Blue for recognition
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
    
    # Add trendline with correlation coefficient
    if len(rec_ut_data) > 1 and len(rec_at_data) > 1:
        # Recognition trendline
        rec_ut_array = np.array(rec_ut_data).reshape(-1, 1)
        rec_at_array = np.array(rec_at_data)
        rec_model = LinearRegression().fit(rec_ut_array, rec_at_array)
        # Calculate Pearson correlation
        rec_corr, rec_p_value = pearsonr(rec_ut_data, rec_at_data)
        # Extend trendline across full chart width (0 to 1)
        rec_ut_range = np.linspace(0, 1.02, 100)
        rec_at_pred = rec_model.predict(rec_ut_range.reshape(-1, 1))
        plt.plot(rec_ut_range, rec_at_pred, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Recognition Trend (r={rec_corr:.3f})')
    
    # Customize plot
    plt.xlabel('User Tags (UT) Recognition Accuracy', fontsize=12)
    plt.ylabel('Assistant Tags (AT) Recognition Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to 0-1.02
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    
    # Add diagonal line (AT = UT)
    plt.plot([0, 1.02], [0, 1.02], 'k--', alpha=0.5, linewidth=1)
    
    # Create legend for data types (recognition data, trendline, and diagonal)
    # Get correlation value for legend label
    rec_corr_text = ""
    if len(rec_ut_data) > 1 and len(rec_at_data) > 1:
        rec_corr, _ = pearsonr(rec_ut_data, rec_at_data)
        rec_corr_text = f" (r={rec_corr:.3f})"
    
    data_type_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None', markersize=8, label='Recognition'),
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
    
    print(f"✅ Created AT vs UT recognition plot: {output_path}")


def create_at_ut_recognition_dual_trendlines_plot(at_caps_typo_df: pd.DataFrame, ut_caps_typo_df: pd.DataFrame, 
                                                 at_model_comps_df: pd.DataFrame, ut_model_comps_df: pd.DataFrame,
                                                 title: str, output_path: str) -> None:
    """
    Create AT vs UT recognition comparison scatter plot with two trendlines.
    
    Args:
        at_caps_typo_df: AT caps/typo DataFrame
        ut_caps_typo_df: UT caps/typo DataFrame
        at_model_comps_df: AT model comparison DataFrame
        ut_model_comps_df: UT model comparison DataFrame
        title: Plot title
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Collect data for caps/typo trendline
    caps_typo_ut_data = []
    caps_typo_at_data = []
    
    # Plot caps/typo data
    for row_idx in at_caps_typo_df.index:
        for col in at_caps_typo_df.columns:
            if col.startswith('rec_'):
                at_val = at_caps_typo_df.loc[row_idx, col]
                ut_val = ut_caps_typo_df.loc[row_idx, col]
                
                if pd.isna(at_val) or pd.isna(ut_val):
                    continue
                
                caps_typo_ut_data.append(ut_val)
                caps_typo_at_data.append(at_val)
                
                # Plot point as blue dot for caps/typo
                plt.scatter(ut_val, at_val, 
                           marker='o', 
                           color='#1f77b4',  # Blue for caps/typo
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5,
                           label='Caps & Typos' if len(caps_typo_ut_data) == 1 else "")
    
    # Collect data for model comparison trendline
    model_comps_ut_data = []
    model_comps_at_data = []
    
    # Plot model comparison data
    for row_idx in at_model_comps_df.index:
        for col in at_model_comps_df.columns:
            if col.startswith('rec_'):
                at_val = at_model_comps_df.loc[row_idx, col]
                ut_val = ut_model_comps_df.loc[row_idx, col]
                
                if pd.isna(at_val) or pd.isna(ut_val):
                    continue
                
                model_comps_ut_data.append(ut_val)
                model_comps_at_data.append(at_val)
                
                # Plot point as red dot for model comparisons
                plt.scatter(ut_val, at_val, 
                           marker='o', 
                           color='#d62728',  # Red for model comparisons
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5,
                           label='Model Comparisons' if len(model_comps_ut_data) == 1 else "")
    
    # Add trendlines with correlation coefficients
    if len(caps_typo_ut_data) > 1 and len(caps_typo_at_data) > 1:
        # Caps/typo trendline
        caps_typo_ut_array = np.array(caps_typo_ut_data).reshape(-1, 1)
        caps_typo_at_array = np.array(caps_typo_at_data)
        caps_typo_model = LinearRegression().fit(caps_typo_ut_array, caps_typo_at_array)
        caps_typo_corr, _ = pearsonr(caps_typo_ut_data, caps_typo_at_data)
        caps_typo_ut_range = np.linspace(0, 1.02, 100)
        caps_typo_at_pred = caps_typo_model.predict(caps_typo_ut_range.reshape(-1, 1))
        plt.plot(caps_typo_ut_range, caps_typo_at_pred, color='#1f77b4', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Caps & Typos Trend (r={caps_typo_corr:.3f})')
    
    if len(model_comps_ut_data) > 1 and len(model_comps_at_data) > 1:
        # Model comparison trendline
        model_comps_ut_array = np.array(model_comps_ut_data).reshape(-1, 1)
        model_comps_at_array = np.array(model_comps_at_data)
        model_comps_model = LinearRegression().fit(model_comps_ut_array, model_comps_at_array)
        model_comps_corr, _ = pearsonr(model_comps_ut_data, model_comps_at_data)
        model_comps_ut_range = np.linspace(0, 1.02, 100)
        model_comps_at_pred = model_comps_model.predict(model_comps_ut_range.reshape(-1, 1))
        plt.plot(model_comps_ut_range, model_comps_at_pred, color='#d62728', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Model Comparisons Trend (r={model_comps_corr:.3f})')
    
    # Customize plot
    plt.xlabel('User Tags (UT) Recognition Accuracy', fontsize=12)
    plt.ylabel('Assistant Tags (AT) Recognition Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to 0-1.02
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    
    # Add diagonal line (AT = UT)
    plt.plot([0, 1.02], [0, 1.02], 'k--', alpha=0.5, linewidth=1, label='AT = UT')
    
    # Add legend
    plt.legend(title='Treatment Type', loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created AT vs UT dual trendline plot: {output_path}")


def process_at_ut_recognition(input_dir: str, output_dir: str) -> None:
    """
    Process AT and UT CSV files (both caps/typo and model comparison) and create recognition comparison plots.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save scatter plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load AT and UT data from both caps/typo and model comparison files
    at_caps_typo_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_AT_caps_typo.csv')
    ut_caps_typo_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_UT_caps_typo.csv')
    at_model_comps_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_AT_model_comps.csv')
    ut_model_comps_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_UT_model_comps.csv')
    
    if not all(os.path.exists(f) for f in [at_caps_typo_file, ut_caps_typo_file, at_model_comps_file, ut_model_comps_file]):
        print("❌ Required CSV files not found!")
        print(f"Looking for: {at_caps_typo_file}")
        print(f"Looking for: {ut_caps_typo_file}")
        print(f"Looking for: {at_model_comps_file}")
        print(f"Looking for: {ut_model_comps_file}")
        return
    
    try:
        # Read CSV files
        at_caps_typo_df = pd.read_csv(at_caps_typo_file, index_col=0)
        ut_caps_typo_df = pd.read_csv(ut_caps_typo_file, index_col=0)
        at_model_comps_df = pd.read_csv(at_model_comps_file, index_col=0)
        ut_model_comps_df = pd.read_csv(ut_model_comps_file, index_col=0)
        
        print(f"📊 Loaded AT caps/typo data: {at_caps_typo_df.shape}")
        print(f"📊 Loaded UT caps/typo data: {ut_caps_typo_df.shape}")
        print(f"📊 Loaded AT model comps data: {at_model_comps_df.shape}")
        print(f"📊 Loaded UT model comps data: {ut_model_comps_df.shape}")
        
        # Combine all data vertically
        at_combined_df = pd.concat([at_caps_typo_df, at_model_comps_df], axis=0)
        ut_combined_df = pd.concat([ut_caps_typo_df, ut_model_comps_df], axis=0)
        
        print(f"📊 Combined AT data: {at_combined_df.shape}")
        print(f"📊 Combined UT data: {ut_combined_df.shape}")
        
        # Create three separate plots
        
        # 1. Caps/Typos only
        create_at_ut_recognition_plot(at_caps_typo_df, ut_caps_typo_df, 
                                     "AT vs UT Recognition Accuracy (Caps & Typos)", 
                                     os.path.join(output_dir, "AT_vs_UT_recognition_caps_typo.png"))
        
        # 2. Model comparisons only
        create_at_ut_recognition_plot(at_model_comps_df, ut_model_comps_df, 
                                     "AT vs UT Recognition Accuracy (Model Comparisons)", 
                                     os.path.join(output_dir, "AT_vs_UT_recognition_model_comps.png"))
        
        # 3. Combined (all treatments)
        create_at_ut_recognition_plot(at_combined_df, ut_combined_df, 
                                     "AT vs UT Recognition Accuracy (All Treatments)", 
                                     os.path.join(output_dir, "AT_vs_UT_recognition_all_treatments.png"))
        
        # 4. Dual trendlines (caps/typo vs model comparison)
        create_at_ut_recognition_dual_trendlines_plot(at_caps_typo_df, ut_caps_typo_df, 
                                                     at_model_comps_df, ut_model_comps_df,
                                                     "Assistant vs User Tags Recognition Accuracy", 
                                                     os.path.join(output_dir, "AT_vs_UT_recognition_dual_trendlines.png"))
        
        print(f"\n🎉 AT vs UT recognition plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create AT vs UT recognition comparison scatter plots')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2',
                       help='Directory containing CSV files (default: results_and_data/analysis/analyze_results_2)')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/at_ut_recognition',
                       help='Directory to save recognition comparison plots (default: results_and_data/analysis/at_ut_recognition)')
    
    args = parser.parse_args()
    
    print("🎯 Creating AT vs UT Recognition Comparison Scatter Plots")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process AT vs UT recognition comparison
    process_at_ut_recognition(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
