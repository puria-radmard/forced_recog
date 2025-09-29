#!/usr/bin/env python3
"""
Create AT vs UT combined preference and recognition scatter plots from analyze_results_2 CSV files.

This script compares Assistant Tags (AT) vs User Tags (UT) accuracy rates
with preference data in purple and recognition data in green. Creates two plots:
1. Combined plot (no treatment splitting): Two trendlines (preference + recognition)
2. Four trendlines plot: Caps/Typos (triangles, dashed) vs Model Comparisons (squares, solid)
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


def create_at_ut_combined_plot(at_df: pd.DataFrame, ut_df: pd.DataFrame, title: str, output_path: str) -> None:
    """
    Create AT vs UT combined preference and recognition scatter plot (no treatment splitting).
    
    Args:
        at_df: AT accuracy DataFrame (contains both pref and rec columns)
        ut_df: UT accuracy DataFrame (contains both pref and rec columns)
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
                
                # Determine color and data collection based on pref/rec
                if col.startswith('pref_'):
                    color = '#9467bd'  # Purple for preference
                    pref_ut_data.append(ut_val)
                    pref_at_data.append(at_val)
                else:
                    color = '#2ca02c'  # Green for recognition
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
        # Extend trendline across full chart width (0 to 1.02)
        pref_ut_range = np.linspace(0, 1.02, 100)
        pref_at_pred = pref_model.predict(pref_ut_range.reshape(-1, 1))
        plt.plot(pref_ut_range, pref_at_pred, color='#9467bd', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Preference Trend (r={pref_corr:.3f})')
    
    if len(rec_ut_data) > 1 and len(rec_at_data) > 1:
        # Recognition trendline
        rec_ut_array = np.array(rec_ut_data).reshape(-1, 1)
        rec_at_array = np.array(rec_at_data)
        rec_model = LinearRegression().fit(rec_ut_array, rec_at_array)
        # Calculate Pearson correlation
        rec_corr, rec_p_value = pearsonr(rec_ut_data, rec_at_data)
        # Extend trendline across full chart width (0 to 1.02)
        rec_ut_range = np.linspace(0, 1.02, 100)
        rec_at_pred = rec_model.predict(rec_ut_range.reshape(-1, 1))
        plt.plot(rec_ut_range, rec_at_pred, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Recognition Trend (r={rec_corr:.3f})')
    
    # Customize plot
    plt.xlabel('User Tags (UT) Accuracy', fontsize=12)
    plt.ylabel('Assistant Tags (AT) Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to 0-1.02
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    
    # Add diagonal line (AT = UT)
    plt.plot([0, 1.02], [0, 1.02], 'k--', alpha=0.5, linewidth=1)
    
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
        plt.Line2D([0], [0], marker='o', color='#9467bd', linestyle='None', markersize=8, label='Preference'),
        plt.Line2D([0], [0], marker='o', color='#2ca02c', linestyle='None', markersize=8, label='Recognition'),
        plt.Line2D([0], [0], marker='', color='#9467bd', linestyle='-', linewidth=2, label=f'Preference Trend{pref_corr_text}'),
        plt.Line2D([0], [0], marker='', color='#2ca02c', linestyle='-', linewidth=2, label=f'Recognition Trend{rec_corr_text}'),
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
    
    print(f"✅ Created AT vs UT combined plot: {output_path}")


def create_at_ut_four_trendlines_plot(at_caps_typo_df: pd.DataFrame, ut_caps_typo_df: pd.DataFrame, 
                                      at_model_comps_df: pd.DataFrame, ut_model_comps_df: pd.DataFrame,
                                      title: str, output_path: str) -> None:
    """
    Create AT vs UT plot with 4 trendlines: caps/typos vs model comparisons, pref vs rec.
    
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
    
    # Collect data for 4 trendlines
    caps_typo_pref_ut_data = []
    caps_typo_pref_at_data = []
    caps_typo_rec_ut_data = []
    caps_typo_rec_at_data = []
    model_comps_pref_ut_data = []
    model_comps_pref_at_data = []
    model_comps_rec_ut_data = []
    model_comps_rec_at_data = []
    
    # Plot caps/typo data
    for row_idx in at_caps_typo_df.index:
        for col in at_caps_typo_df.columns:
            if col.startswith('pref_') or col.startswith('rec_'):
                at_val = at_caps_typo_df.loc[row_idx, col]
                ut_val = ut_caps_typo_df.loc[row_idx, col]
                
                if pd.isna(at_val) or pd.isna(ut_val):
                    continue
                
                # Determine color and marker based on pref/rec
                if col.startswith('pref_'):
                    color = '#9467bd'  # Purple for preference
                    caps_typo_pref_ut_data.append(ut_val)
                    caps_typo_pref_at_data.append(at_val)
                    marker = '^'  # Triangle for caps/typos
                else:
                    color = '#2ca02c'  # Green for recognition
                    caps_typo_rec_ut_data.append(ut_val)
                    caps_typo_rec_at_data.append(at_val)
                    marker = '^'  # Triangle for caps/typos
                
                # Plot point
                plt.scatter(ut_val, at_val, 
                           marker=marker, 
                           color=color, 
                           s=100, 
                           alpha=0.4,
                           edgecolors='black',
                           linewidth=0.5)
    
    # Plot model comparison data
    for row_idx in at_model_comps_df.index:
        for col in at_model_comps_df.columns:
            if col.startswith('pref_') or col.startswith('rec_'):
                at_val = at_model_comps_df.loc[row_idx, col]
                ut_val = ut_model_comps_df.loc[row_idx, col]
                
                if pd.isna(at_val) or pd.isna(ut_val):
                    continue
                
                # Determine color and marker based on pref/rec
                if col.startswith('pref_'):
                    color = '#9467bd'  # Purple for preference
                    model_comps_pref_ut_data.append(ut_val)
                    model_comps_pref_at_data.append(at_val)
                    marker = 's'  # Square for model comparisons
                else:
                    color = '#2ca02c'  # Green for recognition
                    model_comps_rec_ut_data.append(ut_val)
                    model_comps_rec_at_data.append(at_val)
                    marker = 's'  # Square for model comparisons
                
                # Plot point
                plt.scatter(ut_val, at_val, 
                           marker=marker, 
                           color=color, 
                           s=100, 
                           alpha=0.4,
                           edgecolors='black',
                           linewidth=0.5)
    
    # Add 4 trendlines with correlation coefficients
    trendline_data = [
        (caps_typo_pref_ut_data, caps_typo_pref_at_data, '#9467bd', '--', 'Caps/Typos Pref'),
        (caps_typo_rec_ut_data, caps_typo_rec_at_data, '#2ca02c', '--', 'Caps/Typos Rec'),
        (model_comps_pref_ut_data, model_comps_pref_at_data, '#9467bd', '-', 'Model Comps Pref'),
        (model_comps_rec_ut_data, model_comps_rec_at_data, '#2ca02c', '-', 'Model Comps Rec')
    ]
    
    # Store correlation values for legend
    correlation_values = {}
    
    for ut_data, at_data, color, linestyle, label_prefix in trendline_data:
        if len(ut_data) > 1 and len(at_data) > 1:
            # Calculate trendline
            ut_array = np.array(ut_data).reshape(-1, 1)
            at_array = np.array(at_data)
            model = LinearRegression().fit(ut_array, at_array)
            # Calculate Pearson correlation
            corr, _ = pearsonr(ut_data, at_data)
            correlation_values[label_prefix] = corr
            # Extend trendline across full chart width (0 to 1.02)
            ut_range = np.linspace(0, 1.02, 100)
            at_pred = model.predict(ut_range.reshape(-1, 1))
            plt.plot(ut_range, at_pred, color=color, linestyle=linestyle, linewidth=3, alpha=1.0)
    
    # Customize plot
    plt.xlabel('User Tags (UT) Accuracy', fontsize=12)
    plt.ylabel('Assistant Tags (AT) Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to 0-1.02
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    
    # Add diagonal line (AT = UT)
    plt.plot([0, 1.02], [0, 1.02], 'k--', alpha=0.5, linewidth=1, label='AT = UT')
    
    # Create comprehensive legend with correlation values
    legend_elements = [
        # Color explanations
        plt.Line2D([0], [0], marker='o', color='#9467bd', linestyle='None', markersize=8, label='Preference'),
        plt.Line2D([0], [0], marker='o', color='#2ca02c', linestyle='None', markersize=8, label='Recognition'),
        # Marker explanations
        plt.Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8, label='Caps/Typos'),
        plt.Line2D([0], [0], marker='s', color='black', linestyle='None', markersize=8, label='Model Comparisons'),
    ]
    
    # Add trendlines with correlation values
    if 'Caps/Typos Pref' in correlation_values:
        legend_elements.append(plt.Line2D([0], [0], marker='', color='#9467bd', linestyle='--', linewidth=3, alpha=1.0,
                                        label=f'Caps/Typos Pref (r={correlation_values["Caps/Typos Pref"]:.3f})'))
    if 'Caps/Typos Rec' in correlation_values:
        legend_elements.append(plt.Line2D([0], [0], marker='', color='#2ca02c', linestyle='--', linewidth=3, alpha=1.0,
                                        label=f'Caps/Typos Rec (r={correlation_values["Caps/Typos Rec"]:.3f})'))
    if 'Model Comps Pref' in correlation_values:
        legend_elements.append(plt.Line2D([0], [0], marker='', color='#9467bd', linestyle='-', linewidth=3, alpha=1.0,
                                        label=f'Model Comps Pref (r={correlation_values["Model Comps Pref"]:.3f})'))
    if 'Model Comps Rec' in correlation_values:
        legend_elements.append(plt.Line2D([0], [0], marker='', color='#2ca02c', linestyle='-', linewidth=3, alpha=1.0,
                                        label=f'Model Comps Rec (r={correlation_values["Model Comps Rec"]:.3f})'))
    
    # Add diagonal line
    legend_elements.append(plt.Line2D([0], [0], marker='', color='black', linestyle='--', linewidth=1, alpha=0.5, label='AT = UT'))
    
    plt.legend(handles=legend_elements, title='Legend', loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created AT vs UT four trendlines plot: {output_path}")


def process_at_ut_combined(input_dir: str, output_dir: str) -> None:
    """
    Process AT and UT CSV files (both caps/typo and model comparison) and create combined preference and recognition plots.
    
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
        
        # Combine all data vertically (no treatment category splitting)
        at_combined_df = pd.concat([at_caps_typo_df, at_model_comps_df], axis=0)
        ut_combined_df = pd.concat([ut_caps_typo_df, ut_model_comps_df], axis=0)
        
        print(f"📊 Combined AT data: {at_combined_df.shape}")
        print(f"📊 Combined UT data: {ut_combined_df.shape}")
        
        # Create two plots
        
        # 1. Combined plot (no treatment splitting) - original request
        create_at_ut_combined_plot(at_combined_df, ut_combined_df, 
                                  "Assistant vs User Tags Accuracy", 
                                  os.path.join(output_dir, "AT_vs_UT_all_treatments.png"))
        
        # 2. Four trendlines plot (with treatment splitting)
        create_at_ut_four_trendlines_plot(at_caps_typo_df, ut_caps_typo_df, 
                                         at_model_comps_df, ut_model_comps_df,
                                         "Assistant vs User Tags Accuracy", 
                                         os.path.join(output_dir, "AT_vs_UT_four_trendlines.png"))
        
        print(f"\n🎉 AT vs UT combined plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create AT vs UT combined preference and recognition scatter plots')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2',
                       help='Directory containing CSV files (default: results_and_data/analysis/analyze_results_2)')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/at_ut_combined',
                       help='Directory to save combined plots (default: results_and_data/analysis/at_ut_combined)')
    
    args = parser.parse_args()
    
    print("🎯 Creating AT vs UT Combined Preference and Recognition Scatter Plots")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process AT vs UT combined comparison
    process_at_ut_combined(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
