#!/usr/bin/env python3
"""
Create comparison scatter plots from analyze_results_2_combined CSV files.

This script creates comparison plots between different experiment types:
- AT_2T vs UT_2T
- AT_2T vs AT_IR
- UT_2T vs UT_IR
- AT_IR vs UT_IR

Each comparison has versions for:
1. Caps/typos treatments
2. Model comparisons/other_models treatments

And two types of trendlines:
1. Pref vs Rec (only for AT_2T vs UT_2T where pref data exists)
2. Primed vs Not-Primed (rec experiments only)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


def load_combined_data(input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from analyze_results_2_combined directory.
    
    Args:
        input_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping file keys to DataFrames
    """
    data = {}
    
    # Define expected files
    # Both IR and 2T now have model_comps (IR has inferred model_other)
    exp_types = ['AT_2T', 'AT_IR', 'UT_2T', 'UT_IR']
    treatment_types = {
        'AT_2T': ['caps_typo', 'model_comps'],
        'AT_IR': ['caps_typo', 'model_comps'],
        'UT_2T': ['caps_typo', 'model_comps'],
        'UT_IR': ['caps_typo', 'model_comps']
    }
    
    for exp in exp_types:
        for treatment in treatment_types[exp]:
            key = f"{exp}_{treatment}"
            filename = f"detailed_accuracy_pivot_table_{key}.csv"
            filepath = os.path.join(input_dir, filename)
            
            if os.path.exists(filepath):
                data[key] = pd.read_csv(filepath, index_col=0)
                print(f"  [OK] Loaded {key}: {data[key].shape}")
            else:
                print(f"  [MISSING] {filename}")
    
    return data


def create_comparison_plot(df1: pd.DataFrame, df2: pd.DataFrame, 
                          label1: str, label2: str,
                          title: str, output_path: str,
                          trendline_type: str = 'pref_rec',
                          exclude_control: bool = False) -> None:
    """
    Create comparison scatter plot between two experiment types.
    
    Args:
        df1: First DataFrame (x-axis)
        df2: Second DataFrame (y-axis)
        label1: Label for x-axis
        label2: Label for y-axis
        title: Plot title
        output_path: Path to save the plot
        trendline_type: 'pref_rec' or 'primed' for trendline coloring
        exclude_control: Whether to exclude control rows
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Collect data for trendlines
    group1_x_data = []
    group1_y_data = []
    group2_x_data = []
    group2_y_data = []
    
    # Determine grouping based on trendline type
    if trendline_type == 'pref_rec':
        # Group by pref/rec
        group1_prefix = 'pref_'
        group2_prefix = 'rec_'
        group1_color = '#d62728'  # Red for preference
        group2_color = '#1f77b4'  # Blue for recognition
        group1_label = 'Preference'
        group2_label = 'Recognition'
    else:  # primed
        # Group by primed/not-primed (only rec columns)
        group1_prefix = None  # Will check for _Pr suffix
        group2_prefix = None  # Will check for _NPr suffix
        group1_color = '#2ca02c'  # Green for primed
        group2_color = '#ff7f0e'  # Orange for not-primed
        group1_label = 'Primed'
        group2_label = 'Not-Primed'
    
    # Get common rows (treatments)
    common_rows = df1.index.intersection(df2.index)
    
    # Exclude control if requested
    if exclude_control:
        common_rows = [r for r in common_rows if r != 'control']
    
    # Plot each row-column combination
    for row_idx in common_rows:
        # Get all columns from df1 and df2
        for col in df1.columns:
            # Skip if column doesn't exist in df2
            if col not in df2.columns:
                continue
            
            val1 = df1.loc[row_idx, col]
            val2 = df2.loc[row_idx, col]
            
            # Skip NaN values
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            # Determine which group this point belongs to
            if trendline_type == 'pref_rec':
                if col.startswith(group1_prefix):
                    color = group1_color
                    group1_x_data.append(val1)
                    group1_y_data.append(val2)
                elif col.startswith(group2_prefix):
                    color = group2_color
                    group2_x_data.append(val1)
                    group2_y_data.append(val2)
                else:
                    continue  # Skip columns that don't match
            else:  # primed
                # Only look at rec_ columns, check for Pr/NPr in column name
                if not col.startswith('rec_'):
                    continue
                
                # Determine if primed or not-primed based on which DataFrame it came from
                # This is a simplified approach - we'll need to track the source
                # For now, we'll use a heuristic based on column patterns
                # Columns from primed experiments vs non-primed experiments
                # Since we're comparing same columns across datasets, we just plot all as one color
                # We'll differentiate in the trendlines by dataset
                
                # For primed comparison, we want to show all points in neutral color
                # and distinguish via trendlines
                color = 'gray'
                # Add to appropriate group based on whether we're comparing primed datasets
                # We'll need to track this externally - for now add to both
                group1_x_data.append(val1)
                group1_y_data.append(val2)
            
            # Plot point
            plt.scatter(val1, val2, 
                       marker='o', 
                       color=color, 
                       s=100, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=0.5)
    
    # Add trendlines with correlation coefficients
    if trendline_type == 'pref_rec':
        # Pref vs Rec trendlines
        if len(group1_x_data) > 1 and len(group1_y_data) > 1:
            x_array = np.array(group1_x_data).reshape(-1, 1)
            y_array = np.array(group1_y_data)
            model = LinearRegression().fit(x_array, y_array)
            corr, _ = pearsonr(group1_x_data, group1_y_data)
            x_range = np.linspace(0, 1, 100)
            y_pred = model.predict(x_range.reshape(-1, 1))
            plt.plot(x_range, y_pred, color=group1_color, linestyle='-', linewidth=2, alpha=0.8, 
                    label=f'{group1_label} Trend (r={corr:.3f})')
        
        if len(group2_x_data) > 1 and len(group2_y_data) > 1:
            x_array = np.array(group2_x_data).reshape(-1, 1)
            y_array = np.array(group2_y_data)
            model = LinearRegression().fit(x_array, y_array)
            corr, _ = pearsonr(group2_x_data, group2_y_data)
            x_range = np.linspace(0, 1, 100)
            y_pred = model.predict(x_range.reshape(-1, 1))
            plt.plot(x_range, y_pred, color=group2_color, linestyle='-', linewidth=2, alpha=0.8, 
                    label=f'{group2_label} Trend (r={corr:.3f})')
    else:
        # For primed comparison, just plot one trendline for all data
        if len(group1_x_data) > 1 and len(group1_y_data) > 1:
            x_array = np.array(group1_x_data).reshape(-1, 1)
            y_array = np.array(group1_y_data)
            model = LinearRegression().fit(x_array, y_array)
            corr, _ = pearsonr(group1_x_data, group1_y_data)
            x_range = np.linspace(0, 1, 100)
            y_pred = model.predict(x_range.reshape(-1, 1))
            plt.plot(x_range, y_pred, color='darkgray', linestyle='-', linewidth=2, alpha=0.8, 
                    label=f'Overall Trend (r={corr:.3f})')
    
    # Customize plot
    plt.xlabel(f'{label1} Accuracy', fontsize=12)
    plt.ylabel(f'{label2} Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to exactly 0-1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label=f'{label1} = {label2}')
    
    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Created: {os.path.basename(output_path)}")


def create_primed_comparison_plot(df1_primed: pd.DataFrame, df1_not_primed: pd.DataFrame,
                                  df2_primed: pd.DataFrame, df2_not_primed: pd.DataFrame,
                                  label1: str, label2: str,
                                  title: str, output_path: str,
                                  exclude_control: bool = False) -> None:
    """
    Create comparison plot showing primed vs not-primed trendlines.
    
    Args:
        df1_primed: Primed data for experiment 1 (x-axis)
        df1_not_primed: Not-primed data for experiment 1 (x-axis)
        df2_primed: Primed data for experiment 2 (y-axis)
        df2_not_primed: Not-primed data for experiment 2 (y-axis)
        label1: Label for x-axis
        label2: Label for y-axis
        title: Plot title
        output_path: Path to save the plot
        exclude_control: Whether to exclude control rows
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Collect data for trendlines
    primed_x_data = []
    primed_y_data = []
    not_primed_x_data = []
    not_primed_y_data = []
    
    # Get common columns (models) between datasets
    common_cols = df1_primed.columns.intersection(df2_primed.columns)
    common_cols = common_cols.intersection(df1_not_primed.columns).intersection(df2_not_primed.columns)
    
    # Filter to only rec columns (no pref)
    rec_cols = [c for c in common_cols if not c.startswith('pref_')]
    
    # Debug info
    if len(rec_cols) == 0:
        print(f"    [DEBUG] No common rec columns found for {os.path.basename(output_path)}")
        print(f"    [DEBUG] df1_primed cols: {df1_primed.columns.tolist()}")
        print(f"    [DEBUG] df2_primed cols: {df2_primed.columns.tolist()}")
    
    # For each column (model), collect all valid data points matching by order
    for col in rec_cols:
        # Get valid rows for each dataset (exclude control if needed)
        rows1_primed = [r for r in df1_primed.index if not pd.isna(df1_primed.loc[r, col])]
        rows2_primed = [r for r in df2_primed.index if not pd.isna(df2_primed.loc[r, col])]
        rows1_not_primed = [r for r in df1_not_primed.index if not pd.isna(df1_not_primed.loc[r, col])]
        rows2_not_primed = [r for r in df2_not_primed.index if not pd.isna(df2_not_primed.loc[r, col])]
        
        if exclude_control:
            rows1_primed = [r for r in rows1_primed if r != 'control']
            rows2_primed = [r for r in rows2_primed if r != 'control']
            rows1_not_primed = [r for r in rows1_not_primed if r != 'control']
            rows2_not_primed = [r for r in rows2_not_primed if r != 'control']
        
        # Match by order of appearance
        # Primed data
        min_len_primed = min(len(rows1_primed), len(rows2_primed))
        for i in range(min_len_primed):
            val1 = df1_primed.loc[rows1_primed[i], col]
            val2 = df2_primed.loc[rows2_primed[i], col]
            
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            plt.scatter(val1, val2, 
                       marker='o', 
                       color='#2ca02c',  # Green for primed
                       s=100, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=0.5)
            
            primed_x_data.append(val1)
            primed_y_data.append(val2)
        
        # Not-primed data
        min_len_not_primed = min(len(rows1_not_primed), len(rows2_not_primed))
        for i in range(min_len_not_primed):
            val1 = df1_not_primed.loc[rows1_not_primed[i], col]
            val2 = df2_not_primed.loc[rows2_not_primed[i], col]
            
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            plt.scatter(val1, val2, 
                       marker='o', 
                       color='#ff7f0e',  # Orange for not-primed
                       s=100, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=0.5)
            
            not_primed_x_data.append(val1)
            not_primed_y_data.append(val2)
    
    # Add trendlines
    if len(primed_x_data) > 1 and len(primed_y_data) > 1:
        x_array = np.array(primed_x_data).reshape(-1, 1)
        y_array = np.array(primed_y_data)
        model = LinearRegression().fit(x_array, y_array)
        corr, _ = pearsonr(primed_x_data, primed_y_data)
        x_range = np.linspace(0, 1, 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        plt.plot(x_range, y_pred, color='#2ca02c', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Primed Trend (r={corr:.3f})')
    
    if len(not_primed_x_data) > 1 and len(not_primed_y_data) > 1:
        x_array = np.array(not_primed_x_data).reshape(-1, 1)
        y_array = np.array(not_primed_y_data)
        model = LinearRegression().fit(x_array, y_array)
        corr, _ = pearsonr(not_primed_x_data, not_primed_y_data)
        x_range = np.linspace(0, 1, 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        plt.plot(x_range, y_pred, color='#ff7f0e', linestyle='-', linewidth=2, alpha=0.8, 
                label=f'Not-Primed Trend (r={corr:.3f})')
    
    # Customize plot
    plt.xlabel(f'{label1} Accuracy', fontsize=12)
    plt.ylabel(f'{label2} Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label=f'{label1} = {label2}')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#2ca02c', linestyle='None', markersize=8, label='Primed'),
        plt.Line2D([0], [0], marker='o', color='#ff7f0e', linestyle='None', markersize=8, label='Not-Primed'),
    ]
    
    # Get existing legend from plot
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_elements.extend(handles)
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Created: {os.path.basename(output_path)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create combined experiment comparison plots')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2_combined',
                       help='Directory containing CSV files')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/combined_comparison',
                       help='Directory to save comparison plots')
    
    args = parser.parse_args()
    
    # Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Convert to absolute paths
    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(project_root, args.input_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    print("="*80)
    print("CREATING COMBINED EXPERIMENT COMPARISON PLOTS")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_combined_data(args.input_dir)
    print()
    
    # Create pref vs rec plots (only AT_2T vs UT_2T)
    print("Creating Pref vs Rec comparison plots...")
    if 'AT_2T_caps_typo' in data and 'UT_2T_caps_typo' in data:
        create_comparison_plot(
            data['UT_2T_caps_typo'], data['AT_2T_caps_typo'],
            'UT_2T', 'AT_2T',
            'AT_2T vs UT_2T Accuracy (Caps & Typos) - Pref vs Rec',
            os.path.join(args.output_dir, 'AT_2T_vs_UT_2T_caps_typo_pref_rec.png'),
            trendline_type='pref_rec'
        )
    
    if 'AT_2T_model_comps' in data and 'UT_2T_model_comps' in data:
        create_comparison_plot(
            data['UT_2T_model_comps'], data['AT_2T_model_comps'],
            'UT_2T', 'AT_2T',
            'AT_2T vs UT_2T Accuracy (Model Comparisons) - Pref vs Rec',
            os.path.join(args.output_dir, 'AT_2T_vs_UT_2T_model_comps_pref_rec.png'),
            trendline_type='pref_rec'
        )
    print()
    
    # For primed comparisons, we need to load the original data to separate primed vs not-primed
    # Since the aggregated data combines them, we need to go back to the source
    print("Loading individual experiment data for primed comparisons...")
    
    # Load individual experiments for primed analysis
    # Use the new hyphen-named directories with inferred model_other for IR
    individual_data = {}
    exp_configs = {
        'AT_2T_rec_Pr': ('WikiSum-AT_2T_rec_Pr', ['caps_typo', 'model_comps']),
        'AT_2T_rec_NPr': ('WikiSum-AT_2T_rec_NPr', ['caps_typo', 'model_comps']),
        'UT_2T_rec_Pr': ('WikiSum-UT_2T_rec_Pr', ['caps_typo', 'model_comps']),
        'UT_2T_rec_NPr': ('WikiSum-UT_2T_rec_NPr', ['caps_typo', 'model_comps']),
        'AT_IR_rec_Pr': ('WikiSum-AT_IR_rec_Pr', ['caps_typo', 'model_comps']),
        'AT_IR_rec_NPr': ('WikiSum-AT_IR_rec_NPr', ['caps_typo', 'model_comps']),
        'UT_IR_rec_Pr': ('WikiSum-UT_IR_rec_Pr', ['caps_typo', 'model_comps']),
        'UT_IR_rec_NPr': ('WikiSum-UT_IR_rec_NPr', ['caps_typo', 'model_comps']),
    }
    
    # Go up to analyze_results_1_combined
    individual_dir = os.path.join(project_root, 'results_and_data', 'analysis', 'analyze_results_1_combined')
    
    for key, (subdir, treatments) in exp_configs.items():
        individual_data[key] = {}
        for treatment in treatments:
            filepath = os.path.join(individual_dir, subdir, f'detailed_accuracy_pivot_table.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0)
                # Filter rows for this treatment type
                if treatment == 'caps_typo':
                    rows = [r for r in df.index if r in ['typo_S2', 'typo_S4', 'capitalization_S2', 'capitalization_S4']]
                elif treatment == 'model_comps':
                    rows = [r for r in df.index if 'model_comparison' in r]
                else:  # other_models
                    rows = [r for r in df.index if r in ['control', 'other_model']]
                
                individual_data[key][treatment] = df.loc[rows]
                print(f"  [OK] Loaded {key} {treatment}: {individual_data[key][treatment].shape}")
    print()
    
    # Create primed vs not-primed comparison plots
    print("Creating Primed vs Not-Primed comparison plots...")
    
    # AT_2T vs UT_2T
    if all(k in individual_data for k in ['AT_2T_rec_Pr', 'AT_2T_rec_NPr', 'UT_2T_rec_Pr', 'UT_2T_rec_NPr']):
        for treatment in ['caps_typo', 'model_comps']:
            if all(treatment in individual_data[k] for k in ['AT_2T_rec_Pr', 'AT_2T_rec_NPr', 'UT_2T_rec_Pr', 'UT_2T_rec_NPr']):
                create_primed_comparison_plot(
                    individual_data['UT_2T_rec_Pr'][treatment],
                    individual_data['UT_2T_rec_NPr'][treatment],
                    individual_data['AT_2T_rec_Pr'][treatment],
                    individual_data['AT_2T_rec_NPr'][treatment],
                    'UT_2T', 'AT_2T',
                    f'AT_2T vs UT_2T Accuracy ({treatment.replace("_", " ").title()}) - Primed vs Not-Primed',
                    os.path.join(args.output_dir, f'AT_2T_vs_UT_2T_{treatment}_primed.png')
                )
    
    # AT_2T vs AT_IR
    if all(k in individual_data for k in ['AT_2T_rec_Pr', 'AT_2T_rec_NPr', 'AT_IR_rec_Pr', 'AT_IR_rec_NPr']):
        # Caps_typo
        if 'caps_typo' in individual_data['AT_2T_rec_Pr'] and 'caps_typo' in individual_data['AT_IR_rec_Pr']:
            create_primed_comparison_plot(
                individual_data['AT_IR_rec_Pr']['caps_typo'],
                individual_data['AT_IR_rec_NPr']['caps_typo'],
                individual_data['AT_2T_rec_Pr']['caps_typo'],
                individual_data['AT_2T_rec_NPr']['caps_typo'],
                'AT_IR', 'AT_2T',
                'AT_2T vs AT_IR Accuracy (Caps Typo) - Primed vs Not-Primed',
                os.path.join(args.output_dir, 'AT_2T_vs_AT_IR_caps_typo_primed.png')
            )
        
        # Model_comps for both (IR now has model_comps with inferred model_other)
        if 'model_comps' in individual_data['AT_2T_rec_Pr'] and 'model_comps' in individual_data['AT_IR_rec_Pr']:
            create_primed_comparison_plot(
                individual_data['AT_IR_rec_Pr']['model_comps'],
                individual_data['AT_IR_rec_NPr']['model_comps'],
                individual_data['AT_2T_rec_Pr']['model_comps'],
                individual_data['AT_2T_rec_NPr']['model_comps'],
                'AT_IR', 'AT_2T',
                'AT_2T vs AT_IR Accuracy (Model Comps) - Primed vs Not-Primed',
                os.path.join(args.output_dir, 'AT_2T_vs_AT_IR_model_comps_primed.png'),
                exclude_control=False  # No control in model_comps rows
            )
    
    # UT_2T vs UT_IR
    if all(k in individual_data for k in ['UT_2T_rec_Pr', 'UT_2T_rec_NPr', 'UT_IR_rec_Pr', 'UT_IR_rec_NPr']):
        # Caps_typo
        if 'caps_typo' in individual_data['UT_2T_rec_Pr'] and 'caps_typo' in individual_data['UT_IR_rec_Pr']:
            create_primed_comparison_plot(
                individual_data['UT_IR_rec_Pr']['caps_typo'],
                individual_data['UT_IR_rec_NPr']['caps_typo'],
                individual_data['UT_2T_rec_Pr']['caps_typo'],
                individual_data['UT_2T_rec_NPr']['caps_typo'],
                'UT_IR', 'UT_2T',
                'UT_2T vs UT_IR Accuracy (Caps Typo) - Primed vs Not-Primed',
                os.path.join(args.output_dir, 'UT_2T_vs_UT_IR_caps_typo_primed.png')
            )
        
        # Model_comps for both (IR now has model_comps with inferred model_other)
        if 'model_comps' in individual_data['UT_2T_rec_Pr'] and 'model_comps' in individual_data['UT_IR_rec_Pr']:
            create_primed_comparison_plot(
                individual_data['UT_IR_rec_Pr']['model_comps'],
                individual_data['UT_IR_rec_NPr']['model_comps'],
                individual_data['UT_2T_rec_Pr']['model_comps'],
                individual_data['UT_2T_rec_NPr']['model_comps'],
                'UT_IR', 'UT_2T',
                'UT_2T vs UT_IR Accuracy (Model Comps) - Primed vs Not-Primed',
                os.path.join(args.output_dir, 'UT_2T_vs_UT_IR_model_comps_primed.png'),
                exclude_control=False  # No control in model_comps rows
            )
    
    # AT_IR vs UT_IR
    if all(k in individual_data for k in ['AT_IR_rec_Pr', 'AT_IR_rec_NPr', 'UT_IR_rec_Pr', 'UT_IR_rec_NPr']):
        for treatment in ['caps_typo', 'model_comps']:
            if all(treatment in individual_data[k] for k in ['AT_IR_rec_Pr', 'AT_IR_rec_NPr', 'UT_IR_rec_Pr', 'UT_IR_rec_NPr']):
                create_primed_comparison_plot(
                    individual_data['UT_IR_rec_Pr'][treatment],
                    individual_data['UT_IR_rec_NPr'][treatment],
                    individual_data['AT_IR_rec_Pr'][treatment],
                    individual_data['AT_IR_rec_NPr'][treatment],
                    'UT_IR', 'AT_IR',
                    f'AT_IR vs UT_IR Accuracy ({treatment.replace("_", " ").title()}) - Primed vs Not-Primed',
                    os.path.join(args.output_dir, f'AT_IR_vs_UT_IR_{treatment}_primed.png'),
                    exclude_control=False  # No control in model_comps rows
                )
    
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

