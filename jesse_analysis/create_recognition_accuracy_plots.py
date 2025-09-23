#!/usr/bin/env python3
"""
Create recognition accuracy plots from analyze_results_2 CSV files.

This script creates bar plots with treatments on x-axis and recognition accuracy on y-axis.
Separate plots for model comparisons vs caps/typos treatments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List, Tuple
import re


def extract_model_name(column_name: str) -> str:
    """
    Extract model name from column name (e.g., 'rec_claude-3-5-haiku-20241022' -> 'claude-3-5-haiku').
    
    Args:
        column_name: Column name with rec_ prefix
        
    Returns:
        Cleaned model name
    """
    # Remove rec_ prefix
    if column_name.startswith('rec_'):
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


def get_model_marker_map(columns: List[str]) -> Dict[str, str]:
    """
    Create mapping from model names to marker shapes.
    
    Args:
        columns: List of column names
        
    Returns:
        Dictionary mapping model names to marker shapes
    """
    # Extract unique model names
    model_names = set()
    for col in columns:
        if col.startswith('rec_'):
            model_name = extract_model_name(col)
            model_names.add(model_name)
    
    # Sort models by size (smallest to largest)
    sorted_models = sorted(model_names, key=get_model_size_order)
    
    # Assign specific marker shapes for requested models
    model_marker_map = {}
    
    for model_name in sorted_models:
        if 'claude-3-5-haiku' in model_name:
            model_marker_map[model_name] = '^'  # Triangle
        elif 'claude-sonnet-4' in model_name:
            model_marker_map[model_name] = 'v'  # Upside down triangle
        elif 'gpt-4.1' in model_name and 'mini' not in model_name:
            model_marker_map[model_name] = 's'  # Square
        elif 'gpt-4.1-mini' in model_name:
            model_marker_map[model_name] = 'D'  # Diamond
        else:
            # Default markers for other models
            default_markers = ['o', '<', '>', 'p', '*', 'h', 'H', '1', '2', '3', '4']
            model_marker_map[model_name] = default_markers[len(model_marker_map) % len(default_markers)]
    
    return model_marker_map


def clean_treatment_name(treatment: str) -> str:
    """
    Clean treatment name for display.
    
    Args:
        treatment: Original treatment name
        
    Returns:
        Cleaned treatment name
    """
    # Handle caps and typos
    if 'capitalization' in treatment:
        if 'S2' in treatment:
            return "cApItALizaTiOn"
        elif 'S4' in treatment:
            return "CAPITALIZATION"
    elif 'typo' in treatment:
        if 'S2' in treatment:
            return "Few Typos (Ex@mple)"
        elif 'S4' in treatment:
            return "Many Typos (Exx@mpl)"
    
    # Handle model comparisons
    if 'model_comparison' in treatment:
        # Remove model_comparison_ prefix
        cleaned = treatment.replace('model_comparison_', '')
        
        # Remove company prefixes
        cleaned = cleaned.replace('anthropic_', '')
        cleaned = cleaned.replace('google_', '')
        
        # Remove date suffixes
        cleaned = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', cleaned)
        cleaned = re.sub(r'-\d{8}$', '', cleaned)
        
        return cleaned
    
    return treatment


def get_company_from_model(model_name: str) -> str:
    """
    Get company name from model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Company name
    """
    if 'claude' in model_name.lower():
        return 'claude'
    elif 'gemini' in model_name.lower():
        return 'gemini'
    elif 'gpt' in model_name.lower():
        return 'gpt'
    else:
        return 'other'


def create_recognition_plot(at_df: pd.DataFrame, ut_df: pd.DataFrame, title: str, output_path: str, is_model_comparison: bool = True) -> None:
    """
    Create recognition accuracy plot.
    
    Args:
        at_df: AT accuracy DataFrame
        ut_df: UT accuracy DataFrame
        title: Plot title
        output_path: Path to save the plot
        is_model_comparison: Whether this is model comparison data
    """
    # Create figure
    plt.figure(figsize=(18, 12) if is_model_comparison else (14, 10))
    
    # Get model marker mapping
    model_marker_map = get_model_marker_map(at_df.columns)
    
    # Get recognition columns
    rec_cols = [col for col in at_df.columns if col.startswith('rec_')]
    
    if not rec_cols:
        print(f"Warning: No recognition columns found in {title}")
        return
    
    # Plot each treatment
    x_pos = 0
    treatment_positions = {}
    treatment_labels = []
    
    for treatment in at_df.index:
        treatment_positions[treatment] = x_pos
        treatment_labels.append(clean_treatment_name(treatment))
        x_pos += 1
    
    # Store data points for connecting lines
    at_data_points = {}  # {model_name: [(x_pos, y_val), ...]}
    ut_data_points = {}  # {model_name: [(x_pos, y_val), ...]}
    
    # Plot AT and UT data for each treatment
    for treatment in at_df.index:
        x_pos = treatment_positions[treatment]
        
        for rec_col in rec_cols:
            at_val = at_df.loc[treatment, rec_col]
            ut_val = ut_df.loc[treatment, rec_col]
            
            # Skip NaN values
            if pd.isna(at_val) and pd.isna(ut_val):
                continue
            
            # Get model name and marker
            model_name = extract_model_name(rec_col)
            marker = model_marker_map.get(model_name, 'o')
            
            # Plot AT point if not NaN
            if not pd.isna(at_val):
                plt.scatter(x_pos, at_val, 
                           marker=marker, 
                           color='#d62728',  # Red for AT
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
                # Store for connecting lines
                if model_name not in at_data_points:
                    at_data_points[model_name] = []
                at_data_points[model_name].append((x_pos, at_val))
            
            # Plot UT point if not NaN
            if not pd.isna(ut_val):
                plt.scatter(x_pos, ut_val, 
                           marker=marker, 
                           color='#1f77b4',  # Blue for UT
                           s=100, 
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)
                # Store for connecting lines
                if model_name not in ut_data_points:
                    ut_data_points[model_name] = []
                ut_data_points[model_name].append((x_pos, ut_val))
    
    # Add self-comparison markers at 0.5 for each model
    # Find all unique models that appear in the data
    all_models = set()
    for model_name in model_marker_map.keys():
        all_models.add(model_name)
    
    # Get treatments from the DataFrame index
    treatments = at_df.index.tolist()
    
    # Add self-comparison points for each model
    for model_name in all_models:
        # Find the x position for this model's self-comparison
        # This would be where the model appears as both base and other
        model_self_x = None
        for i, treatment in enumerate(treatments):
            # More precise matching to avoid conflicts between gpt-4.1 and gpt-4.1-mini
            if model_name == 'gpt-4.1':
                # Only match if it's gpt-4.1 but not gpt-4.1-mini
                if 'gpt-4.1' in treatment.lower() and 'gpt-4.1-mini' not in treatment.lower():
                    model_self_x = i
                    break
            elif model_name == 'gpt-4.1-mini':
                # Only match if it contains gpt-4.1-mini
                if 'gpt-4.1-mini' in treatment.lower():
                    model_self_x = i
                    break
            else:
                # For other models, use the original matching
                if model_name in treatment.lower():
                    model_self_x = i
                    break
        
        if model_self_x is not None:
            # Get the marker for this model
            marker = model_marker_map.get(model_name, 'o')
            
            # Plot AT self-comparison point with model-specific marker (gray border, translucent center)
            plt.scatter(model_self_x, 0.5, 
                       color='gray', s=100, marker=marker, 
                       edgecolors='gray', linewidth=2, alpha=0.3, zorder=5)
            
            # Plot UT self-comparison point with model-specific marker (gray border, translucent center)
            plt.scatter(model_self_x, 0.5, 
                       color='gray', s=100, marker=marker, 
                       edgecolors='gray', linewidth=2, alpha=0.3, zorder=5)
            
            # Add to data points for connecting lines
            if model_name not in at_data_points:
                at_data_points[model_name] = []
            at_data_points[model_name].append((model_self_x, 0.5))
            
            if model_name not in ut_data_points:
                ut_data_points[model_name] = []
            ut_data_points[model_name].append((model_self_x, 0.5))
    
    # Draw connecting lines for each model across treatments
    # Group models by company for line styling
    claude_models = []
    gemini_models = []
    gpt_models = []
    
    for model_name in model_marker_map.keys():
        company = get_company_from_model(model_name)
        if company == 'claude':
            claude_models.append(model_name)
        elif company == 'gemini':
            gemini_models.append(model_name)
        elif company == 'gpt':
            gpt_models.append(model_name)
    
    # Draw AT connecting lines for each model
    for model_name in at_data_points.keys():
        if len(at_data_points[model_name]) > 1:
            # Sort points by x position (treatment order)
            sorted_points = sorted(at_data_points[model_name], key=lambda x: x[0])
            x_coords = [point[0] for point in sorted_points]
            y_coords = [point[1] for point in sorted_points]
            
            # Use solid lines for all companies
            linestyle = '-'
            alpha = 0.4
            
            plt.plot(x_coords, y_coords, 
                    color='#d62728', alpha=alpha, linewidth=1.5, linestyle=linestyle)
    
    # Draw UT connecting lines for each model
    for model_name in ut_data_points.keys():
        if len(ut_data_points[model_name]) > 1:
            # Sort points by x position (treatment order)
            sorted_points = sorted(ut_data_points[model_name], key=lambda x: x[0])
            x_coords = [point[0] for point in sorted_points]
            y_coords = [point[1] for point in sorted_points]
            
            # Use solid lines for all companies
            linestyle = '-'
            alpha = 0.4
            
            plt.plot(x_coords, y_coords, 
                    color='#1f77b4', alpha=alpha, linewidth=1.5, linestyle=linestyle)
    
    # Add horizontal dotted line at 0.5 (self-comparison baseline)
    plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    
    # Customize plot
    plt.xlabel('Treatments', fontsize=12)
    plt.ylabel('Recognition Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis
    plt.xticks(range(len(treatment_labels)), treatment_labels, rotation=45, ha='right')
    plt.xlim(-0.5, len(treatment_labels) - 0.5)
    plt.ylim(0, 1.05)
    
    # Create legend for models (markers) in size order
    model_legend_elements = []
    sorted_models = sorted(model_marker_map.keys(), key=get_model_size_order)
    for model_name in sorted_models:
        marker = model_marker_map[model_name]
        model_legend_elements.append(plt.Line2D([0], [0], 
                                              marker=marker, 
                                              color='black', 
                                              linestyle='None',
                                              markersize=8,
                                              label=model_name))
    
    # Create legend for data types (colors)
    data_type_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='#d62728', linestyle='None', markersize=8, label='AT'),
        plt.Line2D([0], [0], marker='o', color='#1f77b4', linestyle='None', markersize=8, label='UT')
    ]
    
    # Position legends based on plot type
    if is_model_comparison:
        # For model comparison plots: position 3cm higher
        all_legend_elements = model_legend_elements + data_type_legend_elements
        if all_legend_elements:
            plt.legend(handles=all_legend_elements, 
                     title='Models & Data Types', 
                     loc='center left',
                     bbox_to_anchor=(0.98, 0.8))
    else:
        # For caps/typo plots: position 3cm lower
        all_legend_elements = model_legend_elements + data_type_legend_elements
        if all_legend_elements:
            plt.legend(handles=all_legend_elements, 
                     title='Models & Data Types', 
                     loc='center left',
                     bbox_to_anchor=(0.98, 0.2))
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created recognition accuracy plot: {output_path}")


def process_recognition_plots(input_dir: str, output_dir: str) -> None:
    """
    Process CSV files and create recognition accuracy plots.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load AT and UT data
    at_model_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_AT_model_comps.csv')
    ut_model_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_UT_model_comps.csv')
    at_caps_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_AT_caps_typo.csv')
    ut_caps_file = os.path.join(input_dir, 'detailed_accuracy_pivot_table_UT_caps_typo.csv')
    
    # Process model comparison plots
    if os.path.exists(at_model_file) and os.path.exists(ut_model_file):
        try:
            at_df = pd.read_csv(at_model_file, index_col=0)
            ut_df = pd.read_csv(ut_model_file, index_col=0)
            
            print(f"📊 Loaded model comparison data: AT {at_df.shape}, UT {ut_df.shape}")
            
            create_recognition_plot(at_df, ut_df, 
                                  "Recognition Accuracy - Model Comparisons", 
                                  os.path.join(output_dir, "recognition_accuracy_model_comps.png"),
                                  is_model_comparison=True)
            
        except Exception as e:
            print(f"❌ Error processing model comparison files: {e}")
    
    # Process caps/typo plots
    if os.path.exists(at_caps_file) and os.path.exists(ut_caps_file):
        try:
            at_df = pd.read_csv(at_caps_file, index_col=0)
            ut_df = pd.read_csv(ut_caps_file, index_col=0)
            
            print(f"📊 Loaded caps/typo data: AT {at_df.shape}, UT {ut_df.shape}")
            
            create_recognition_plot(at_df, ut_df, 
                                  "Recognition Accuracy - Caps & Typos", 
                                  os.path.join(output_dir, "recognition_accuracy_caps_typo.png"),
                                  is_model_comparison=False)
            
        except Exception as e:
            print(f"❌ Error processing caps/typo files: {e}")
    
    print(f"\n🎉 Recognition accuracy plots saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create recognition accuracy plots from analyze_results_2 CSV files')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2',
                       help='Directory containing CSV files (default: results_and_data/analysis/analyze_results_2)')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/recognition_plots',
                       help='Directory to save plots (default: results_and_data/analysis/recognition_plots)')
    
    args = parser.parse_args()
    
    print("🎯 Creating Recognition Accuracy Plots")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process recognition plots
    process_recognition_plots(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
