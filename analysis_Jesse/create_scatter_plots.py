#!/usr/bin/env python3
"""
Create scatter plots from analyze_results_2 CSV files.

This script reads the detailed pivot table CSV files and creates scatter plots
with preference accuracy on y-axis and recognition accuracy on x-axis.
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
        if col.startswith(('pref_', 'rec_')):
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


def get_treatment_colors(treatments: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Assign colors to treatment types with specific color schemes.
    Colors are aligned with model size (lighter for smaller models).
    
    Args:
        treatments: List of treatment names
        
    Returns:
        Tuple of (treatment_colors, cleaned_treatment_names) dictionaries
    """
    # Define color palettes optimized for actual variation counts with maximum contrast
    # Lighter colors for smaller models within each company
    caps_colors = ['#87CEEB', '#000080']  # Light sky blue (S2) vs Dark navy (S4)
    typo_colors = ['#FFB6C1', '#8B0000']  # Light pink (S2) vs Dark red (S4)
    claude_colors = ['#FFE4B5', '#FF4500']  # Light peach (haiku) vs Dark orange (sonnet)
    gpt_colors = ['#90EE90', '#ADFF2F', '#006400']  # Light green (4o-mini), Yellow-green (4.1-mini), Dark green (4.1)
    gemini_colors = ['#ADD8E6', '#0000CD']  # Light blue (flash) vs Medium blue (pro)
    
    treatment_colors = {}
    cleaned_treatment_names = {}
    
    # Sort treatments by model size for consistent color assignment
    model_comparison_treatments = [t for t in treatments if 'model_comparison' in t]
    sorted_model_treatments = sorted(model_comparison_treatments, key=lambda x: get_model_size_order(x))
    
    # Process caps and typos first (keep existing order)
    color_index = 0
    for treatment in treatments:
        if 'capitalization' in treatment:
            # Use different shades of blue for capitalization treatments
            treatment_colors[treatment] = caps_colors[color_index % len(caps_colors)]
            # Clean the name based on S2/S4
            if 'S2' in treatment:
                cleaned_treatment_names[treatment] = "cApItALizaTiOn"
            elif 'S4' in treatment:
                cleaned_treatment_names[treatment] = "CAPITALIZATION"
            else:
                cleaned_treatment_names[treatment] = treatment
            color_index += 1
        elif 'typo' in treatment:
            # Use different shades of red for typo treatments
            treatment_colors[treatment] = typo_colors[color_index % len(typo_colors)]
            # Clean the name based on S2/S4
            if 'S2' in treatment:
                cleaned_treatment_names[treatment] = "Few Typos (Ex@mple)"
            elif 'S4' in treatment:
                cleaned_treatment_names[treatment] = "Many Typos (Exx@mpl)"
            else:
                cleaned_treatment_names[treatment] = treatment
            color_index += 1
    
    # Process model comparisons in size order
    claude_index = 0
    gemini_index = 0
    gpt_index = 0
    
    for treatment in sorted_model_treatments:
        if 'claude' in treatment.lower():
            treatment_colors[treatment] = claude_colors[claude_index % len(claude_colors)]
            # Clean the name: remove model_comparison_anthropic_ and date suffix
            cleaned_name = treatment.replace('model_comparison_anthropic_', '').replace('model_comparison_', '')
            cleaned_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', cleaned_name)
            cleaned_treatment_names[treatment] = cleaned_name
            claude_index += 1
        elif 'gpt' in treatment.lower():
            treatment_colors[treatment] = gpt_colors[gpt_index % len(gpt_colors)]
            # Clean the name: remove model_comparison_ and date suffix
            cleaned_name = treatment.replace('model_comparison_', '')
            cleaned_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', cleaned_name)
            cleaned_treatment_names[treatment] = cleaned_name
            gpt_index += 1
        elif 'gemini' in treatment.lower():
            treatment_colors[treatment] = gemini_colors[gemini_index % len(gemini_colors)]
            # Clean the name: remove model_comparison_google_ and date suffix
            cleaned_name = treatment.replace('model_comparison_google_', '').replace('model_comparison_', '')
            cleaned_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', cleaned_name)
            cleaned_treatment_names[treatment] = cleaned_name
            gemini_index += 1
        else:
            # Default color for unknown model comparisons
            treatment_colors[treatment] = '#6c757d'  # Gray
            cleaned_treatment_names[treatment] = treatment
    
    return treatment_colors, cleaned_treatment_names


def create_scatter_plot(df: pd.DataFrame, title: str, output_path: str) -> None:
    """
    Create a scatter plot from the DataFrame.
    
    Args:
        df: DataFrame with pref/rec columns
        title: Plot title
        output_path: Path to save the plot
    """
    # Separate pref and rec columns
    pref_cols = [col for col in df.columns if col.startswith('pref_')]
    rec_cols = [col for col in df.columns if col.startswith('rec_')]
    
    if not pref_cols or not rec_cols:
        print(f"Warning: No pref/rec columns found in {title}")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get model marker mapping
    model_marker_map = get_model_marker_map(df.columns)
    
    # Get treatment colors and cleaned names
    treatment_colors, cleaned_treatment_names = get_treatment_colors(df.index.tolist())
    
    # Plot each treatment
    for treatment in df.index:
        for pref_col in pref_cols:
            rec_col = pref_col.replace('pref_', 'rec_')
            
            if rec_col not in df.columns:
                continue
            
            pref_val = df.loc[treatment, pref_col]
            rec_val = df.loc[treatment, rec_col]
            
            # Skip NaN values
            if pd.isna(pref_val) or pd.isna(rec_val):
                continue
            
            # Get model name and marker
            model_name = extract_model_name(pref_col)
            marker = model_marker_map.get(model_name, 'o')
            
            # Get treatment color
            color = treatment_colors.get(treatment, '#d62728')
            
            # Plot point
            plt.scatter(rec_val, pref_val, 
                       marker=marker, 
                       color=color, 
                       s=100, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=0.5)
    
    # Customize plot
    plt.xlabel('Recognition Accuracy', fontsize=12)
    plt.ylabel('Preference Accuracy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, 1.05)
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
    
    # Create legend for treatments (colors)
    treatment_legend_elements = []
    for treatment, color in treatment_colors.items():
        cleaned_name = cleaned_treatment_names.get(treatment, treatment)
        treatment_legend_elements.append(plt.Line2D([0], [0], 
                                                  marker='o', 
                                                  color=color, 
                                                  linestyle='None',
                                                  markersize=8,
                                                  label=cleaned_name))
    
    # Add legends
    if model_legend_elements:
        legend1 = plt.legend(handles=model_legend_elements, 
                           title='Models', 
                           loc='upper left',
                           bbox_to_anchor=(1.02, 1))
        plt.gca().add_artist(legend1)
    
    if treatment_legend_elements:
        plt.legend(handles=treatment_legend_elements, 
                 title='Treatments', 
                 loc='lower left',
                 bbox_to_anchor=(1.02, 0))
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created scatter plot: {output_path}")


def process_csv_files(input_dir: str, output_dir: str) -> None:
    """
    Process all CSV files and create scatter plots.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save scatter plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all detailed pivot table CSV files
    csv_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv') and 'detailed_' in file and 'pivot_table' in file:
            csv_files.append(file)
    
    if not csv_files:
        print("No detailed pivot table CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in sorted(csv_files):
        print(f"  - {file}")
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(input_dir, csv_file)
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path, index_col=0)
            
            # Create simplified title and filename
            if 'accuracy' in csv_file:
                data_type = "accuracy"
                data_title = "Accuracy"
            elif 'choice_1' in csv_file:
                data_type = "choice1"
                data_title = "Choice 1"
            else:
                data_type = "unknown"
                data_title = "Data"
            
            if 'AT_caps_typo' in csv_file:
                title = f"Assistant Tags - Caps & Typos ({data_title})"
                output_filename = f"AT_caps_typo_{data_type}.png"
            elif 'AT_model_comps' in csv_file:
                title = f"Assistant Tags - Model Comparisons ({data_title})"
                output_filename = f"AT_model_comps_{data_type}.png"
            elif 'UT_caps_typo' in csv_file:
                title = f"User Tags - Caps & Typos ({data_title})"
                output_filename = f"UT_caps_typo_{data_type}.png"
            elif 'UT_model_comps' in csv_file:
                title = f"User Tags - Model Comparisons ({data_title})"
                output_filename = f"UT_model_comps_{data_type}.png"
            else:
                # Fallback for unexpected filenames
                title = csv_file.replace('.csv', '').replace('_', ' ').title()
                output_filename = csv_file.replace('.csv', '.png')
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Create scatter plot
            create_scatter_plot(df, title, output_path)
            
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            continue
    
    print(f"\n🎉 Scatter plots saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create scatter plots from analyze_results_2 CSV files')
    parser.add_argument('--input-dir', 
                       default='results_and_data/analysis/analyze_results_2',
                       help='Directory containing CSV files (default: results_and_data/analysis/analyze_results_2)')
    parser.add_argument('--output-dir', 
                       default='results_and_data/analysis/scatter_plots',
                       help='Directory to save scatter plots (default: results_and_data/analysis/scatter_plots)')
    
    args = parser.parse_args()
    
    print("🎯 Creating Scatter Plots from analyze_results_2 CSV Files")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process CSV files
    process_csv_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
