#!/usr/bin/env python3
"""
IR Results Aggregation Script (analyze_results_2_IR.py)

This script aggregates results from multiple analyze_results_1_IR subdirectories to look at trends
across different dataset designs and evaluations. It combines CSV files from different experiments
with prefixed column headers to distinguish between different conditions.

USAGE:
    python analysis_Jesse/analyze_results_2_IR.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def find_analyze_results_1_ir_dirs(input_dir: str) -> List[str]:
    """
    Find all subdirectories in the analyze_results_1_IR directory.
    
    Args:
        input_dir: Directory containing analyze_results_1_IR subdirs
        
    Returns:
        List of subdirectory names (dataset_design format)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    subdirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {input_dir}")
    
    print(f"Found {len(subdirs)} analyze_results_1_IR subdirectories:")
    for subdir in sorted(subdirs):
        print(f"  - {subdir}")
    
    return sorted(subdirs)


def load_csv_with_prefix(csv_path: str, prefix: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and prefix all column names with the given prefix.
    
    Args:
        csv_path: Path to the CSV file
        prefix: Prefix to add to column names
        
    Returns:
        DataFrame with prefixed column names, or None if file doesn't exist
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # Add prefix to all column names
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        return df
    except Exception as e:
        print(f"  [WARNING] Warning: Could not load {csv_path}: {e}")
        return None


def clean_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column headers by removing dataset name and keeping only model name.
    
    Args:
        df: DataFrame with prefixed column names
        
    Returns:
        DataFrame with cleaned column names
    """
    cleaned_df = df.copy()
    new_columns = []
    
    for col in df.columns:
        # Split by underscore to get parts
        parts = col.split('_')
        
        # For IR experiments, we want to extract the model name
        # Skip the dataset prefix (e.g., "WikiSum-AT_IR") and keep the model parts
        model_parts = []
        found_model = False
        
        for i, part in enumerate(parts):
            # Skip company names at the beginning
            if part in ['anthropic', 'google', 'openai'] and not found_model:
                found_model = True
                continue
            
            # Once we've found the model, collect model name parts
            if found_model or part.startswith('gpt') or part.startswith('claude') or part.startswith('gemini'):
                found_model = True
                # Stop at common suffixes that indicate end of model name
                if part in ['accuracy', 'choice', 'pct', 'total', 'correct']:
                    break
                # Skip company names in the middle
                if part in ['anthropic', 'google', 'openai']:
                    continue
                model_parts.append(part)
        
        if model_parts:
            new_col = '_'.join(model_parts)
        else:
            # Fallback: take the last two parts if no model found
            if len(parts) >= 2:
                new_col = '_'.join(parts[-2:])
            else:
                new_col = col
        
        new_columns.append(new_col)
    
    cleaned_df.columns = new_columns
    return cleaned_df


def split_by_tag_type_and_treatment(df: pd.DataFrame, subdirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Split aggregated DataFrame into UT/AT tables and further by treatment type (rows).
    
    Args:
        df: Aggregated DataFrame with prefixed columns
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary with split DataFrames (e.g., 'UT_caps_typo', 'AT_other_models')
    """
    # First split by tag type (UT/AT)
    ut_cols = []
    at_cols = []
    
    for col in df.columns:
        # Find which subdirectory this column belongs to
        for subdir in subdirs:
            if col.startswith(f"{subdir}_"):
                if 'UT' in subdir or 'UT_' in subdir:
                    ut_cols.append(col)
                elif 'AT' in subdir or 'AT_' in subdir:
                    at_cols.append(col)
                break
    
    result = {}
    
    # Process UT columns
    if ut_cols:
        ut_df = df[ut_cols].copy()
        ut_df = clean_column_headers(ut_df)
        
        # Split UT by treatment type (rows)
        ut_caps_typo_rows = []
        ut_other_models_rows = []
        
        for idx in ut_df.index:
            # Check if this row is control or other_model treatment
            if idx in ['control', 'other_model']:
                ut_other_models_rows.append(idx)
            else:
                # typo and capitalization treatments
                ut_caps_typo_rows.append(idx)
        
        if ut_caps_typo_rows:
            ut_caps_typo_df = ut_df.loc[ut_caps_typo_rows].copy()
            result['UT_caps_typo'] = ut_caps_typo_df
        
        if ut_other_models_rows:
            ut_other_models_df = ut_df.loc[ut_other_models_rows].copy()
            result['UT_other_models'] = ut_other_models_df
    
    # Process AT columns
    if at_cols:
        at_df = df[at_cols].copy()
        at_df = clean_column_headers(at_df)
        
        # Split AT by treatment type (rows)
        at_caps_typo_rows = []
        at_other_models_rows = []
        
        for idx in at_df.index:
            # Check if this row is control or other_model treatment
            if idx in ['control', 'other_model']:
                at_other_models_rows.append(idx)
            else:
                # typo and capitalization treatments
                at_caps_typo_rows.append(idx)
        
        if at_caps_typo_rows:
            at_caps_typo_df = at_df.loc[at_caps_typo_rows].copy()
            result['AT_caps_typo'] = at_caps_typo_df
        
        if at_other_models_rows:
            at_other_models_df = at_df.loc[at_other_models_rows].copy()
            result['AT_other_models'] = at_other_models_df
    
    return result


def aggregate_detailed_pivot_tables(input_dir: str, subdirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate detailed pivot tables from all subdirectories and split by tag type.
    
    Args:
        input_dir: Directory containing analyze_results_1_IR subdirs
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary with aggregated DataFrames split by tag type
    """
    print(f"\nAggregating detailed pivot tables...")
    
    aggregated_data = {}
    
    # Files to aggregate
    csv_files = [
        'detailed_accuracy_pivot_table.csv',
        'detailed_choice_1_pct_pivot_table.csv'
    ]
    
    for csv_file in csv_files:
        print(f"\n  Processing {csv_file}...")
        all_dfs = []
        
        for subdir in subdirs:
            csv_path = os.path.join(input_dir, subdir, csv_file)
            df = load_csv_with_prefix(csv_path, subdir)
            
            if df is not None:
                all_dfs.append(df)
                print(f"    [OK] Loaded {subdir}: {df.shape[0]} rows x {df.shape[1]} cols")
            else:
                print(f"    [MISSING] Missing: {subdir}")
        
        if all_dfs:
            # Concatenate horizontally (along columns)
            try:
                aggregated_df = pd.concat(all_dfs, axis=1, sort=False)
                print(f"    [AGGREGATED] Aggregated: {aggregated_df.shape[0]} rows x {aggregated_df.shape[1]} cols")
                
                # Split by tag type and treatment type
                tag_treatment_split = split_by_tag_type_and_treatment(aggregated_df, subdirs)
                
                # Create separate files for each combination
                base_name = csv_file.replace('.csv', '')
                for split_name, split_df in tag_treatment_split.items():
                    if not split_df.empty:
                        key_name = f"{base_name}_{split_name}"
                        aggregated_data[key_name] = split_df
                        print(f"    [CREATED] Created {split_name} table: {split_df.shape[0]} rows x {split_df.shape[1]} cols")
                
            except Exception as e:
                print(f"    [ERROR] Error concatenating {csv_file}: {e}")
        else:
            print(f"    [WARNING] No data found for {csv_file}")
    
    return aggregated_data


def aggregate_experiment_data(input_dir: str, subdirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate experiment-level data from all subdirectories.
    
    Args:
        input_dir: Directory containing analyze_results_1_IR subdirs
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary with aggregated DataFrames
    """
    print(f"\nAggregating experiment data...")
    
    aggregated_data = {}
    
    # Files to aggregate
    csv_files = [
        'accuracy_by_experiment.csv',
        'detailed_breakdown_data.csv'
    ]
    
    for csv_file in csv_files:
        print(f"\n  Processing {csv_file}...")
        all_dfs = []
        
        for subdir in subdirs:
            csv_path = os.path.join(input_dir, subdir, csv_file)
            df = load_csv_with_prefix(csv_path, subdir)
            
            if df is not None:
                all_dfs.append(df)
                print(f"    [OK] Loaded {subdir}: {df.shape[0]} rows x {df.shape[1]} cols")
            else:
                print(f"    [MISSING] Missing: {subdir}")
        
        if all_dfs:
            # Concatenate horizontally (along columns)
            try:
                aggregated_df = pd.concat(all_dfs, axis=1, sort=False)
                aggregated_data[csv_file.replace('.csv', '')] = aggregated_df
                print(f"    [AGGREGATED] Aggregated: {aggregated_df.shape[0]} rows x {aggregated_df.shape[1]} cols")
            except Exception as e:
                print(f"    [ERROR] Error concatenating {csv_file}: {e}")
        else:
            print(f"    [WARNING] No data found for {csv_file}")
    
    return aggregated_data


def create_summary_statistics(aggregated_data: Dict[str, pd.DataFrame], subdirs: List[str]) -> pd.DataFrame:
    """
    Create summary statistics across all experiments.
    
    Args:
        aggregated_data: Dictionary of aggregated DataFrames
        subdirs: List of subdirectory names
        
    Returns:
        Summary statistics DataFrame
    """
    print(f"\nCreating summary statistics...")
    
    summary_stats = []
    
    # Analyze each subdirectory
    for subdir in subdirs:
        print(f"  Analyzing {subdir}...")
        
        # Get experiment data for this subdirectory
        exp_data = None
        for key, df in aggregated_data.items():
            if 'accuracy_by_experiment' in key:
                # Find columns that belong to this subdirectory
                subdir_cols = [col for col in df.columns if col.startswith(f"{subdir}_")]
                if subdir_cols:
                    exp_data = df[subdir_cols]
                    break
        
        if exp_data is not None and not exp_data.empty:
            # Calculate summary statistics
            accuracy_cols = [col for col in exp_data.columns if 'accuracy' in col]
            choice_1_cols = [col for col in exp_data.columns if 'choice_1_pct' in col]
            
            if accuracy_cols:
                mean_accuracy = exp_data[accuracy_cols].mean().mean()
                std_accuracy = exp_data[accuracy_cols].std().mean()
            else:
                mean_accuracy = np.nan
                std_accuracy = np.nan
            
            if choice_1_cols:
                mean_choice_1 = exp_data[choice_1_cols].mean().mean()
                std_choice_1 = exp_data[choice_1_cols].std().mean()
            else:
                mean_choice_1 = np.nan
                std_choice_1 = np.nan
            
            summary_stats.append({
                'subdirectory': subdir,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'mean_choice_1_pct': mean_choice_1,
                'std_choice_1_pct': std_choice_1,
                'num_experiments': len(exp_data)
            })
        else:
            print(f"    [WARNING] No experiment data found for {subdir}")
            summary_stats.append({
                'subdirectory': subdir,
                'mean_accuracy': np.nan,
                'std_accuracy': np.nan,
                'mean_choice_1_pct': np.nan,
                'std_choice_1_pct': np.nan,
                'num_experiments': 0
            })
    
    summary_df = pd.DataFrame(summary_stats)
    return summary_df


def save_aggregated_results(aggregated_data: Dict[str, pd.DataFrame], 
                          summary_stats: pd.DataFrame, 
                          output_dir: str) -> None:
    """
    Save all aggregated results to CSV files.
    
    Args:
        aggregated_data: Dictionary of aggregated DataFrames
        summary_stats: Summary statistics DataFrame
        output_dir: Directory to save results
    """
    print(f"\nSaving aggregated results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save aggregated data
        for key, df in aggregated_data.items():
            # For detailed pivot tables, use the new naming convention
            if 'detailed_accuracy_pivot_table' in key or 'detailed_choice_1_pct_pivot_table' in key:
                output_path = os.path.join(output_dir, f"{key}.csv")
            else:
                output_path = os.path.join(output_dir, f"aggregated_{key}.csv")
            
            df.to_csv(output_path)
            print(f"  [OK] Saved {key}: {df.shape[0]} rows x {df.shape[1]} cols")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, "summary_statistics.csv")
        summary_stats.to_csv(summary_path, index=False)
        print(f"  [OK] Saved summary statistics: {len(summary_stats)} experiments")
        
    except Exception as e:
        print(f"  [ERROR] Error saving results: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Aggregate IR Results from analyze_results_1_IR (analyze_results_2_IR.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", 
                       default="results_and_data/analysis/analyze_results_1_IR",
                       help="Directory containing analyze_results_1_IR subdirectories")
    
    parser.add_argument("--output-dir",
                       default="results_and_data/analysis/analyze_results_2_IR",
                       help="Directory to save aggregated results")
    
    args = parser.parse_args()
    
    # Get the script's directory to construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Parent of analysis_Jesse/
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(args.input_dir):
        args.input_dir = os.path.join(project_root, args.input_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    print("="*80)
    print("IR RESULTS AGGREGATION (analyze_results_2_IR.py)")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Find all analyze_results_1_IR subdirectories
        subdirs = find_analyze_results_1_ir_dirs(args.input_dir)
        
        # Aggregate detailed pivot tables
        aggregated_pivot_data = aggregate_detailed_pivot_tables(args.input_dir, subdirs)
        
        # Aggregate experiment data
        aggregated_exp_data = aggregate_experiment_data(args.input_dir, subdirs)
        
        # Combine all aggregated data
        all_aggregated_data = {**aggregated_pivot_data, **aggregated_exp_data}
        
        # Create summary statistics
        summary_stats = create_summary_statistics(all_aggregated_data, subdirs)
        
        # Save all results
        save_aggregated_results(all_aggregated_data, summary_stats, args.output_dir)
        
        print("\n" + "="*80)
        print("AGGREGATION COMPLETE!")
        print("="*80)
        print(f"Aggregated data saved to: {args.output_dir}")
        print(f"Summary statistics: {len(summary_stats)} experiments")
        print(f"CSV files: {args.output_dir}/*.csv")
        
    except Exception as e:
        print(f"\n[ERROR] Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

