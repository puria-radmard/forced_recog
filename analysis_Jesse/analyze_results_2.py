#!/usr/bin/env python3
"""
Results Aggregation Script (analyze_results_2.py)

This script aggregates results from multiple analyze_results_1 subdirectories to look at trends
across different dataset designs and evaluations. It combines CSV files from different experiments
with prefixed column headers to distinguish between different conditions.

USAGE:
    python analysis_Jesse/analyze_results_2.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def find_analyze_results_1_dirs(input_dir: str) -> List[str]:
    """
    Find all subdirectories in the analyze_results_1 directory.
    
    Args:
        input_dir: Directory containing analyze_results_1 subdirs
        
    Returns:
        List of subdirectory names (dataset_design_evaluation format)
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
    
    print(f"📁 Found {len(subdirs)} analyze_results_1 subdirectories:")
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
        print(f"  ⚠️  Warning: Could not load {csv_path}: {e}")
        return None


def clean_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column headers by removing dataset name and keeping only pref/rec + model name.
    
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
        
        # Look for 'pref' or 'rec' in the parts to identify the correct suffix
        pref_rec_parts = []
        for i, part in enumerate(parts):
            if part in ['pref', 'rec']:
                # Found pref or rec, now find the model name
                # Skip company names (anthropic, google) and look for model names
                model_parts = []
                for j in range(i + 1, len(parts)):
                    next_part = parts[j]
                    # Stop at common suffixes that indicate end of model name
                    if next_part in ['accuracy', 'choice', 'pct', 'total', 'correct']:
                        break
                    # Skip company names
                    if next_part in ['anthropic', 'google', 'openai']:
                        continue
                    model_parts.append(next_part)
                
                # Take pref/rec + model name
                if model_parts:
                    pref_rec_parts = [part] + model_parts
                else:
                    pref_rec_parts = [part]
                break
        
        if pref_rec_parts:
            new_col = '_'.join(pref_rec_parts)
        else:
            # Fallback: take the last two parts if no pref/rec found
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
        Dictionary with split DataFrames (e.g., 'UT_caps_typo', 'AT_model_comps')
    """
    # First split by tag type (UT/AT)
    ut_cols = []
    at_cols = []
    
    for col in df.columns:
        # Find which subdirectory this column belongs to
        for subdir in subdirs:
            if col.startswith(f"{subdir}_"):
                if 'UT' in subdir:
                    ut_cols.append(col)
                elif 'AT' in subdir:
                    at_cols.append(col)
                break
    
    result = {}
    
    # Process UT columns
    if ut_cols:
        ut_df = df[ut_cols].copy()
        ut_df = clean_column_headers(ut_df)
        
        # Split UT by treatment type (rows)
        ut_caps_typo_rows = []
        ut_model_comps_rows = []
        
        for idx in ut_df.index:
            # Check if this row is a model comparison treatment
            if 'model_comparison' in idx:
                ut_model_comps_rows.append(idx)
            else:
                ut_caps_typo_rows.append(idx)
        
        if ut_caps_typo_rows:
            ut_caps_typo_df = ut_df.loc[ut_caps_typo_rows].copy()
            result['UT_caps_typo'] = ut_caps_typo_df
        
        if ut_model_comps_rows:
            ut_model_comps_df = ut_df.loc[ut_model_comps_rows].copy()
            result['UT_model_comps'] = ut_model_comps_df
    
    # Process AT columns
    if at_cols:
        at_df = df[at_cols].copy()
        at_df = clean_column_headers(at_df)
        
        # Split AT by treatment type (rows)
        at_caps_typo_rows = []
        at_model_comps_rows = []
        
        for idx in at_df.index:
            # Check if this row is a model comparison treatment
            if 'model_comparison' in idx:
                at_model_comps_rows.append(idx)
            else:
                at_caps_typo_rows.append(idx)
        
        if at_caps_typo_rows:
            at_caps_typo_df = at_df.loc[at_caps_typo_rows].copy()
            result['AT_caps_typo'] = at_caps_typo_df
        
        if at_model_comps_rows:
            at_model_comps_df = at_df.loc[at_model_comps_rows].copy()
            result['AT_model_comps'] = at_model_comps_df
    
    return result


def aggregate_detailed_pivot_tables(input_dir: str, subdirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate detailed pivot tables from all subdirectories and split by tag type.
    
    Args:
        input_dir: Directory containing analyze_results_1 subdirs
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary with aggregated DataFrames split by tag type
    """
    print(f"\n📊 Aggregating detailed pivot tables...")
    
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
                print(f"    ✅ Loaded {subdir}: {df.shape[0]} rows × {df.shape[1]} cols")
            else:
                print(f"    ❌ Missing: {subdir}")
        
        if all_dfs:
            # Concatenate horizontally (along columns)
            try:
                aggregated_df = pd.concat(all_dfs, axis=1, sort=False)
                print(f"    📈 Aggregated: {aggregated_df.shape[0]} rows × {aggregated_df.shape[1]} cols")
                
                # Split by tag type and treatment type
                tag_treatment_split = split_by_tag_type_and_treatment(aggregated_df, subdirs)
                
                # Create separate files for each combination
                base_name = csv_file.replace('.csv', '')
                for split_name, split_df in tag_treatment_split.items():
                    if not split_df.empty:
                        key_name = f"{base_name}_{split_name}"
                        aggregated_data[key_name] = split_df
                        print(f"    📊 Created {split_name} table: {split_df.shape[0]} rows × {split_df.shape[1]} cols")
                
            except Exception as e:
                print(f"    ❌ Error concatenating {csv_file}: {e}")
        else:
            print(f"    ⚠️  No data found for {csv_file}")
    
    return aggregated_data


def aggregate_experiment_data(input_dir: str, subdirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate experiment-level data from all subdirectories.
    
    Args:
        input_dir: Directory containing analyze_results_1 subdirs
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary with aggregated DataFrames
    """
    print(f"\n📊 Aggregating experiment data...")
    
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
                print(f"    ✅ Loaded {subdir}: {df.shape[0]} rows × {df.shape[1]} cols")
            else:
                print(f"    ❌ Missing: {subdir}")
        
        if all_dfs:
            # Concatenate horizontally (along columns)
            try:
                aggregated_df = pd.concat(all_dfs, axis=1, sort=False)
                aggregated_data[csv_file.replace('.csv', '')] = aggregated_df
                print(f"    📈 Aggregated: {aggregated_df.shape[0]} rows × {aggregated_df.shape[1]} cols")
            except Exception as e:
                print(f"    ❌ Error concatenating {csv_file}: {e}")
        else:
            print(f"    ⚠️  No data found for {csv_file}")
    
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
    print(f"\n📊 Creating summary statistics...")
    
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
            print(f"    ⚠️  No experiment data found for {subdir}")
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
    print(f"\n💾 Saving aggregated results...")
    
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
            print(f"  ✅ Saved {key}: {df.shape[0]} rows × {df.shape[1]} cols")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, "summary_statistics.csv")
        summary_stats.to_csv(summary_path, index=False)
        print(f"  ✅ Saved summary statistics: {len(summary_stats)} experiments")
        
    except Exception as e:
        print(f"  ❌ Error saving results: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Aggregate Results from analyze_results_1 (analyze_results_2.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", 
                       default="results_and_data/analysis/analyze_results_1",
                       help="Directory containing analyze_results_1 subdirectories")
    
    parser.add_argument("--output-dir",
                       default="results_and_data/analysis/analyze_results_2",
                       help="Directory to save aggregated results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 RESULTS AGGREGATION (analyze_results_2.py)")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Find all analyze_results_1 subdirectories
        subdirs = find_analyze_results_1_dirs(args.input_dir)
        
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
        print("✅ AGGREGATION COMPLETE!")
        print("="*80)
        print(f"📊 Aggregated data saved to: {args.output_dir}")
        print(f"📈 Summary statistics: {len(summary_stats)} experiments")
        print(f"📋 CSV files: {args.output_dir}/*.csv")
        
    except Exception as e:
        print(f"\n❌ Error during aggregation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
