#!/usr/bin/env python3
"""
Combined Results Aggregation Script (analyze_results_2_combined.py)

This script aggregates results from the analyze_results_1_combined directory, which contains
both IR and 2T experiments. It parses subdirectory names to extract:
- Tag type: AT (Assistant Tag) or UT (User Tag)
- Experiment type: IR (Injection Recognition) or 2T (2-Turn)
- Paradigm: pref (Preference) or rec (Recognition)
- Priming: Pr (Primed) or NPr (Not Primed)

USAGE:
    python analysis_Jesse/analyze_results_2_combined.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def parse_subdir_name(subdir: str) -> Dict[str, str]:
    """
    Parse subdirectory name to extract experiment characteristics.
    
    Expected format: WikiSum_<TAG>_<EXP>_<PARADIGM>_<PRIMING>
    or: WikiSum-<TAG>_<EXP>_<PARADIGM>_<PRIMING>
    
    Args:
        subdir: Subdirectory name
        
    Returns:
        Dictionary with parsed components
    """
    # Replace hyphens with underscores for consistent splitting
    normalized = subdir.replace('-', '_')
    parts = normalized.split('_')
    
    parsed = {
        'tag_type': None,      # AT or UT
        'exp_type': None,      # IR or 2T
        'paradigm': None,      # pref or rec
        'priming': None,       # Pr or NPr
        'original': subdir
    }
    
    # Look for tag type (AT or UT)
    for i, part in enumerate(parts):
        if part in ['AT', 'UT']:
            parsed['tag_type'] = part
            # Check next part for experiment type
            if i + 1 < len(parts):
                if parts[i + 1] in ['IR', '2T']:
                    parsed['exp_type'] = parts[i + 1]
            break
    
    # Look for paradigm (pref or rec)
    for part in parts:
        if part in ['pref', 'rec']:
            parsed['paradigm'] = part
            break
    
    # Look for priming (Pr or NPr)
    for part in parts:
        if part in ['Pr', 'NPr']:
            parsed['priming'] = part
            break
    
    return parsed


def find_analyze_results_1_combined_dirs(input_dir: str) -> Tuple[List[str], List[Dict]]:
    """
    Find all subdirectories in the analyze_results_1_combined directory and parse their names.
    
    Args:
        input_dir: Directory containing analyze_results_1_combined subdirs
        
    Returns:
        Tuple of (list of subdirectory names, list of parsed metadata)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    subdirs = []
    parsed_info = []
    
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
            parsed_info.append(parse_subdir_name(item))
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {input_dir}")
    
    print(f"Found {len(subdirs)} analyze_results_1_combined subdirectories:")
    for i, subdir in enumerate(sorted(subdirs)):
        info = parsed_info[sorted(subdirs).index(subdir)]
        print(f"  - {subdir}")
        print(f"    Tag: {info['tag_type']}, Exp: {info['exp_type']}, Paradigm: {info['paradigm']}, Priming: {info['priming']}")
    
    return sorted(subdirs), parsed_info


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
        
        # Extract the model name
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


def categorize_treatments(df: pd.DataFrame, parsed_info_list: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Categorize treatments based on experiment type (IR vs 2T).
    
    Args:
        df: Aggregated DataFrame with prefixed columns
        parsed_info_list: List of parsed subdirectory metadata
        
    Returns:
        Dictionary with categorized DataFrames
    """
    result = {}
    
    # Get all unique tag types and experiment types
    tag_types = set(info['tag_type'] for info in parsed_info_list if info['tag_type'])
    exp_types = set(info['exp_type'] for info in parsed_info_list if info['exp_type'])
    
    for tag in sorted(tag_types):
        for exp in sorted(exp_types):
            # Find columns that belong to this tag and experiment type
            matching_cols = []
            
            for col in df.columns:
                for info in parsed_info_list:
                    if info['tag_type'] == tag and info['exp_type'] == exp:
                        if col.startswith(f"{info['original']}_"):
                            matching_cols.append(col)
                            break
            
            if matching_cols:
                subset_df = df[matching_cols].copy()
                subset_df = clean_column_headers(subset_df)
                
                # Further split by treatment category
                # Both IR and 2T now have model_comps (IR has inferred model_other)
                caps_typo_rows = []
                model_comps_rows = []
                
                for idx in subset_df.index:
                    if 'model_comparison' in idx:
                        model_comps_rows.append(idx)
                    else:
                        caps_typo_rows.append(idx)
                
                if caps_typo_rows:
                    result[f"{tag}_{exp}_caps_typo"] = subset_df.loc[caps_typo_rows]
                if model_comps_rows:
                    result[f"{tag}_{exp}_model_comps"] = subset_df.loc[model_comps_rows]
    
    return result


def aggregate_detailed_pivot_tables(input_dir: str, subdirs: List[str], 
                                   parsed_info_list: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate detailed pivot tables from all subdirectories and categorize by experiment type.
    
    Args:
        input_dir: Directory containing analyze_results_1_combined subdirs
        subdirs: List of subdirectory names
        parsed_info_list: List of parsed subdirectory metadata
        
    Returns:
        Dictionary with aggregated DataFrames categorized by experiment type
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
                
                # Categorize by tag type and experiment type
                categorized = categorize_treatments(aggregated_df, parsed_info_list)
                
                # Create separate files for each category
                base_name = csv_file.replace('.csv', '')
                for category_name, category_df in categorized.items():
                    if not category_df.empty:
                        key_name = f"{base_name}_{category_name}"
                        aggregated_data[key_name] = category_df
                        print(f"    [CREATED] Created {category_name} table: {category_df.shape[0]} rows x {category_df.shape[1]} cols")
                
            except Exception as e:
                print(f"    [ERROR] Error concatenating {csv_file}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    [WARNING] No data found for {csv_file}")
    
    return aggregated_data


def aggregate_experiment_data(input_dir: str, subdirs: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate experiment-level data from all subdirectories.
    
    Args:
        input_dir: Directory containing analyze_results_1_combined subdirs
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary with aggregated DataFrames
    """
    print(f"\nAggregating experiment data...")
    
    aggregated_data = {}
    
    # Files to aggregate
    csv_files = [
        'accuracy_by_experiment.csv'
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


def create_summary_statistics(aggregated_data: Dict[str, pd.DataFrame], 
                             subdirs: List[str],
                             parsed_info_list: List[Dict]) -> pd.DataFrame:
    """
    Create summary statistics across all experiments.
    
    Args:
        aggregated_data: Dictionary of aggregated DataFrames
        subdirs: List of subdirectory names
        parsed_info_list: List of parsed subdirectory metadata
        
    Returns:
        Summary statistics DataFrame
    """
    print(f"\nCreating summary statistics...")
    
    summary_stats = []
    
    # Analyze each subdirectory
    for i, subdir in enumerate(subdirs):
        print(f"  Analyzing {subdir}...")
        
        info = next((p for p in parsed_info_list if p['original'] == subdir), None)
        
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
                'tag_type': info['tag_type'] if info else None,
                'exp_type': info['exp_type'] if info else None,
                'paradigm': info['paradigm'] if info else None,
                'priming': info['priming'] if info else None,
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
                'tag_type': info['tag_type'] if info else None,
                'exp_type': info['exp_type'] if info else None,
                'paradigm': info['paradigm'] if info else None,
                'priming': info['priming'] if info else None,
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
            # For detailed pivot tables, use the categorized naming
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
        description="Aggregate Combined Results from analyze_results_1_combined (analyze_results_2_combined.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--input-dir", 
                       default="results_and_data/analysis/analyze_results_1_combined",
                       help="Directory containing analyze_results_1_combined subdirectories")
    
    parser.add_argument("--output-dir",
                       default="results_and_data/analysis/analyze_results_2_combined",
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
    print("COMBINED RESULTS AGGREGATION (analyze_results_2_combined.py)")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Find all analyze_results_1_combined subdirectories
        subdirs, parsed_info_list = find_analyze_results_1_combined_dirs(args.input_dir)
        
        # Aggregate detailed pivot tables
        aggregated_pivot_data = aggregate_detailed_pivot_tables(args.input_dir, subdirs, parsed_info_list)
        
        # Aggregate experiment data
        aggregated_exp_data = aggregate_experiment_data(args.input_dir, subdirs)
        
        # Combine all aggregated data
        all_aggregated_data = {**aggregated_pivot_data, **aggregated_exp_data}
        
        # Create summary statistics
        summary_stats = create_summary_statistics(all_aggregated_data, subdirs, parsed_info_list)
        
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

