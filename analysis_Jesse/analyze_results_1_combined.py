#!/usr/bin/env python3
"""
Combined Results Analysis Script (analyze_results_1_combined.py)

This script processes all WikiSum- results directories and creates unified analysis outputs.
It handles both 2T and IR experiments:
- 2T experiments: Already have model_other column
- IR experiments: Infers model_other column based on the systematic 40-row cycle pattern

Outputs are saved to results_and_data/analysis/analyze_results_1_combined/ with the same
directory naming convention as the source data.

USAGE:
    python analysis_Jesse/analyze_results_1_combined.py --results-base-dir RESULTS_DIR
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import re
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Standard model cycle for IR experiment inference
ALL_MODELS = [
    'anthropic_claude-3-5-haiku-20241022',
    'anthropic_claude-sonnet-4-20250514',
    'google_gemini-2.5-flash',
    'google_gemini-2.5-pro',
    'gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14',
    'gpt-4o-mini'
]


def parse_dirname(dirname: str) -> Dict[str, str]:
    """
    Parse directory name to extract experiment characteristics.
    
    Expected format: WikiSum-<TAG>_<EXP>_<PARADIGM>_<PRIMING>
    
    Args:
        dirname: Directory name
        
    Returns:
        Dictionary with parsed components
    """
    # Replace hyphens after WikiSum with underscores for consistent splitting
    normalized = dirname.replace('WikiSum-', 'WikiSum_')
    parts = normalized.split('_')
    
    parsed = {
        'tag_type': None,      # AT or UT
        'exp_type': None,      # IR or 2T
        'paradigm': None,      # pref or rec
        'priming': None,       # Pr or NPr
        'original': dirname
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


def find_wikisum_dirs(results_base_dir: str) -> List[Tuple[str, Dict]]:
    """
    Find all WikiSum- directories in the results directory.
    
    Args:
        results_base_dir: Base results directory
        
    Returns:
        List of (directory_path, parsed_info) tuples
    """
    if not os.path.exists(results_base_dir):
        raise FileNotFoundError(f"Results directory not found: {results_base_dir}")
    
    wikisum_dirs = []
    
    for item in os.listdir(results_base_dir):
        if item.startswith('WikiSum-'):
            item_path = os.path.join(results_base_dir, item)
            if os.path.isdir(item_path):
                parsed = parse_dirname(item)
                wikisum_dirs.append((item_path, parsed))
    
    if not wikisum_dirs:
        raise ValueError(f"No WikiSum- directories found in {results_base_dir}")
    
    print(f"Found {len(wikisum_dirs)} WikiSum directories:")
    for dir_path, info in wikisum_dirs:
        dirname = os.path.basename(dir_path)
        print(f"  - {dirname}")
        print(f"    Tag: {info['tag_type']}, Exp: {info['exp_type']}, Paradigm: {info['paradigm']}, Priming: {info['priming']}")
    
    return wikisum_dirs


def infer_model_other_for_ir(df: pd.DataFrame, base_model: str) -> pd.DataFrame:
    """
    Infer the model_other column for IR experiments based on row position.
    
    IR experiments follow a pattern:
    - First 40 rows: control (treatment = "control")
    - Next 40 rows: other_model = first in cycle (excluding base)
    - Next 40 rows: other_model = second in cycle
    - ... and so on for 6 other models (240 rows total)
    
    Args:
        df: DataFrame with IR results
        base_model: The base model name
        
    Returns:
        DataFrame with added model_other column
    """
    df = df.copy()
    
    # Initialize model_other column
    df['model_other'] = None
    
    # Create the cycle of other models (excluding base model)
    other_models_cycle = [m for m in ALL_MODELS if m != base_model]
    
    print(f"    Base model: {base_model}")
    print(f"    Other models cycle ({len(other_models_cycle)} models): {other_models_cycle}")
    
    # Filter to other_model treatment rows
    other_model_mask = df['treatment'] == 'other_model'
    other_model_indices = df[other_model_mask].index.tolist()
    
    print(f"    Total other_model rows: {len(other_model_indices)}")
    print(f"    Expected: {len(other_models_cycle) * 40} rows (6 models × 40 rows)")
    
    # Assign model_other based on position in chunks of 40
    for i, idx in enumerate(other_model_indices):
        chunk_index = i // 40  # Which 40-row chunk (0-5)
        if chunk_index < len(other_models_cycle):
            df.loc[idx, 'model_other'] = other_models_cycle[chunk_index]
    
    # For control rows, set model_other to base_model
    control_mask = df['treatment'] == 'control'
    df.loc[control_mask, 'model_other'] = base_model
    
    # Verify the assignment
    model_other_counts = df['model_other'].value_counts()
    print(f"    Assigned model_other distribution:")
    for model, count in model_other_counts.items():
        print(f"      {model}: {count} rows")
    
    return df


def load_and_process_results(dir_path: str, exp_info: Dict) -> pd.DataFrame:
    """
    Load and process results from a WikiSum directory.
    
    Args:
        dir_path: Path to the WikiSum directory
        exp_info: Parsed experiment information
        
    Returns:
        Combined DataFrame with all results from this experiment
    """
    all_results = []
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('_choice_results.csv')]
    
    if not csv_files:
        print(f"  [WARNING] No CSV files found in {os.path.basename(dir_path)}")
        return pd.DataFrame()
    
    print(f"  Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        csv_path = os.path.join(dir_path, csv_file)
        try:
            df = pd.read_csv(csv_path)
            
            # Extract base model from filename or data
            if 'model_base' in df.columns:
                base_model = df['model_base'].iloc[0]
            else:
                # Try to extract from filename
                base_model = csv_file.replace('_choice_results.csv', '').replace('_vs_all_others_control_comparison', '')
            
            # Add experiment metadata
            df['experiment_file'] = csv_file.replace('_choice_results.csv', '')
            
            # If IR experiment and no model_other column, infer it
            if exp_info['exp_type'] == 'IR' and 'model_other' not in df.columns:
                print(f"    Processing IR experiment: {csv_file}")
                df = infer_model_other_for_ir(df, base_model)
            
            all_results.append(df)
            print(f"    [OK] Loaded {csv_file}: {len(df)} rows")
            
        except Exception as e:
            print(f"    [ERROR] Error loading {csv_file}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def organize_models_by_company_and_size(models: List[str]) -> List[str]:
    """
    Organize models by company and then by size within each company.
    
    Args:
        models: List of model names
        
    Returns:
        List of models organized by company and size
    """
    def get_company(model_name: str) -> str:
        """Extract company from model name."""
        if model_name.startswith('anthropic_') or 'claude' in model_name:
            return 'anthropic'
        elif model_name.startswith('google_') or 'gemini' in model_name:
            return 'google'
        elif model_name.startswith('gpt-') or 'gpt' in model_name:
            return 'openai'
        else:
            return 'other'
    
    def get_model_size(model_name: str) -> int:
        """Extract model size for ordering within company based on actual model capabilities."""
        clean_name = model_name.lower()
        
        # Anthropic models (ordered by capability: haiku < sonnet < opus)
        if 'claude' in clean_name:
            if 'haiku' in clean_name:
                return 1
            elif 'sonnet' in clean_name:
                return 2
            elif 'opus' in clean_name:
                return 3
            else:
                return 2
        
        # Google models (ordered by capability: flash < pro)
        elif 'gemini' in clean_name:
            if 'flash' in clean_name:
                return 1
            else:  # pro models
                return 2
        
        # OpenAI models (ordered by capability: 4o-mini < 4.1-mini < 4.1)
        elif 'gpt' in clean_name:
            if '4o-mini' in clean_name:
                return 1
            elif '4.1-mini' in clean_name:
                return 2
            elif '4.1' in clean_name:
                return 3
            else:
                return 999
        
        return 999
    
    # Group by company
    company_groups = defaultdict(list)
    for model in models:
        company = get_company(model)
        company_groups[company].append(model)
    
    # Sort within each company by size, then alphabetically
    organized_models = []
    for company in ['anthropic', 'google', 'openai', 'other']:
        if company in company_groups:
            company_models = company_groups[company]
            company_models.sort(key=lambda x: (get_model_size(x), x))
            organized_models.extend(company_models)
    
    return organized_models


def clean_model_label(model_name: str) -> str:
    """Clean model name by removing prefixes and suffixes."""
    # Remove company prefixes
    if model_name.startswith('anthropic_'):
        model_name = model_name.replace('anthropic_', '')
    elif model_name.startswith('google_'):
        model_name = model_name.replace('google_', '')
    
    # Remove date suffixes
    model_name = re.sub(r'-\d{8}$', '', model_name)
    model_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_name)
    
    return model_name


def clean_treatment_label(treatment_name: str) -> str:
    """Clean treatment name for display."""
    # Remove model_comparison_ prefix
    if treatment_name.startswith('model_comparison_'):
        treatment_name = treatment_name.replace('model_comparison_', '')
    
    # Remove company prefixes
    treatment_name = treatment_name.replace('anthropic_', '')
    treatment_name = treatment_name.replace('google_', '')
    
    # Remove date suffixes
    treatment_name = re.sub(r'-\d{8}$', '', treatment_name)
    treatment_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', treatment_name)
    
    return treatment_name


def analyze_accuracy_by_conditions(df: pd.DataFrame, exp_info: Dict) -> Dict:
    """
    Analyze accuracy across different conditions.
    
    Args:
        df: Combined results DataFrame
        exp_info: Parsed experiment information
        
    Returns:
        Dictionary with analysis results
    """
    print("\n  Analyzing accuracy by conditions...")
    
    analysis = {}
    
    # 1. Accuracy by experiment file
    experiment_accuracy = df.groupby('experiment_file').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    experiment_accuracy.columns = ['total_trials', 'correct_trials', 'accuracy']
    
    # Calculate choice 1 percentage
    choice_1_pct = df.groupby('experiment_file')['selected_choice'].apply(lambda x: (x == 1).sum() / len(x) * 100)
    experiment_accuracy['choice_1_pct'] = choice_1_pct.round(1)
    experiment_accuracy = experiment_accuracy.sort_values('accuracy', ascending=False)
    
    analysis['by_experiment'] = experiment_accuracy
    
    # 2. Create detailed pivot table with model_other breakdown
    print("  Creating detailed breakdown by model and treatment...")
    
    # For 2T experiments or IR with inferred model_other
    if 'model_other' in df.columns:
        # Parse experiment files to get base model and treatment info
        detailed_analysis = []
        
        for exp_file in df['experiment_file'].unique():
            exp_data = df[df['experiment_file'] == exp_file]
            
            # Get base model
            if 'model_base' in exp_data.columns:
                base_model = exp_data['model_base'].iloc[0]
            else:
                base_model = exp_file.split('_')[0]
            
            # Determine treatment type from filename
            if 'vs_all_others' in exp_file:
                # Break down by model_other
                for other_model in exp_data['model_other'].unique():
                    if pd.isna(other_model):
                        continue
                    
                    model_data = exp_data[exp_data['model_other'] == other_model]
                    
                    # Check if self-comparison
                    is_self_comparison = (base_model == other_model)
                    
                    if is_self_comparison:
                        accuracy = np.nan
                        choice_1_pct = np.nan
                    else:
                        accuracy = model_data['is_correct'].mean()
                        choice_1_pct = (model_data['selected_choice'] == 1).mean() * 100
                    
                    detailed_analysis.append({
                        'experiment_file': exp_file,
                        'base_model': base_model,
                        'treatment': f"model_comparison_{other_model}",
                        'other_model': other_model,
                        'accuracy': accuracy,
                        'choice_1_pct': choice_1_pct,
                        'total_trials': len(model_data),
                        'is_self_comparison': is_self_comparison
                    })
            elif 'control_vs_' in exp_file:
                # Simple treatment comparison (typo, capitalization)
                if 'typo' in exp_file:
                    treatment_type = 'typo_S4' if 'S4' in exp_file else 'typo_S2'
                elif 'capitalization' in exp_file:
                    treatment_type = 'capitalization_S4' if 'S4' in exp_file else 'capitalization_S2'
                else:
                    treatment_type = 'unknown'
                
                accuracy = exp_data['is_correct'].mean()
                choice_1_pct = (exp_data['selected_choice'] == 1).mean() * 100
                
                detailed_analysis.append({
                    'experiment_file': exp_file,
                    'base_model': base_model,
                    'treatment': treatment_type,
                    'other_model': None,
                    'accuracy': accuracy,
                    'choice_1_pct': choice_1_pct,
                    'total_trials': len(exp_data)
                })
        
        # Create detailed dataframe
        detailed_df = pd.DataFrame(detailed_analysis)
        
        if not detailed_df.empty:
            # Get organized model order
            models = detailed_df['base_model'].unique().tolist()
            organized_models = organize_models_by_company_and_size(models)
            
            # Create accuracy pivot table
            detailed_accuracy_pivot = detailed_df.pivot_table(
                index='treatment',
                columns='base_model',
                values='accuracy',
                fill_value=None
            ).round(3)
            
            # Reorder columns
            detailed_accuracy_pivot = detailed_accuracy_pivot[organized_models]
            
            # Create choice 1 pct pivot table
            detailed_choice_1_pct_pivot = detailed_df.pivot_table(
                index='treatment',
                columns='base_model',
                values='choice_1_pct',
                fill_value=None
            ) / 100
            detailed_choice_1_pct_pivot = detailed_choice_1_pct_pivot.round(3)
            
            # Reorder columns
            detailed_choice_1_pct_pivot = detailed_choice_1_pct_pivot[organized_models]
            
            # Organize row order
            all_treatments = detailed_accuracy_pivot.index.tolist()
            model_comparison_treatments = [t for t in all_treatments if t.startswith('model_comparison_')]
            other_treatments = [t for t in all_treatments if not t.startswith('model_comparison_')]
            
            # Sort model_comparison by model
            if model_comparison_treatments:
                model_comparison_models = [t.replace('model_comparison_', '') for t in model_comparison_treatments]
                model_comparison_organized = organize_models_by_company_and_size(model_comparison_models)
                model_comparison_organized_treatments = [f"model_comparison_{m}" for m in model_comparison_organized]
            else:
                model_comparison_organized_treatments = []
            
            # Sort other treatments
            other_treatments.sort()
            
            organized_treatments = other_treatments + model_comparison_organized_treatments
            
            # Reorder both pivot tables
            detailed_accuracy_pivot = detailed_accuracy_pivot.reindex(organized_treatments)
            detailed_choice_1_pct_pivot = detailed_choice_1_pct_pivot.reindex(organized_treatments)
            
            analysis['detailed_accuracy_pivot'] = detailed_accuracy_pivot
            analysis['detailed_choice_1_pct_pivot'] = detailed_choice_1_pct_pivot
            analysis['detailed_breakdown'] = detailed_df
    
    return analysis


def create_detailed_heatmaps(analysis: Dict, output_dir: str, organized_models: List[str]) -> None:
    """Create heatmaps for the detailed pivot tables."""
    print("  Creating heatmaps...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Accuracy heatmap
        if 'detailed_accuracy_pivot' in analysis and not analysis['detailed_accuracy_pivot'].empty:
            pivot = analysis['detailed_accuracy_pivot']
            
            n_models = len(pivot.columns)
            n_treatments = len(pivot.index)
            
            fig_width = min(20, max(12, n_models * 0.8))
            fig_height = min(15, max(8, n_treatments * 0.6))
            
            plt.figure(figsize=(fig_width, fig_height))
            
            # Clean labels
            cleaned_pivot = pivot.copy()
            cleaned_pivot.columns = [clean_model_label(col) for col in pivot.columns]
            cleaned_pivot.index = [clean_treatment_label(idx) for idx in pivot.index]
            
            # Create mask for NaN values
            nan_mask = cleaned_pivot.isna()
            
            # Create heatmap
            sns.heatmap(cleaned_pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                       cbar_kws={'label': 'Accuracy'}, linewidths=0.5,
                       mask=nan_mask, cbar=True, vmin=0, vmax=1)
            
            # Add grey rectangles for NaN
            ax = plt.gca()
            for i in range(len(cleaned_pivot.index)):
                for j in range(len(cleaned_pivot.columns)):
                    if nan_mask.iloc[i, j]:
                        rect = plt.Rectangle((j, i), 1, 1, facecolor='darkgrey',
                                           edgecolor='white', linewidth=0.5)
                        ax.add_patch(rect)
            
            if nan_mask.any().any():
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='darkgrey', label='Not Applicable')]
                ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.15, -0.15))
            
            plt.title('Accuracy Heatmap: Model vs Treatment', fontsize=14, pad=20)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Treatment Type', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            dpi = 150 if n_models * n_treatments > 50 else 300
            plt.savefig(os.path.join(output_dir, 'detailed_accuracy_heatmap.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("    [OK] Accuracy heatmap saved")
        
        # Choice 1 pct heatmap
        if 'detailed_choice_1_pct_pivot' in analysis and not analysis['detailed_choice_1_pct_pivot'].empty:
            pivot = analysis['detailed_choice_1_pct_pivot']
            
            n_models = len(pivot.columns)
            n_treatments = len(pivot.index)
            
            fig_width = min(20, max(12, n_models * 0.8))
            fig_height = min(15, max(8, n_treatments * 0.6))
            
            plt.figure(figsize=(fig_width, fig_height))
            
            # Clean labels
            cleaned_pivot = pivot.copy()
            cleaned_pivot.columns = [clean_model_label(col) for col in pivot.columns]
            cleaned_pivot.index = [clean_treatment_label(idx) for idx in pivot.index]
            
            # Create mask for NaN values
            nan_mask = cleaned_pivot.isna()
            
            # Create heatmap
            sns.heatmap(cleaned_pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                       cbar_kws={'label': 'Choice 1 Proportion'}, linewidths=0.5,
                       mask=nan_mask, cbar=True, vmin=0, vmax=1)
            
            # Add grey rectangles for NaN
            ax = plt.gca()
            for i in range(len(cleaned_pivot.index)):
                for j in range(len(cleaned_pivot.columns)):
                    if nan_mask.iloc[i, j]:
                        rect = plt.Rectangle((j, i), 1, 1, facecolor='darkgrey',
                                           edgecolor='white', linewidth=0.5)
                        ax.add_patch(rect)
            
            if nan_mask.any().any():
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='darkgrey', label='Not Applicable')]
                ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.15, -0.15))
            
            plt.title('Choice 1 Proportion Heatmap: Model vs Treatment', fontsize=14, pad=20)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Treatment Type', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            dpi = 150 if n_models * n_treatments > 50 else 300
            plt.savefig(os.path.join(output_dir, 'detailed_choice_1_pct_heatmap.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("    [OK] Choice 1 pct heatmap saved")
            
    except Exception as e:
        print(f"    [ERROR] Error creating heatmaps: {e}")
        import traceback
        traceback.print_exc()


def save_detailed_results(df: pd.DataFrame, analysis: Dict, output_dir: str) -> None:
    """Save detailed analysis results to CSV files."""
    print("  Saving results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save experiment accuracy
        analysis['by_experiment'].to_csv(os.path.join(output_dir, 'accuracy_by_experiment.csv'))
        
        # Save detailed pivot tables
        if 'detailed_accuracy_pivot' in analysis:
            analysis['detailed_accuracy_pivot'].to_csv(os.path.join(output_dir, 'detailed_accuracy_pivot_table.csv'))
        if 'detailed_choice_1_pct_pivot' in analysis:
            analysis['detailed_choice_1_pct_pivot'].to_csv(os.path.join(output_dir, 'detailed_choice_1_pct_pivot_table.csv'))
        if 'detailed_breakdown' in analysis:
            analysis['detailed_breakdown'].to_csv(os.path.join(output_dir, 'detailed_breakdown_data.csv'), index=False)
        
        print(f"    [OK] Results saved to {output_dir}/")
        
    except PermissionError as e:
        print(f"    [WARNING] Could not save some CSV files: {e}")
    except Exception as e:
        print(f"    [ERROR] Error saving results: {e}")


def process_wikisum_directory(dir_path: str, exp_info: Dict, output_base_dir: str) -> None:
    """
    Process a single WikiSum directory and create analysis outputs.
    
    Args:
        dir_path: Path to the WikiSum directory
        exp_info: Parsed experiment information
        output_base_dir: Base output directory
    """
    dirname = os.path.basename(dir_path)
    print(f"\nProcessing {dirname}...")
    print(f"  Tag: {exp_info['tag_type']}, Exp: {exp_info['exp_type']}, Paradigm: {exp_info['paradigm']}, Priming: {exp_info['priming']}")
    
    # Load and process results
    df = load_and_process_results(dir_path, exp_info)
    
    if df.empty:
        print(f"  [WARNING] No data loaded for {dirname}")
        return
    
    print(f"  Loaded {len(df)} total rows")
    
    # Perform analysis
    analysis = analyze_accuracy_by_conditions(df, exp_info)
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, dirname)
    
    # Get organized models for heatmaps
    if 'detailed_accuracy_pivot' in analysis:
        organized_models = list(analysis['detailed_accuracy_pivot'].columns)
    else:
        organized_models = []
    
    # Create heatmaps
    create_detailed_heatmaps(analysis, output_dir, organized_models)
    
    # Save results
    save_detailed_results(df, analysis, output_dir)
    
    print(f"  [COMPLETE] {dirname}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze Combined Results (analyze_results_1_combined.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--results-base-dir",
                       default="results_and_data/results",
                       help="Base directory containing WikiSum- result directories")
    
    parser.add_argument("--output-base-dir",
                       default="results_and_data/analysis/analyze_results_1_combined",
                       help="Base directory to save analysis results")
    
    args = parser.parse_args()
    
    # Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Convert to absolute paths
    if not os.path.isabs(args.results_base_dir):
        args.results_base_dir = os.path.join(project_root, args.results_base_dir)
    if not os.path.isabs(args.output_base_dir):
        args.output_base_dir = os.path.join(project_root, args.output_base_dir)
    
    print("="*80)
    print("COMBINED RESULTS ANALYSIS (analyze_results_1_combined.py)")
    print("="*80)
    print(f"Results directory: {args.results_base_dir}")
    print(f"Output directory: {args.output_base_dir}")
    print()
    
    try:
        # Find all WikiSum directories
        wikisum_dirs = find_wikisum_dirs(args.results_base_dir)
        print()
        
        # Process each directory
        for dir_path, exp_info in wikisum_dirs:
            try:
                process_wikisum_directory(dir_path, exp_info, args.output_base_dir)
            except Exception as e:
                dirname = os.path.basename(dir_path)
                print(f"\n[ERROR] Error processing {dirname}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {args.output_base_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

