#!/usr/bin/env python3
"""
Assist Tag Recognition Results Analysis Script

This script analyzes the results from assist tag recognition experiments to answer:
1. Number of correct answers given by each model
2. Under which conditions this happened (treatment types, response order)
3. Whether there was an effect due to response order

USAGE:
    python analyze_results_1.py --results-dir RESULTS_DIR --output-dir OUTPUT_DIR
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
        if model_name.startswith('anthropic_'):
            return 'anthropic'
        elif model_name.startswith('google_'):
            return 'google'
        elif model_name.startswith('gpt-'):
            return 'openai'
        else:
            return 'other'
    
    def get_model_size(model_name: str) -> int:
        """Extract model size for ordering within company based on actual model capabilities."""
        # Clean the model name for comparison
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
                return 2  # Default to sonnet level
        
        # Google models (ordered by capability: flash < pro)
        elif 'gemini' in clean_name:
            if 'pro' in clean_name:
                return 2
            else:  # flash models
                return 1
        
        # OpenAI models (ordered by capability: mini < standard)
        elif 'gpt' in clean_name:
            if 'mini' in clean_name:
                # Differentiate between different mini models
                if '4o-mini' in clean_name:
                    return 1  # 4o-mini is smallest
                elif '4.1-mini' in clean_name:
                    return 2  # 4.1-mini is medium
                else:
                    return 1  # Other mini models
            else:  # standard models
                return 3
        
        # Default size if not recognized
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
    """
    Clean model name by removing prefixes, suffixes, and unnecessary parts.
    
    Args:
        model_name: Original model name
        
    Returns:
        Cleaned model name
    """
    # Remove company prefixes
    if model_name.startswith('anthropic_'):
        model_name = model_name.replace('anthropic_', '')
    elif model_name.startswith('google_'):
        model_name = model_name.replace('google_', '')
    
    # Remove date suffixes (e.g., -20241022, -2025-04-14)
    model_name = re.sub(r'-\d{8}$', '', model_name)  # Remove -20241022
    model_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model_name)  # Remove -2025-04-14
    
    return model_name


def clean_treatment_label(treatment_name: str) -> str:
    """
    Clean treatment name by removing model_comparison_ prefix and company names.
    
    Args:
        treatment_name: Original treatment name
        
    Returns:
        Cleaned treatment name
    """
    # Remove model_comparison_ prefix
    if treatment_name.startswith('model_comparison_'):
        treatment_name = treatment_name.replace('model_comparison_', '')
    
    # Remove company prefixes from model names in treatment
    treatment_name = treatment_name.replace('anthropic_', '')
    treatment_name = treatment_name.replace('google_', '')
    
    # Remove date suffixes
    treatment_name = re.sub(r'-\d{8}$', '', treatment_name)
    treatment_name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', treatment_name)
    
    return treatment_name


def load_all_results(results_dir: str) -> pd.DataFrame:
    """
    Load all choice results CSV files from the results directory.
    
    Args:
        results_dir: Path to directory containing choice_results.csv files
        
    Returns:
        Combined DataFrame with all results
    """
    all_results = []
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all choice_results.csv files
    csv_files = []
    for file in os.listdir(results_dir):
        if file.endswith('_choice_results.csv'):
            csv_files.append(os.path.join(results_dir, file))
    
    if not csv_files:
        raise ValueError(f"No choice_results.csv files found in {results_dir}")
    
    print(f"📊 Found {len(csv_files)} result files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load each file
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Add experiment identifier
            experiment_name = os.path.basename(file_path).replace('_choice_results.csv', '')
            df['experiment'] = experiment_name
            all_results.append(df)
            print(f"  ✅ Loaded {len(df)} rows from {experiment_name}")
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
    
    if not all_results:
        raise ValueError("No valid result files could be loaded")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n📈 Total combined results: {len(combined_df)} rows")
    
    return combined_df


def analyze_accuracy_by_conditions(df: pd.DataFrame) -> Dict:
    """
    Analyze accuracy across different conditions.
    
    Args:
        df: Combined results DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*80)
    print("📊 ACCURACY ANALYSIS BY CONDITIONS")
    print("="*80)
    
    analysis = {}
    
    # 1. Accuracy by experiment type
    print(f"\n🔬 ACCURACY BY EXPERIMENT TYPE:")
    experiment_accuracy = df.groupby('experiment').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    experiment_accuracy.columns = ['total_trials', 'correct_trials', 'accuracy']
    
    # Calculate choice 1 percentage separately
    choice_1_pct = df.groupby('experiment')['selected_choice'].apply(lambda x: (x == 1).sum() / len(x) * 100)
    experiment_accuracy['choice_1_pct'] = choice_1_pct.round(1)
    
    experiment_accuracy = experiment_accuracy.sort_values('accuracy', ascending=False)
    
    for experiment, row in experiment_accuracy.iterrows():
        print(f"  {experiment}: {row['accuracy']:.3f} ({int(row['correct_trials'])}/{int(row['total_trials'])}) | Choice 1: {row['choice_1_pct']:.1f}%")
    
    analysis['by_experiment'] = experiment_accuracy
    
    # 2. Create detailed pivot table with "other_model" breakdown
    print(f"\n📊 DETAILED ACCURACY BY MODEL AND TREATMENT (WITH OTHER_MODEL BREAKDOWN):")
    
    # Parse experiment names to extract model and treatment
    def parse_experiment_name(experiment_name: str) -> tuple:
        """Parse experiment name to extract model and treatment."""
        if "_control_vs_" in experiment_name:
            parts = experiment_name.split('_control_vs_')
            if len(parts) == 2:
                model_name = parts[0]
                treatment_name = parts[1]
                return model_name, treatment_name
        elif "_vs_all_others_control_comparison" in experiment_name:
            model_name = experiment_name.replace("_vs_all_others_control_comparison", "")
            return model_name, "vs_all_others"
        return experiment_name, "unknown"
    
    # Create detailed analysis by combining experiment-level data with conversation-level data
    # We need to get the detailed treatment information from the original data
    detailed_analysis = []
    
    for experiment_name in experiment_accuracy.index:
        # Get the experiment data
        exp_data = df[df['experiment'] == experiment_name]
        
        # Parse the experiment name to get base model
        base_model, treatment_type = parse_experiment_name(experiment_name)
        
        # Only break down "vs_all_others" treatments by specific models
        if treatment_type == "vs_all_others":
            # Group by the specific model being compared
            for other_model in exp_data['model_other'].unique():
                model_data = exp_data[exp_data['model_other'] == other_model]
                if len(model_data) > 0:
                    # Check if this is a self-comparison (model_base == model_other)
                    is_self_comparison = (base_model == other_model)
                    
                    if is_self_comparison:
                        # Self-comparisons should be null/NaN
                        accuracy = np.nan
                        choice_1_pct = np.nan
                    else:
                        accuracy = model_data['is_correct'].mean()
                        choice_1_pct = (model_data['selected_choice'] == 1).mean() * 100
                    
                    detailed_analysis.append({
                        'experiment': experiment_name,
                        'base_model': base_model,
                        'treatment': f"model_comparison_{other_model}",
                        'other_model': other_model,
                        'accuracy': accuracy,
                        'choice_1_pct': choice_1_pct,
                        'total_trials': len(model_data),
                        'is_self_comparison': is_self_comparison
                    })
        else:
            # For non-vs_all_others treatments (typo, capitalization), use the original data
            accuracy = exp_data['is_correct'].mean()
            choice_1_pct = (exp_data['selected_choice'] == 1).mean() * 100
            detailed_analysis.append({
                'experiment': experiment_name,
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
        # Get organized model order for columns
        models = detailed_df['base_model'].unique().tolist()
        organized_models = organize_models_by_company_and_size(models)
        
        # Create detailed accuracy pivot table with organized columns
        # Use fill_value=None to preserve NaN values for self-comparisons
        detailed_accuracy_pivot = detailed_df.pivot_table(
            index='treatment',
            columns='base_model',
            values='accuracy',
            fill_value=None
        ).round(3)
        
        # Reorder columns to match organized model order
        detailed_accuracy_pivot = detailed_accuracy_pivot[organized_models]
        
        print("\n--- Detailed Accuracy Table (Model vs Treatment with Other Model Breakdown) ---")
        print(detailed_accuracy_pivot.to_string())
        
        # Create detailed choice 1 proportion pivot table with organized columns
        # Convert percentages to proportions (divide by 100) and use fill_value=None to preserve NaN values for self-comparisons
        detailed_choice_1_pct_pivot = detailed_df.pivot_table(
            index='treatment',
            columns='base_model',
            values='choice_1_pct',
            fill_value=None
        ) / 100  # Convert percentages to proportions
        detailed_choice_1_pct_pivot = detailed_choice_1_pct_pivot.round(3)
        
        # Reorder columns to match organized model order
        detailed_choice_1_pct_pivot = detailed_choice_1_pct_pivot[organized_models]
        
        # Reorder rows for model_comparison treatments by the model being compared
        def get_model_comparison_order(treatment_name):
            """Extract model name from model_comparison treatment for ordering."""
            if treatment_name.startswith('model_comparison_'):
                model_name = treatment_name.replace('model_comparison_', '')
                return model_name
            return treatment_name
        
        # Get all treatment names and sort model_comparison by model
        all_treatments = detailed_accuracy_pivot.index.tolist()
        model_comparison_treatments = [t for t in all_treatments if t.startswith('model_comparison_')]
        other_treatments = [t for t in all_treatments if not t.startswith('model_comparison_')]
        
        # Sort model_comparison treatments by the model name
        model_comparison_models = [get_model_comparison_order(t) for t in model_comparison_treatments]
        model_comparison_organized = organize_models_by_company_and_size(model_comparison_models)
        
        # Reconstruct treatment names with organized models
        model_comparison_organized_treatments = [f"model_comparison_{model}" for model in model_comparison_organized]
        
        # Combine with other treatments (keep original order)
        organized_treatments = other_treatments + model_comparison_organized_treatments
        
        # Reorder both pivot tables
        detailed_accuracy_pivot = detailed_accuracy_pivot.reindex(organized_treatments)
        detailed_choice_1_pct_pivot = detailed_choice_1_pct_pivot.reindex(organized_treatments)
        
        print("\n--- Detailed Choice 1 Proportion Table (Model vs Treatment with Other Model Breakdown) ---")
        print(detailed_choice_1_pct_pivot.to_string())
        
        # Store detailed pivot tables
        analysis['detailed_accuracy_pivot'] = detailed_accuracy_pivot
        analysis['detailed_choice_1_pct_pivot'] = detailed_choice_1_pct_pivot
        analysis['detailed_breakdown'] = detailed_df
    
    return analysis






def create_detailed_heatmaps(analysis: Dict, output_dir: str) -> None:
    """
    Create heatmaps for the detailed pivot tables with organized model ordering.
    
    Args:
        analysis: Analysis results dictionary containing pivot tables
        output_dir: Directory to save plots
    """
    print(f"\n📊 Creating detailed heatmaps...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Detailed Accuracy Heatmap
        if 'detailed_accuracy_pivot' in analysis and not analysis['detailed_accuracy_pivot'].empty:
            print("  Creating accuracy heatmap...")
            # Adjust figure size based on data size to prevent memory issues
            n_models = len(analysis['detailed_accuracy_pivot'].columns)
            n_treatments = len(analysis['detailed_accuracy_pivot'].index)
            
            # Scale figure size based on data dimensions
            fig_width = min(20, max(12, n_models * 0.8))
            fig_height = min(15, max(8, n_treatments * 0.6))
            
            plt.figure(figsize=(fig_width, fig_height))
        
            # Get organized model order
            models = list(analysis['detailed_accuracy_pivot'].columns)
            organized_models = organize_models_by_company_and_size(models)
            
            # Reorder the pivot table
            reordered_pivot = analysis['detailed_accuracy_pivot'][organized_models]
            
            # Clean the labels
            cleaned_columns = [clean_model_label(col) for col in reordered_pivot.columns]
            cleaned_index = [clean_treatment_label(idx) for idx in reordered_pivot.index]
            
            # Create a copy with cleaned labels
            cleaned_pivot = reordered_pivot.copy()
            cleaned_pivot.columns = cleaned_columns
            cleaned_pivot.index = cleaned_index
            
            # Create heatmap with proper NaN handling using mask
            # First, create a mask for NaN values
            nan_mask = cleaned_pivot.isna()
            
            # Create heatmap with mask for NaN values using red-to-blue gradient
            sns.heatmap(cleaned_pivot, annot=True, fmt='.3f', cmap='RdBu_r', 
                        cbar_kws={'label': 'Accuracy'}, linewidths=0.5, 
                        mask=nan_mask, cbar=True, vmin=0, vmax=1)
            
            # Manually add grey rectangles for NaN values
            ax = plt.gca()
            for i in range(len(cleaned_pivot.index)):
                for j in range(len(cleaned_pivot.columns)):
                    if nan_mask.iloc[i, j]:
                        # Add a grey rectangle for NaN values
                        rect = plt.Rectangle((j, i), 1, 1, facecolor='darkgrey', 
                                           edgecolor='white', linewidth=0.5)
                        ax.add_patch(rect)
            
            # Add legend for NaN values
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='darkgrey', label='Not Applicable')]
            ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.15, -0.15))
            
            plt.title('Accuracy Heatmap: Model vs Treatment', 
                     fontsize=14, pad=20)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Treatment Type', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add company separators for columns
            for i, model in enumerate(organized_models):
                company = 'anthropic' if model.startswith('anthropic_') else \
                         'google' if model.startswith('google_') else \
                         'openai' if model.startswith('gpt-') else 'other'
                
                if i > 0 and company != ('anthropic' if organized_models[i-1].startswith('anthropic_') else \
                                       'google' if organized_models[i-1].startswith('google_') else \
                                       'openai' if organized_models[i-1].startswith('gpt-') else 'other'):
                    plt.axvline(x=i, color='black', linewidth=2, alpha=0.7)
            
            # Add treatment category separators for rows
            # Use original treatment names before cleaning to detect vs_all_others
            treatment_categories = []
            current_category = None
            current_company = None
            
            for i, treatment in enumerate(cleaned_pivot.index):
                # Check original treatment name before cleaning
                original_treatment = reordered_pivot.index[i]
                
                if original_treatment.startswith('model_comparison_'):
                    category = 'model_comparison'
                    # Extract company from the model name in model_comparison treatment
                    model_name = original_treatment.replace('model_comparison_', '')
                    if model_name.startswith('anthropic_'):
                        company = 'anthropic'
                    elif model_name.startswith('google_'):
                        company = 'google'
                    elif model_name.startswith('gpt-'):
                        company = 'openai'
                    else:
                        company = 'other'
                elif 'capitalization' in treatment:
                    category = 'capitalization'
                    company = None  # Not applicable for non-model_comparison treatments
                elif 'typo' in treatment:
                    category = 'typo'
                    company = None  # Not applicable for non-model_comparison treatments
                else:
                    category = 'other'
                    company = None  # Not applicable for non-model_comparison treatments
                
                # Add separator if category changes
                if current_category is not None and category != current_category:
                    plt.axhline(y=i, color='black', linewidth=2, alpha=0.7)
                # Add separator if within model_comparison and company changes
                elif (category == 'model_comparison' and current_category == 'model_comparison' 
                      and current_company is not None and company != current_company):
                    plt.axhline(y=i, color='black', linewidth=2, alpha=0.7)
                
                current_category = category
                current_company = company
                treatment_categories.append(category)
            
            plt.tight_layout()
            # Use lower DPI for large datasets to prevent memory issues
            dpi = 150 if n_models * n_treatments > 50 else 300
            plt.savefig(os.path.join(output_dir, 'detailed_accuracy_heatmap.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("  ✅ Accuracy heatmap saved")
    
        # 2. Detailed Choice 1 Percentage Heatmap
        if 'detailed_choice_1_pct_pivot' in analysis and not analysis['detailed_choice_1_pct_pivot'].empty:
            print("  Creating choice 1 percentage heatmap...")
            # Adjust figure size based on data size to prevent memory issues
            n_models = len(analysis['detailed_choice_1_pct_pivot'].columns)
            n_treatments = len(analysis['detailed_choice_1_pct_pivot'].index)
            
            # Scale figure size based on data dimensions
            fig_width = min(20, max(12, n_models * 0.8))
            fig_height = min(15, max(8, n_treatments * 0.6))
            
            plt.figure(figsize=(fig_width, fig_height))
            
            # Get organized model order
            models = list(analysis['detailed_choice_1_pct_pivot'].columns)
            organized_models = organize_models_by_company_and_size(models)
            
            # Reorder the pivot table
            reordered_pivot = analysis['detailed_choice_1_pct_pivot'][organized_models]
            
            # Clean the labels
            cleaned_columns = [clean_model_label(col) for col in reordered_pivot.columns]
            cleaned_index = [clean_treatment_label(idx) for idx in reordered_pivot.index]
            
            # Create a copy with cleaned labels
            cleaned_pivot = reordered_pivot.copy()
            cleaned_pivot.columns = cleaned_columns
            cleaned_pivot.index = cleaned_index
            
            # Create heatmap with proper NaN handling using mask
            # First, create a mask for NaN values
            nan_mask = cleaned_pivot.isna()
            
            # Create heatmap with mask for NaN values using red-to-blue gradient
            sns.heatmap(cleaned_pivot, annot=True, fmt='.3f', cmap='RdBu_r', 
                        cbar_kws={'label': 'Choice 1 Proportion'}, linewidths=0.5,
                        mask=nan_mask, cbar=True, vmin=0, vmax=1)
            
            # Manually add grey rectangles for NaN values
            ax = plt.gca()
            for i in range(len(cleaned_pivot.index)):
                for j in range(len(cleaned_pivot.columns)):
                    if nan_mask.iloc[i, j]:
                        # Add a grey rectangle for NaN values
                        rect = plt.Rectangle((j, i), 1, 1, facecolor='darkgrey', 
                                           edgecolor='white', linewidth=0.5)
                        ax.add_patch(rect)
            
            # Add legend for NaN values
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='darkgrey', label='Not Applicable')]
            ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.15, -0.15))
            
            plt.title('Choice 1 Proportion Heatmap: Model vs Treatment', 
                     fontsize=14, pad=20)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Treatment Type', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add company separators for columns
            for i, model in enumerate(organized_models):
                company = 'anthropic' if model.startswith('anthropic_') else \
                         'google' if model.startswith('google_') else \
                         'openai' if model.startswith('gpt-') else 'other'
                
                if i > 0 and company != ('anthropic' if organized_models[i-1].startswith('anthropic_') else \
                                       'google' if organized_models[i-1].startswith('google_') else \
                                       'openai' if organized_models[i-1].startswith('gpt-') else 'other'):
                    plt.axvline(x=i, color='black', linewidth=2, alpha=0.7)
            
            # Add treatment category separators for rows
            # Use original treatment names before cleaning to detect vs_all_others
            treatment_categories = []
            current_category = None
            current_company = None
            
            for i, treatment in enumerate(cleaned_pivot.index):
                # Check original treatment name before cleaning
                original_treatment = reordered_pivot.index[i]
                
                if original_treatment.startswith('model_comparison_'):
                    category = 'model_comparison'
                    # Extract company from the model name in model_comparison treatment
                    model_name = original_treatment.replace('model_comparison_', '')
                    if model_name.startswith('anthropic_'):
                        company = 'anthropic'
                    elif model_name.startswith('google_'):
                        company = 'google'
                    elif model_name.startswith('gpt-'):
                        company = 'openai'
                    else:
                        company = 'other'
                elif 'capitalization' in treatment:
                    category = 'capitalization'
                    company = None  # Not applicable for non-model_comparison treatments
                elif 'typo' in treatment:
                    category = 'typo'
                    company = None  # Not applicable for non-model_comparison treatments
                else:
                    category = 'other'
                    company = None  # Not applicable for non-model_comparison treatments
                
                # Add separator if category changes
                if current_category is not None and category != current_category:
                    plt.axhline(y=i, color='black', linewidth=2, alpha=0.7)
                # Add separator if within model_comparison and company changes
                elif (category == 'model_comparison' and current_category == 'model_comparison' 
                      and current_company is not None and company != current_company):
                    plt.axhline(y=i, color='black', linewidth=2, alpha=0.7)
                
                current_category = category
                current_company = company
                treatment_categories.append(category)
            
            plt.tight_layout()
            # Use lower DPI for large datasets to prevent memory issues
            dpi = 150 if n_models * n_treatments > 50 else 300
            plt.savefig(os.path.join(output_dir, 'detailed_choice_1_pct_heatmap.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("  ✅ Choice 1 percentage heatmap saved")
    
    except Exception as e:
        print(f"  ❌ Error creating detailed heatmaps: {e}")
        print(f"  💡 This might be due to memory constraints or data size issues")
        print(f"  🔄 Continuing with other analysis...")
    
    print(f"  ✅ Heatmaps processing completed")


def save_detailed_results(df: pd.DataFrame, analysis: Dict, output_dir: str) -> None:
    """
    Save detailed analysis results to CSV files.
    
    Args:
        df: Combined results DataFrame
        analysis: Analysis results dictionary
        output_dir: Directory to save results
    """
    print(f"\n💾 Saving detailed results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save essential data
    try:
        # Save experiment accuracy
        analysis['by_experiment'].to_csv(os.path.join(output_dir, 'accuracy_by_experiment.csv'))
        
        # Save detailed pivot tables with other_model breakdown
        if 'detailed_accuracy_pivot' in analysis:
            analysis['detailed_accuracy_pivot'].to_csv(os.path.join(output_dir, 'detailed_accuracy_pivot_table.csv'))
        if 'detailed_choice_1_pct_pivot' in analysis:
            analysis['detailed_choice_1_pct_pivot'].to_csv(os.path.join(output_dir, 'detailed_choice_1_pct_pivot_table.csv'))
        if 'detailed_breakdown' in analysis:
            analysis['detailed_breakdown'].to_csv(os.path.join(output_dir, 'detailed_breakdown_data.csv'), index=False)
            
    except PermissionError as e:
        print(f"  ⚠️  Warning: Could not save some CSV files due to permission error: {e}")
        print(f"  💡 Try closing any open CSV files and run again")
    
    print(f"  ✅ Detailed results saved to {output_dir}/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze Assist Tag Recognition Results (analyze_results_1.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--results-dir", 
                       default="results_and_data/results/to_run",
                       help="Directory containing choice_results.csv files")
    
    parser.add_argument("--output-dir",
                       default=None,
                       help="Directory to save analysis results and plots (default: results_and_data/analysis/{results_subdir})")
    
    args = parser.parse_args()
    
    # Set default output directory based on results subdirectory if not provided
    if args.output_dir is None:
        results_subdir = os.path.basename(args.results_dir.rstrip('/'))
        args.output_dir = f"results_and_data/analysis/analyze_results_1/{results_subdir}"
    
    print("="*80)
    print("🔍 ASSIST TAG RECOGNITION RESULTS ANALYSIS (analyze_results_1.py)")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Load all results
        df = load_all_results(args.results_dir)
        
        # Perform analysis
        analysis = analyze_accuracy_by_conditions(df)
        
        # Create detailed heatmaps
        create_detailed_heatmaps(analysis, args.output_dir)
        
        # Save detailed results
        save_detailed_results(df, analysis, args.output_dir)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"🔥 Heatmaps: {args.output_dir}/*_heatmap.png")
        print(f"📋 Data files: {args.output_dir}/*.csv")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
