#!/usr/bin/env python3
"""
Assist Tag Recognition Results Analysis Script

This script analyzes the results from assist tag recognition experiments to answer:
1. Number of correct answers given by each model
2. Under which conditions this happened (treatment types, response order)
3. Whether there was an effect due to response order

USAGE:
    python analyze_assist_tag_results.py [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]
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
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


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
    
    # 1. Overall accuracy
    overall_accuracy = df['is_correct'].mean()
    total_trials = len(df)
    correct_trials = df['is_correct'].sum()
    
    print(f"\n🎯 OVERALL ACCURACY:")
    print(f"  Total trials: {total_trials:,}")
    print(f"  Correct answers: {correct_trials:,}")
    print(f"  Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    
    analysis['overall'] = {
        'total_trials': total_trials,
        'correct_trials': correct_trials,
        'accuracy': overall_accuracy
    }
    
    # 2. Accuracy by model
    print(f"\n🤖 ACCURACY BY MODEL:")
    model_accuracy = df.groupby('model_base').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    model_accuracy.columns = ['total_trials', 'correct_trials', 'accuracy']
    model_accuracy = model_accuracy.sort_values('accuracy', ascending=False)
    
    for model, row in model_accuracy.iterrows():
        print(f"  {model}: {row['accuracy']:.3f} ({int(row['correct_trials'])}/{int(row['total_trials'])})")
    
    analysis['by_model'] = model_accuracy
    
    # 3. Accuracy by treatment type
    print(f"\n🧪 ACCURACY BY TREATMENT TYPE:")
    treatment_accuracy = df.groupby('treatment_other').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    treatment_accuracy.columns = ['total_trials', 'correct_trials', 'accuracy']
    treatment_accuracy = treatment_accuracy.sort_values('accuracy', ascending=False)
    
    for treatment, row in treatment_accuracy.iterrows():
        print(f"  {treatment}: {row['accuracy']:.3f} ({int(row['correct_trials'])}/{int(row['total_trials'])})")
    
    analysis['by_treatment'] = treatment_accuracy
    
    # 3.5. Accuracy by treatment type with "other_model" broken down by specific models
    print(f"\n🔍 ACCURACY BY TREATMENT TYPE (DETAILED - OTHER_MODEL BREAKDOWN):")
    
    # Create a detailed treatment column that breaks down "other_model" by specific model
    df_detailed = df.copy()
    df_detailed['treatment_detailed'] = df_detailed['treatment_other']
    
    # For "other_model" treatments, replace with the actual model being compared
    other_model_mask = df_detailed['treatment_other'] == 'other_model'
    df_detailed.loc[other_model_mask, 'treatment_detailed'] = df_detailed.loc[other_model_mask, 'model_other']
    
    # Analyze with detailed breakdown
    treatment_detailed_accuracy = df_detailed.groupby('treatment_detailed').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    treatment_detailed_accuracy.columns = ['total_trials', 'correct_trials', 'accuracy']
    treatment_detailed_accuracy = treatment_detailed_accuracy.sort_values('accuracy', ascending=False)
    
    for treatment, row in treatment_detailed_accuracy.iterrows():
        print(f"  {treatment}: {row['accuracy']:.3f} ({int(row['correct_trials'])}/{int(row['total_trials'])})")
    
    analysis['by_treatment_detailed'] = treatment_detailed_accuracy
    
    # 4. Accuracy by response order
    print(f"\n🔄 ACCURACY BY RESPONSE ORDER:")
    order_accuracy = df.groupby('response_1_source').agg({
        'is_correct': ['count', 'sum', 'mean']
    }).round(3)
    order_accuracy.columns = ['total_trials', 'correct_trials', 'accuracy']
    
    for order, row in order_accuracy.iterrows():
        order_name = "Control first" if order == 'control' else "Treatment first"
        print(f"  {order_name} ({order}): {row['accuracy']:.3f} ({int(row['correct_trials'])}/{int(row['total_trials'])})")
    
    analysis['by_order'] = order_accuracy
    
    # 5. Response order effect analysis
    print(f"\n📈 RESPONSE ORDER EFFECT:")
    if len(order_accuracy) == 2:
        control_first_acc = order_accuracy.loc['control', 'accuracy']
        treatment_first_acc = order_accuracy.loc['treatment', 'accuracy']
        order_effect = control_first_acc - treatment_first_acc
        
        print(f"  Control first: {control_first_acc:.3f}")
        print(f"  Treatment first: {treatment_first_acc:.3f}")
        print(f"  Order effect: {order_effect:+.3f} ({'Control first better' if order_effect > 0 else 'Treatment first better' if order_effect < 0 else 'No effect'})")
        
        # Statistical significance test
        from scipy import stats
        control_first_trials = df[df['response_1_source'] == 'control']['is_correct']
        treatment_first_trials = df[df['response_1_source'] == 'treatment']['is_correct']
        
        if len(control_first_trials) > 0 and len(treatment_first_trials) > 0:
            chi2, p_value = stats.chi2_contingency([
                [control_first_trials.sum(), len(control_first_trials) - control_first_trials.sum()],
                [treatment_first_trials.sum(), len(treatment_first_trials) - treatment_first_trials.sum()]
            ])[:2]
            
            print(f"  Chi-square test p-value: {p_value:.4f}")
            print(f"  {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
        
        analysis['order_effect'] = {
            'control_first_accuracy': control_first_acc,
            'treatment_first_accuracy': treatment_first_acc,
            'effect_size': order_effect,
            'p_value': p_value if 'p_value' in locals() else None
        }
    
    # 6. Accuracy by experiment type
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
    
    # 7. Create pivot tables for better organization
    print(f"\n📊 ACCURACY BY MODEL AND TREATMENT (PIVOT TABLE):")
    
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
    
    # Create temporary dataframe with parsed model and treatment
    temp_df = experiment_accuracy.reset_index()
    temp_df[['model', 'treatment']] = temp_df['experiment'].apply(
        lambda x: pd.Series(parse_experiment_name(x))
    )
    
    # Create accuracy pivot table
    accuracy_pivot = temp_df.pivot_table(
        index='treatment',
        columns='model', 
        values='accuracy',
        fill_value=0.0
    ).round(3)
    
    print("\n--- Accuracy Table (Model vs Treatment) ---")
    print(accuracy_pivot.to_string())
    
    # Create choice 1 percentage pivot table
    print(f"\n📊 CHOICE 1 PERCENTAGE BY MODEL AND TREATMENT (PIVOT TABLE):")
    choice_1_pct_pivot = temp_df.pivot_table(
        index='treatment',
        columns='model',
        values='choice_1_pct', 
        fill_value=0.0
    ).round(1)
    
    print("\n--- Choice 1 Percentage Table (Model vs Treatment) ---")
    print(choice_1_pct_pivot.to_string())
    
    # Store pivot tables in analysis results
    analysis['accuracy_pivot'] = accuracy_pivot
    analysis['choice_1_pct_pivot'] = choice_1_pct_pivot
    
    # 8. Create detailed pivot table with "other_model" breakdown
    print(f"\n📊 DETAILED ACCURACY BY MODEL AND TREATMENT (WITH OTHER_MODEL BREAKDOWN):")
    
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
                    accuracy = model_data['is_correct'].mean()
                    choice_1_pct = (model_data['selected_choice'] == 1).mean() * 100
                    detailed_analysis.append({
                        'experiment': experiment_name,
                        'base_model': base_model,
                        'treatment': f"vs_all_others_{other_model}",
                        'other_model': other_model,
                        'accuracy': accuracy,
                        'choice_1_pct': choice_1_pct,
                        'total_trials': len(model_data)
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
        # Create detailed accuracy pivot table
        detailed_accuracy_pivot = detailed_df.pivot_table(
            index='treatment',
            columns='base_model',
            values='accuracy',
            fill_value=0.0
        ).round(3)
        
        print("\n--- Detailed Accuracy Table (Model vs Treatment with Other Model Breakdown) ---")
        print(detailed_accuracy_pivot.to_string())
        
        # Create detailed choice 1 percentage pivot table
        detailed_choice_1_pct_pivot = detailed_df.pivot_table(
            index='treatment',
            columns='base_model',
            values='choice_1_pct',
            fill_value=0.0
        ).round(1)
        
        print("\n--- Detailed Choice 1 Percentage Table (Model vs Treatment with Other Model Breakdown) ---")
        print(detailed_choice_1_pct_pivot.to_string())
        
        # Store detailed pivot tables
        analysis['detailed_accuracy_pivot'] = detailed_accuracy_pivot
        analysis['detailed_choice_1_pct_pivot'] = detailed_choice_1_pct_pivot
        analysis['detailed_breakdown'] = detailed_df
    
    return analysis


def analyze_response_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze response patterns and choice probabilities.
    
    Args:
        df: Combined results DataFrame
        
    Returns:
        Dictionary with pattern analysis
    """
    print("\n" + "="*80)
    print("🎲 RESPONSE PATTERN ANALYSIS")
    print("="*80)
    
    patterns = {}
    
    # 1. Choice distribution
    print(f"\n🎯 CHOICE DISTRIBUTION:")
    choice_dist = df['selected_choice'].value_counts().sort_index()
    choice_pct = df['selected_choice'].value_counts(normalize=True).sort_index() * 100
    
    for choice, count in choice_dist.items():
        pct = choice_pct[choice]
        print(f"  Choice {choice}: {count:,} times ({pct:.1f}%)")
    
    patterns['choice_distribution'] = choice_dist
    
    # 2. Probability analysis
    print(f"\n📊 PROBABILITY ANALYSIS:")
    prob_stats = df[['prob_choice_1', 'prob_choice_2']].describe()
    print(prob_stats.round(4))
    
    # Probability difference analysis
    df['prob_diff'] = df['prob_choice_1'] - df['prob_choice_2']
    print(f"\n  Probability difference (Choice 1 - Choice 2):")
    print(f"    Mean: {df['prob_diff'].mean():.4f}")
    print(f"    Std:  {df['prob_diff'].std():.4f}")
    print(f"    Min:  {df['prob_diff'].min():.4f}")
    print(f"    Max:  {df['prob_diff'].max():.4f}")
    
    patterns['probability_stats'] = prob_stats
    patterns['probability_difference'] = df['prob_diff'].describe()
    
    # 3. Confidence analysis (how confident are the choices?)
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    df['max_prob'] = df[['prob_choice_1', 'prob_choice_2']].max(axis=1)
    df['confidence'] = df['max_prob'] - df[['prob_choice_1', 'prob_choice_2']].min(axis=1)
    
    confidence_stats = df['confidence'].describe()
    print(f"  Confidence (max_prob - min_prob):")
    print(f"    Mean: {confidence_stats['mean']:.4f}")
    print(f"    Median: {confidence_stats['50%']:.4f}")
    print(f"    Std: {confidence_stats['std']:.4f}")
    
    # Confidence by correctness
    correct_confidence = df[df['is_correct'] == True]['confidence']
    incorrect_confidence = df[df['is_correct'] == False]['confidence']
    
    print(f"\n  Confidence by correctness:")
    print(f"    Correct choices: {correct_confidence.mean():.4f} ± {correct_confidence.std():.4f}")
    print(f"    Incorrect choices: {incorrect_confidence.mean():.4f} ± {incorrect_confidence.std():.4f}")
    
    patterns['confidence_stats'] = confidence_stats
    patterns['confidence_by_correctness'] = {
        'correct': correct_confidence.describe(),
        'incorrect': incorrect_confidence.describe()
    }
    
    return patterns


def create_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualization plots for the analysis.
    
    Args:
        df: Combined results DataFrame
        output_dir: Directory to save plots
    """
    print(f"\n📊 Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Accuracy by model
    plt.figure(figsize=(14, 8))
    model_acc = df.groupby('model_base')['is_correct'].agg(['count', 'mean']).reset_index()
    model_acc = model_acc.sort_values('mean', ascending=True)
    
    bars = plt.barh(range(len(model_acc)), model_acc['mean'], 
                    color=plt.cm.viridis(np.linspace(0, 1, len(model_acc))))
    plt.yticks(range(len(model_acc)), model_acc['model_base'])
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    
    # Add count labels
    for i, (idx, row) in enumerate(model_acc.iterrows()):
        plt.text(row['mean'] + 0.01, i, f"n={int(row['count'])}", 
                va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_model.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy by treatment type
    plt.figure(figsize=(10, 6))
    treatment_acc = df.groupby('treatment_other')['is_correct'].mean().sort_values(ascending=True)
    
    bars = plt.bar(range(len(treatment_acc)), treatment_acc.values,
                   color=plt.cm.Set2(np.linspace(0, 1, len(treatment_acc))))
    plt.xticks(range(len(treatment_acc)), treatment_acc.index, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Treatment Type')
    
    # Add value labels
    for i, v in enumerate(treatment_acc.values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_treatment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Response order effect
    plt.figure(figsize=(8, 6))
    order_acc = df.groupby('response_1_source')['is_correct'].agg(['mean', 'count']).reset_index()
    
    bars = plt.bar(order_acc['response_1_source'], order_acc['mean'],
                   color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Response Order Effect')
    plt.xticks([0, 1], ['Control First', 'Treatment First'])
    
    # Add count and accuracy labels
    for i, (idx, row) in enumerate(order_acc.iterrows()):
        plt.text(i, row['mean'] + 0.01, f"{row['mean']:.3f}\n(n={int(row['count'])})", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'response_order_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Probability distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['prob_choice_1'], bins=30, alpha=0.7, label='Choice 1', color='skyblue')
    plt.hist(df['prob_choice_2'], bins=30, alpha=0.7, label='Choice 2', color='lightcoral')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Choice Probability Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(df['confidence'], bins=30, alpha=0.7, color='lightgreen')
    plt.xlabel('Confidence (max_prob - min_prob)')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heatmap of accuracy by model and treatment
    plt.figure(figsize=(12, 8))
    pivot_data = df.groupby(['model_base', 'treatment_other'])['is_correct'].mean().unstack(fill_value=0)
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Accuracy'})
    plt.title('Accuracy Heatmap: Model vs Treatment Type')
    plt.xlabel('Treatment Type')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Visualizations saved to {output_dir}/")


def save_detailed_results(df: pd.DataFrame, analysis: Dict, patterns: Dict, output_dir: str) -> None:
    """
    Save detailed analysis results to CSV files.
    
    Args:
        df: Combined results DataFrame
        analysis: Analysis results dictionary
        patterns: Pattern analysis dictionary
        output_dir: Directory to save results
    """
    print(f"\n💾 Saving detailed results...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save raw data with additional columns
    df_enhanced = df.copy()
    df_enhanced['prob_diff'] = df_enhanced['prob_choice_1'] - df_enhanced['prob_choice_2']
    df_enhanced['max_prob'] = df_enhanced[['prob_choice_1', 'prob_choice_2']].max(axis=1)
    df_enhanced['confidence'] = df_enhanced['max_prob'] - df_enhanced[['prob_choice_1', 'prob_choice_2']].min(axis=1)
    
    df_enhanced.to_csv(os.path.join(output_dir, 'enhanced_results.csv'), index=False)
    
    # 2. Save summary statistics
    summary_stats = []
    
    # Overall stats
    summary_stats.append({
        'metric': 'overall_accuracy',
        'value': analysis['overall']['accuracy'],
        'description': 'Overall accuracy across all experiments'
    })
    
    # Model stats
    for model, row in analysis['by_model'].iterrows():
        summary_stats.append({
            'metric': f'model_{model}_accuracy',
            'value': row['accuracy'],
            'description': f'Accuracy for {model}'
        })
    
    # Treatment stats
    for treatment, row in analysis['by_treatment'].iterrows():
        summary_stats.append({
            'metric': f'treatment_{treatment}_accuracy',
            'value': row['accuracy'],
            'description': f'Accuracy for {treatment} treatment'
        })
    
    # Order effect
    if 'order_effect' in analysis:
        summary_stats.append({
            'metric': 'response_order_effect',
            'value': analysis['order_effect']['effect_size'],
            'description': 'Response order effect size (control_first - treatment_first)'
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    
    # 3. Save detailed breakdowns
    try:
        analysis['by_model'].to_csv(os.path.join(output_dir, 'accuracy_by_model.csv'))
        analysis['by_treatment'].to_csv(os.path.join(output_dir, 'accuracy_by_treatment.csv'))
        analysis['by_experiment'].to_csv(os.path.join(output_dir, 'accuracy_by_experiment.csv'))
        
        # Save detailed treatment breakdown
        if 'by_treatment_detailed' in analysis:
            analysis['by_treatment_detailed'].to_csv(os.path.join(output_dir, 'accuracy_by_treatment_detailed.csv'))
        
        # Save pivot tables
        if 'accuracy_pivot' in analysis:
            analysis['accuracy_pivot'].to_csv(os.path.join(output_dir, 'accuracy_pivot_table.csv'))
        if 'choice_1_pct_pivot' in analysis:
            analysis['choice_1_pct_pivot'].to_csv(os.path.join(output_dir, 'choice_1_pct_pivot_table.csv'))
        
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
        description="Analyze Assist Tag Recognition Results",
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
        args.output_dir = f"results_and_data/analysis/{results_subdir}"
    
    print("="*80)
    print("🔍 ASSIST TAG RECOGNITION RESULTS ANALYSIS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Load all results
        df = load_all_results(args.results_dir)
        
        # Perform analysis
        analysis = analyze_accuracy_by_conditions(df)
        patterns = analyze_response_patterns(df)
        
        # Create visualizations
        create_visualizations(df, args.output_dir)
        
        # Save detailed results
        save_detailed_results(df, analysis, patterns, args.output_dir)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"📈 Visualizations: {args.output_dir}/*.png")
        print(f"📋 Data files: {args.output_dir}/*.csv")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
