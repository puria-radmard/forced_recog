#!/usr/bin/env python3
"""
Check for missing trials in assist tag recognition results.

This script compares the expected number of conversations with the actual results
to identify which trials may have failed due to rate limits or other errors.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def check_missing_trials(results_dir: str = "results_and_data/results"):
    """
    Check for missing trials and conversations with likely false data due to rate limit errors.
    
    Args:
        results_dir: Directory containing result CSV files
    """
    # Find all result CSV files
    csv_files = glob.glob(os.path.join(results_dir, "**", "*_choice_results.csv"), recursive=True)
    
    if not csv_files:
        print("No result CSV files found!")
        return
    
    issues_found = False
    
    for csv_file in csv_files:
        try:
            # Load the results
            df = pd.read_csv(csv_file)
            
            if df.empty:
                experiment_name = os.path.basename(csv_file).replace('_choice_results.csv', '')
                print(f"{experiment_name}: No results recorded (file is empty)")
                issues_found = True
                continue
            
            experiment_name = os.path.basename(csv_file).replace('_choice_results.csv', '')
            
            # Check for missing trials
            if 'trial' in df.columns:
                trials = sorted(df['trial'].unique())
                if trials:
                    expected_trials = list(range(1, max(trials) + 1))
                    missing_trials = set(expected_trials) - set(trials)
                    
                    if missing_trials:
                        print(f"{experiment_name}: Missing trials {sorted(missing_trials)}")
                        issues_found = True
                    
                    # Check for incomplete trials (fewer conversations than expected)
                    trial_counts = df['trial'].value_counts().sort_index()
                    if len(trial_counts) > 0:
                        # Get the most common conversation count (should be the expected count)
                        mode_values = trial_counts.mode()
                        if len(mode_values) > 0:
                            expected_conversations_per_trial = int(mode_values.iloc[0])
                            
                            incomplete_trials = []
                            for trial, count in trial_counts.items():
                                if int(count) < expected_conversations_per_trial:
                                    incomplete_trials.append(trial)
                            
                            if incomplete_trials:
                                print(f"{experiment_name}: Incomplete trials {sorted(incomplete_trials)} (expected {expected_conversations_per_trial} conversations each)")
                                issues_found = True
            
            # Check for conversations with likely false data (neutral logits)
            if 'prob_choice_1' in df.columns and 'prob_choice_2' in df.columns:
                # Detect neutral logits (probabilities very close to 0.5)
                neutral_threshold = 0.1  # If both probabilities are within 0.1 of 0.5
                neutral_mask = (
                    (abs(df['prob_choice_1'] - 0.5) < neutral_threshold) & 
                    (abs(df['prob_choice_2'] - 0.5) < neutral_threshold)
                )
                
                neutral_conversations = df[neutral_mask]
                
                if len(neutral_conversations) > 0:
                    print(f"{experiment_name}: {len(neutral_conversations)} conversations with neutral logits (likely rate limit errors)")
                    
                    # Group by trial to show which trials are affected
                    if 'trial' in neutral_conversations.columns:
                        affected_trials = sorted(neutral_conversations['trial'].unique())
                        print(f"  Affected trials: {affected_trials}")
                        
                        # Show sample conversation IDs
                        sample_convs = neutral_conversations['conversation_id'].head(3).tolist()
                        print(f"  Sample conversation IDs: {sample_convs}")
                    
                    issues_found = True
            
            # Check for conversations with identical probabilities (another sign of neutral logits)
            if 'prob_choice_1' in df.columns and 'prob_choice_2' in df.columns:
                identical_probs = df[df['prob_choice_1'] == df['prob_choice_2']]
                if len(identical_probs) > 0:
                    print(f"{experiment_name}: {len(identical_probs)} conversations with identical probabilities (prob_choice_1 == prob_choice_2)")
                    issues_found = True
            
        except Exception as e:
            experiment_name = os.path.basename(csv_file).replace('_choice_results.csv', '')
            print(f"{experiment_name}: Error reading file - {e}")
            issues_found = True
    
    if not issues_found:
        print("No missing trials or false data detected - all experiments completed successfully!")
        print("\nSummary of all experiments:")
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty and 'trial' in df.columns:
                    experiment_name = os.path.basename(csv_file).replace('_choice_results.csv', '')
                    total_conversations = len(df)
                    unique_trials = df['trial'].nunique()
                    trial_counts = df['trial'].value_counts()
                    conversations_per_trial = trial_counts.iloc[0] if len(trial_counts) > 0 else 0
                    print(f"  {experiment_name}: {total_conversations} conversations, {unique_trials} trials, {conversations_per_trial} per trial")
            except:
                continue

def check_experiment_completeness(experiment_dir: str):
    """
    Check if an experiment directory has the expected number of conversations.
    
    Args:
        experiment_dir: Path to experiment directory
    """
    print(f"\n🔍 Checking experiment completeness: {experiment_dir}")
    
    # Load control and treatment data
    control_file = os.path.join(experiment_dir, "control.csv")
    treatment_file = os.path.join(experiment_dir, "treatment.csv")
    
    if not os.path.exists(control_file) or not os.path.exists(treatment_file):
        print("  ❌ Missing control.csv or treatment.csv")
        return
    
    try:
        control_df = pd.read_csv(control_file)
        treatment_df = pd.read_csv(treatment_file)
        
        control_trials = control_df['trial'].nunique() if 'trial' in control_df.columns else 0
        treatment_trials = treatment_df['trial'].nunique() if 'trial' in treatment_df.columns else 0
        
        print(f"  📊 Control trials: {control_trials}")
        print(f"  📊 Treatment trials: {treatment_trials}")
        
        # Expected conversations = control_trials * treatment_trials * 2 (for order effects)
        expected_conversations = control_trials * treatment_trials * 2
        print(f"  📊 Expected conversations: {expected_conversations}")
        
        # Check if results exist
        results_file = os.path.join("results_and_data/results", 
                                  os.path.basename(experiment_dir) + "_choice_results.csv")
        
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            actual_conversations = len(results_df)
            print(f"  📊 Actual conversations: {actual_conversations}")
            
            if actual_conversations < expected_conversations:
                missing = expected_conversations - actual_conversations
                print(f"  ⚠️  Missing {missing} conversations ({missing/expected_conversations*100:.1f}%)")
            else:
                print(f"  ✅ All conversations completed")
        else:
            print(f"  ❌ No results file found: {results_file}")
            
    except Exception as e:
        print(f"  ❌ Error checking experiment: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for missing trials in assist tag results")
    parser.add_argument("--results-dir", default="results_and_data/results",
                       help="Directory containing result CSV files")
    parser.add_argument("--experiment-dir", 
                       help="Check specific experiment directory")
    
    args = parser.parse_args()
    
    if args.experiment_dir:
        check_experiment_completeness(args.experiment_dir)
    else:
        check_missing_trials(args.results_dir)

if __name__ == "__main__":
    main()
