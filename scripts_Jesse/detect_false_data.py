#!/usr/bin/env python3
"""
Detect false data in assist tag recognition results due to rate limit errors.

This script identifies conversations that likely contain false data because
the model wrapper returned neutral logits when encountering API errors.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def detect_false_data(results_dir: str = "results_and_data/results"):
    """
    Detect conversations with false data due to rate limit errors.
    
    Args:
        results_dir: Directory containing result CSV files
    """
    # Find all result CSV files
    csv_files = glob.glob(os.path.join(results_dir, "**", "*_choice_results.csv"), recursive=True)
    
    if not csv_files:
        print("No result CSV files found!")
        return
    
    print("=== False Data Detection Report ===")
    print("Looking for conversations with identical probabilities (indicating neutral logits from rate limit errors)")
    print()
    
    total_false_conversations = 0
    experiments_with_false_data = 0
    
    for csv_file in csv_files:
        try:
            # Load the results
            df = pd.read_csv(csv_file)
            
            if df.empty:
                continue
            
            experiment_name = os.path.basename(csv_file).replace('_choice_results.csv', '')
            
            # Check for conversations with identical probabilities
            if 'prob_choice_1' in df.columns and 'prob_choice_2' in df.columns:
                identical_probs = df[df['prob_choice_1'] == df['prob_choice_2']]
                
                if len(identical_probs) > 0:
                    experiments_with_false_data += 1
                    total_false_conversations += len(identical_probs)
                    
                    print(f"📁 {csv_file}")
                    print(f"   False conversations: {len(identical_probs)}/{len(df)} ({len(identical_probs)/len(df)*100:.1f}%)")
                    
                    # Group by trial to show which trials are affected
                    if 'trial' in identical_probs.columns:
                        affected_trials = sorted(identical_probs['trial'].unique())
                        print(f"   Affected trials: {affected_trials}")
                        
                        # Show the identical probability values
                        unique_probs = identical_probs['prob_choice_1'].unique()
                        print(f"   Identical probability values: {unique_probs}")
                    
                    print()
            
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
    
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"  Experiments with false data: {experiments_with_false_data}/{len(csv_files)}")
    print(f"  Total false conversations: {total_false_conversations}")
    
    if total_false_conversations > 0:
        print(f"\n⚠️  WARNING: {total_false_conversations} conversations contain false data!")
        print("   These should be excluded from analysis or re-run.")
    else:
        print("\n✅ No false data detected - all experiments are clean!")

def get_false_data_conversations(results_dir: str = "results_and_data/results"):
    """
    Get a list of all conversation IDs that contain false data.
    
    Args:
        results_dir: Directory containing result CSV files
        
    Returns:
        List of conversation IDs with false data
    """
    csv_files = glob.glob(os.path.join(results_dir, "**", "*_choice_results.csv"), recursive=True)
    false_conversations = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty and 'prob_choice_1' in df.columns and 'prob_choice_2' in df.columns:
                identical_probs = df[df['prob_choice_1'] == df['prob_choice_2']]
                if 'conversation_id' in identical_probs.columns:
                    false_conversations.extend(identical_probs['conversation_id'].tolist())
        except:
            continue
    
    return false_conversations

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect false data in assist tag results")
    parser.add_argument("--results-dir", default="results_and_data/results/to_run",
                       help="Directory containing result CSV files")
    parser.add_argument("--list-false", action="store_true",
                       help="List all conversation IDs with false data")
    
    args = parser.parse_args()
    
    if args.list_false:
        false_convs = get_false_data_conversations(args.results_dir)
        print("Conversation IDs with false data:")
        for conv_id in false_convs:
            print(f"  {conv_id}")
    else:
        detect_false_data(args.results_dir)

if __name__ == "__main__":
    main()
