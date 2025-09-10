#!/usr/bin/env python3

import os
import yaml
from pathlib import Path

def copy_lora_args_to_steps(source_file: str, target_steps: list):
    """
    Copy lora args file to different step numbers, updating paths and step references.
    
    Args:
        source_file: Path to the source choice_args.yaml file
        target_steps: List of step numbers to copy to
    """
    
    # Load the source file
    with open(source_file, 'r') as f:
        config = yaml.safe_load(f)
    
    source_path = Path(source_file)
    base_dir = source_path.parent.parent  # Go up two levels from pos_economist_600/choice_args.yaml
    
    print(f"Source file: {source_file}")
    print(f"Base directory: {base_dir}")
    print(f"Target steps: {target_steps}")
    
    # Extract current step from source directory name
    source_dir_name = source_path.parent.name  # pos_economist_600
    source_step = source_dir_name.split('_')[-1]  # 600
    
    print(f"Source step: {source_step}")
    print(f"Source config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    for target_step in target_steps:
        # Create new directory path
        new_dir_name = source_dir_name.replace(f"_{source_step}", f"_{target_step}")
        new_dir_path = base_dir / new_dir_name
        new_file_path = new_dir_path / "choice_args.yaml"
        
        # Create directory if it doesn't exist
        new_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Update config for this step
        new_config = config.copy()
        
        # Update positive and negative artifact names (replace step number)
        new_config['positive_artifact_name'] = config['positive_artifact_name'].replace(
            f"step_{source_step}", f"step_{target_step}"
        )
        new_config['negative_artifact_name'] = config['negative_artifact_name'].replace(
            f"step_{source_step}", f"step_{target_step}"
        )
        
        # Judge settings stay the same (no changes needed)
        
        # Write new config file
        with open(new_file_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        print(f"\nCreated: {new_file_path}")
        print(f"  positive_artifact_name: {new_config['positive_artifact_name']}")
        print(f"  negative_artifact_name: {new_config['negative_artifact_name']}")
        print(f"  judge_run_name: {new_config['judge_run_name']} (unchanged)")
        print(f"  judge_artifact_name: {new_config['judge_artifact_name']} (unchanged)")

if __name__ == "__main__":
    # source_file1 = "results_and_data/modal_results/results/main/mistral24b_forward_training/test/contrastive_summary_choices/pos_sun_600/choice_args.yaml"
    source_file2 = "results_and_data/modal_results/results/main/mistral24b_forward_training/test/contrastive_summary_choices/pos_economist_600/choice_args.yaml"
    target_steps = [30, 60, 90]
        
    for source_file in [source_file2]:
        if not os.path.exists(source_file):
            print(f"ERROR: Source file not found: {source_file}")
            exit(1)
        
        copy_lora_args_to_steps(source_file, target_steps)
        print("\nDone! All files created successfully.")