import pandas as pd
import os
import torch
from tqdm import tqdm
from typing import List, Optional
import sys
import copy

from sft_utils.lora import download_and_apply_lora

from load_data import load_dataset
from model.load import load_model
from model.base import ChatTemplateWrapper
from utils.util import YamlConfig

from prompts.strategy import SYSTEM_PROMPTS


@torch.no_grad()
def generate_advice_for_split(
    chat_wrapper: ChatTemplateWrapper,
    split_data: pd.DataFrame,
    split_name: str,
    max_generate: int,
    strategies: List[str],
    styles: List[str],
    run_name: str,
    use_lora: bool = False,
    lora_run_name: str = None,
    artifact_suffix: str = None,
    results_dir: Optional[str] = None,
    continue_mode: bool = False
) -> None:
    """
    Generate advice responses for a single dataset split.
    
    Args:
        chat_wrapper: Loaded model wrapper
        split_data: DataFrame with columns [question_idx, question] (or similar)
        split_name: Name of split (test/validation/train)
        max_generate: Maximum number of questions to process
        strategies: List of strategies ['prorisk', 'antirisk']
        styles: List of styles ['formal', 'casual']
        run_name: Name of the run (for saving)
        use_lora: Whether using LoRA adapters
        lora_run_name: WandB run name for LoRA
        artifact_suffix: Artifact suffix for LoRA
        continue_mode: Whether to continue from existing results
    """
    # Determine base directory based on whether using LoRA
    if use_lora:
        base_dir = f"{results_dir}/strategy/{run_name}/{split_name}/sfted_advice/{lora_run_name}/{artifact_suffix}"
        print(f"Using LoRA model - saving to: {base_dir}")
    else:
        base_dir = f"{results_dir}/strategy/{run_name}/{split_name}/model_advice"
        print(f"Using base model - saving to: {base_dir}")
    
    # Create output directories
    os.makedirs(base_dir, exist_ok=True)

    # Build list of all expected CSV files
    expected_csv_files = []
    for strategy in strategies:
        for style in styles:
            csv_path = f"{base_dir}/strategy_{strategy}_style_{style}.csv"
            expected_csv_files.append(csv_path)
    
    completion_status = {}  # {csv_path: set(completed_question_idx)}
    
    if continue_mode:
        print("Continue mode: Checking existing CSV files...")
        
        # Verify all expected CSVs exist and validate their structure
        for csv_path in expected_csv_files:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Continue mode requires existing CSV, but not found: {csv_path}")
            
            # Read and validate CSV structure
            df = pd.read_csv(csv_path)
            if list(df.columns) != ['question_idx', 'response']:
                raise ValueError(f"Invalid columns in {csv_path}. Expected ['question_idx', 'response'], got {list(df.columns)}")
            
            completion_status[csv_path] = set(df['question_idx'])
        
        # Check if all questions are already completed
        all_expected_questions = set(range(max_generate))
        if all(all_expected_questions.issubset(completed_questions) for completed_questions in completion_status.values()):
            raise RuntimeError(f"All questions up to max_generate ({max_generate}) are already completed - nothing to continue")
        
        print(f"Continue mode: Found existing results, will skip already completed questions")
        
    else:
        print("Creating new CSV files...")
        # Create blank CSV files (overwrite existing)
        for csv_path in expected_csv_files:
            pd.DataFrame(columns=['question_idx', 'response']).to_csv(csv_path, index=False)
            completion_status[csv_path] = set()  # Empty set for new files

    for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Processing {split_name}"):
        
        if idx == max_generate:
            break

        question_idx = row['question_idx'] if 'question_idx' in row else idx
        question = row['question'] if 'question' in row else row[split_data.columns[1]]  # Flexible column handling

        # Check if this question is already completed in ALL CSVs
        if continue_mode and all(question_idx in completion_status[csv_path] for csv_path in expected_csv_files):
            continue  # Skip this question entirely

        # Generate for each strategy/style combination
        for strategy in strategies:
            for style in styles:
                
                output_file = f"{base_dir}/strategy_{strategy}_style_{style}.csv"
                
                # Skip if this specific CSV already has this question_idx
                if continue_mode and question_idx in completion_status[output_file]:
                    continue

                # Get system prompt for this strategy/style combination
                system_prompt = SYSTEM_PROMPTS[(strategy, style)]

                chat = chat_wrapper.format_chat(
                    system_prompt=system_prompt,
                    user_message=question,
                    prefiller=""
                )

                # Generate response
                generation_result = chat_wrapper.generate(
                    chats=[chat],
                    temperature=None,
                    do_sample=False,
                )
                
                response = generation_result["generated_texts"][0].strip().replace('\n', '\\n')
                
                # Save to appropriate file
                result_row = pd.DataFrame({
                    'question_idx': [question_idx],
                    'response': [response]
                })
                result_row.to_csv(output_file, mode='a', header=False, index=False)
                
                # Update completion status
                completion_status[output_file].add(question_idx)


if __name__ == "__main__":

    # Parse command line arguments
    continue_mode = len(sys.argv) > 1 and sys.argv[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(sys.argv) - (1 if continue_mode else 0)
    
    if effective_argc not in [2, 4]:
        print("Usage:")
        print("  Base model: python -m scripts_advice.generate_advice.generate_hf.py /path/to/yaml/args.yaml [continue]")
        print("  With LoRA:  python -m scripts_advice.generate_advice.generate_hf.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_suffix> [continue]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    use_lora = effective_argc == 4
    
    if use_lora:
        wandb_run_name = sys.argv[2]
        artifact_suffix = sys.argv[3]
        print(f"Running with LoRA adapters:")
        print(f"  WandB Run: {wandb_run_name}")
        print(f"  Artifact Suffix: {artifact_suffix}")
    else:
        wandb_run_name = None
        artifact_suffix = None
        print("Running with base model")
    
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
    
    args = YamlConfig(config_path)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_suffix)
    
    # Load dataset (assuming CSV files with questions)
    print(f"Loading dataset: {args.dataset}")
    train_data, test_data, validation_data = load_dataset(args.dataset, splits=list(args.splits.__dict__.keys()))
    
    # Map split names to data
    split_data_map = {
        'train': train_data,
        'test': test_data,
        'validation': validation_data
    }
    
    # Generate advice for each requested split
    for split_name, max_generate in args.splits.__dict__.items():
            
        split_data = split_data_map[split_name]
        print(f"Generating advice for {split_name} split ({len(split_data)} questions)")
        
        generate_advice_for_split(
            chat_wrapper=chat_wrapper,
            split_data=split_data,
            max_generate=max_generate,
            split_name=split_name,
            strategies=args.strategies,  # ['prorisk', 'antirisk']
            styles=args.styles,          # ['formal', 'casual']
            run_name=args.args_name,
            use_lora=use_lora,
            lora_run_name=wandb_run_name,
            artifact_suffix=artifact_suffix,
            continue_mode=continue_mode,
            results_dir='results_and_data/modal_results/results'
        )
    
    print("Generation complete!")