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
from prompts import DATASET_SYSTEM_PROMPTS, SUMMARIZE_PROMPT_TEMPLATES, STYLE_PROMPT_ADDENDUM
from utils.util import YamlConfig


def generate_summaries_for_split(
    chat_wrapper: ChatTemplateWrapper,
    split_data: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    max_generate: int,
    temps: List[float],
    num_trials: List[int],
    styles: List[str | None],
    run_name: str,
    use_lora: bool = False,
    lora_run_name: str = None,
    artifact_suffix: str = None,
    results_dir: Optional[str] = None,
    continue_mode: bool = False
) -> None:
    """
    Generate summaries for a single dataset split.
    
    Args:
        chat_wrapper: Loaded model wrapper
        split_data: DataFrame with columns [document_idx, article, summary]
        dataset_name: Name of dataset (for prompting)
        split_name: Name of split (test/validation/train)
        temps: List of temperatures to use
        num_trials: List of number of trials per temperature
        styles: List of styles to prompt with, keys to STYLE_SYSTEM_PROMPTS
        run_name: Name of the run (for saving)
        use_lora: Whether using LoRA adapters
        lora_run_name: WandB run name for LoRA
        artifact_suffix: Artifact suffix for LoRA
        continue_mode: Whether to continue from existing results
    """
    # Determine base directory based on whether using LoRA
    results_dir = results_dir or 'results_and_data/results'
    if use_lora:
        base_dir = f"{results_dir}/main/{run_name}/{split_name}/forward_sft_summaries/{lora_run_name}/{artifact_suffix}"
        print(f"Using LoRA model - saving to: {base_dir}")
    else:
        base_dir = f"{results_dir}/main/{run_name}/{split_name}/model_summaries"
        print(f"Using base model - saving to: {base_dir}")
    
    # Create output directories
    os.makedirs(base_dir, exist_ok=True)

    # Build list of all expected CSV files
    expected_csv_files = []
    for temp, num_trial, style in zip(temps, num_trials, styles):
        for trial_idx in range(num_trial):
            csv_path = f"{base_dir}/T{temp}_trial{trial_idx}_style{style}.csv"
            expected_csv_files.append(csv_path)
    
    completion_status = {}  # {csv_path: set(completed_document_idx)}
    
    if continue_mode:
        print("Continue mode: Checking existing CSV files...")
        
        # Verify all expected CSVs exist and validate their structure
        for csv_path in expected_csv_files:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Continue mode requires existing CSV, but not found: {csv_path}")
            
            # Read and validate CSV structure
            df = pd.read_csv(csv_path)
            if list(df.columns) != ['document_idx', 'summary']:
                raise ValueError(f"Invalid columns in {csv_path}. Expected ['document_idx', 'summary'], got {list(df.columns)}")
            
            completion_status[csv_path] = set(df['document_idx'])
        
        # Check if all documents are already completed
        all_expected_docs = set(range(max_generate))
        if all(all_expected_docs.issubset(completed_docs) for completed_docs in completion_status.values()):
            raise RuntimeError(f"All documents up to max_generate ({max_generate}) are already completed - nothing to continue")
        
        print(f"Continue mode: Found existing results, will skip already completed documents")
        
    else:
        print("Creating new CSV files...")
        # Create blank CSV files (overwrite existing)
        for csv_path in expected_csv_files:
            pd.DataFrame(columns=['document_idx', 'summary']).to_csv(csv_path, index=False)
            completion_status[csv_path] = set()  # Empty set for new files

    # Get system prompt and user prompt template
    system_prompt = DATASET_SYSTEM_PROMPTS[dataset_name]
    user_prompt_template = SUMMARIZE_PROMPT_TEMPLATES[dataset_name]
    
    for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Processing {split_name}"):
        
        if idx == max_generate:
            break

        document_idx = row['document_idx']
        article = row['article']

        # Check if this document is already completed in ALL CSVs
        if continue_mode and all(document_idx in completion_status[csv_path] for csv_path in expected_csv_files):
            continue  # Skip this document entirely

        # Format user message
        user_message = user_prompt_template.format(article=article)
        
        try:
            # Create cache for this document (don't close user tags in case we want to add style instructions)
            cache_info = chat_wrapper.create_prompt_cache(
                system_prompt=system_prompt,
                user_message=user_message,
                user_message_unfinished=True
            )
        except torch.OutOfMemoryError:
            continue

        # Generate for each temp/trial combination
        for temp, num_trial, style in zip(temps, num_trials, styles):
            
            for trial_idx in range(num_trial):
            
                output_file = f"{base_dir}/T{temp}_trial{trial_idx}_style{style}.csv"
                
                # Skip if this specific CSV already has this document_idx
                if continue_mode and document_idx in completion_status[output_file]:
                    continue

                # FIXME This should be hidden away please.
                extra_chat = "-" if style == 'natural' else f'-\n\n{STYLE_PROMPT_ADDENDUM[style]}'
                extra_chat_with_tags = chat_wrapper.format_chat(user_message = extra_chat, prefiller="").removeprefix(chat_wrapper.format_chat(user_message="-", user_message_unfinished=True))

                # Generate summary with empty chat (continues from cache)
                generation_result = chat_wrapper.generate(
                    chats=[extra_chat_with_tags],  # Empty string starts from cache point
                    past_key_values=copy.deepcopy(cache_info["cache"]),
                    past_key_values_str=cache_info["formatted_prompt"],
                    max_new_tokens=1024,
                    temperature=temp,
                    do_sample=(temp > 0.0),
                    use_cache_position = False,
                    skip_special_tokens = True,
                    return_full_text = False,
                )
                
                summary = generation_result["generated_texts"][0].strip().replace('\n', '\\n')
                
                # Save to appropriate trial file
                result_row = pd.DataFrame({
                    'document_idx': [document_idx],
                    'summary': [summary]
                })
                result_row.to_csv(output_file, mode='a', header=False, index=False)
                
                # Update completion status
                completion_status[output_file].add(document_idx)


if __name__ == "__main__":

    # Parse command line arguments
    continue_mode = len(sys.argv) > 1 and sys.argv[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(sys.argv) - (1 if continue_mode else 0)
    
    if effective_argc not in [2, 4]:
        print("Usage:")
        print("  Base model: python -m generate_hf.py /path/to/yaml/args.yaml [continue]")
        print("  With LoRA:  python -m generate_hf.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_suffix> [continue]")
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
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_data, test_data, validation_data = load_dataset(args.dataset, splits=list(args.splits.__dict__.keys()))
    
    # Map split names to data
    split_data_map = {
        'train': train_data,
        'test': test_data,
        'validation': validation_data
    }
    
    # Generate summaries for each requested split
    for split_name, max_generate in args.splits.__dict__.items():
            
        split_data = split_data_map[split_name]
        print(f"Generating summaries for {split_name} split ({len(split_data)} documents)")
        
        generate_summaries_for_split(
            chat_wrapper=chat_wrapper,
            split_data=split_data,
            max_generate=max_generate,
            dataset_name=args.dataset,
            split_name=split_name,
            temps=args.temps,
            num_trials=args.num_trials,
            styles=args.styles,
            run_name=args.args_name,
            use_lora=use_lora,
            lora_run_name=wandb_run_name,
            artifact_suffix=artifact_suffix,
            continue_mode=continue_mode
        )
    
    print("Generation complete!")