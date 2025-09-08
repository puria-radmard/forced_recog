import pandas as pd
import os
import torch
import tempfile
from tqdm import tqdm
from typing import Optional
import sys

from dotenv import load_dotenv

from load_data import load_dataset
from model.load import load_model
from model.base import ChatTemplateWrapper
from prompts import DATASET_SYSTEM_PROMPTS, SUMMARIZE_PROMPT_TEMPLATES
from utils.util import YamlConfig

from sft_utils.lora import download_and_apply_lora


def generate_summaries_with_contrastive_decoding(
    chat_wrapper: ChatTemplateWrapper,
    split_data: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    max_generate: int,
    run_name: str,
    lora_run_name: str,
    artifact_suffix: str,
    results_dir: Optional[str] = None,
    alpha: float = 1.0,
    continue_mode: bool = False
) -> None:
    """
    Generate summaries using contrastive decoding between LoRA and base model.
    
    Args:
        chat_wrapper: Loaded base model wrapper
        split_data: DataFrame with columns [document_idx, article, summary]
        dataset_name: Name of dataset (for prompting)
        split_name: Name of split (test/validation/train)
        max_generate: Maximum number of documents to process
        run_name: Name of the run (for saving)
        lora_run_name: WandB run name for LoRA
        artifact_suffix: Artifact suffix for LoRA
        lora_adapter_path: Local path to downloaded LoRA adapters
        alpha: Contrastive strength (1.0 = full contrast)
        continue_mode: Whether to continue from existing results
    """
    
    # Set up output directory
    results_dir = results_dir or 'results_and_data/modal_results/results'
    base_dir = f"{results_dir}/main/{run_name}/{split_name}/contrastive_summaries/{lora_run_name}/{artifact_suffix}"
    print(f"Saving contrastive generation results to: {base_dir}")
    
    # Create output directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Define output file
    output_file = f"{base_dir}/contrastive_summaries.csv"
    
    completion_status = set()
    
    if continue_mode:
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Continue mode requires existing CSV, but not found: {output_file}")
        
        # Read existing results
        df = pd.read_csv(output_file)
        if list(df.columns) != ['document_idx', 'summary']:
            raise ValueError(f"Invalid columns in {output_file}. Expected ['document_idx', 'summary'], got {list(df.columns)}")
        
        completion_status = set(df['document_idx'])
        
        # Check if all documents are completed
        all_expected_docs = set(range(max_generate))
        if all_expected_docs.issubset(completion_status):
            raise RuntimeError(f"All documents up to max_generate ({max_generate}) are already completed - nothing to continue")
        
        print(f"Continue mode: Found existing results, will skip {len(completion_status)} completed documents")
    else:
        # Create new CSV file
        pd.DataFrame(columns=['document_idx', 'summary']).to_csv(output_file, index=False)
        print("Created new CSV file for contrastive generation")

    # Get system prompt and user prompt template
    system_prompt = DATASET_SYSTEM_PROMPTS[dataset_name]
    user_prompt_template = SUMMARIZE_PROMPT_TEMPLATES[dataset_name]

    
    for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Contrastive generation for {split_name}"):
        
        if idx == max_generate:
            break

        document_idx = row['document_idx']
        
        # Skip if already completed
        if continue_mode and document_idx in completion_status:
            continue

        article = row['article']
        
        # Format user message (natural style only)
        user_message = user_prompt_template.format(article=article)
        
        try:
            # Generate summary using contrastive decoding
            summary = contrastive_generate_summary_cached(
                chat_wrapper=chat_wrapper,
                system_prompt=system_prompt,
                user_message=user_message,
                alpha=alpha
            )
            
            # Save result
            result_row = pd.DataFrame({
                'document_idx': [document_idx],
                'summary': [summary]
            })
            result_row.to_csv(output_file, mode='a', header=False, index=False)
            completion_status.add(document_idx)
            
        except torch.OutOfMemoryError:
            print(f"OOM error for document {document_idx}, skipping...")
            continue



def contrastive_generate_summary_cached(
    chat_wrapper: ChatTemplateWrapper,
    system_prompt: str,
    user_message: str,
    alpha: float = 1.0,
    max_new_tokens: int = 1024
) -> str:
    """
    Generate a summary using contrastive decoding with KV caching for speed.
    
    Args:
        chat_wrapper: Base model wrapper
        lora_adapter_name: Name of LoRA adapter
        system_prompt: System prompt for the task
        user_message: User message containing the article
        alpha: Contrastive strength
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated summary string
    """
    
    # Format the full prompt
    full_prompt = chat_wrapper.format_chat(
        system_prompt=system_prompt,
        user_message=user_message
    )
    
    # Tokenize the prompt
    tokenizer = chat_wrapper.tokenizer
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(chat_wrapper.model.device)
    
    generated_tokens = []
    generated_text = ""
    
    # Initialize caches
    lora_cache = None
    base_cache = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                # First step: process full input sequence
                current_input = input_ids
            else:
                # Subsequent steps: only process the new token
                current_input = torch.tensor([[generated_tokens[-1]]], 
                                           device=chat_wrapper.model.device)
            
            # Get LoRA logits with caching
            lora_outputs = chat_wrapper.model(
                current_input, 
                past_key_values=lora_cache,
                use_cache=True
            )
            lora_logits = lora_outputs.logits[0, -1, :]
            lora_probs = torch.softmax(lora_logits, dim=-1)
            lora_cache = lora_outputs.past_key_values  # Update cache
            
            # Get base model logits with caching (adapter disabled)
            with chat_wrapper.model.disable_adapter():
                base_outputs = chat_wrapper.model(
                    current_input,
                    past_key_values=base_cache,
                    use_cache=True
                )
                base_logits = base_outputs.logits[0, -1, :]
                base_probs = torch.softmax(base_logits, dim=-1)
                base_cache = base_outputs.past_key_values  # Update cache
            
            # Contrastive sampling
            contrastive_probs = lora_probs - alpha * base_probs
            next_token = torch.argmax(contrastive_probs, dim=-1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            
            # Decode and accumulate text
            token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            generated_text += token_text
    
    return generated_text.strip().replace('\n', '\\n')

if __name__ == "__main__":
    # Parse command line arguments - only LoRA mode supported
    continue_mode = len(sys.argv) > 1 and sys.argv[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(sys.argv) - (1 if continue_mode else 0)

    load_dotenv()
    
    if effective_argc != 4:
        print("Usage:")
        print("  python contrastive_generate.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_suffix> [continue]")
        print("  Note: This script only supports LoRA-based contrastive generation")
        sys.exit(1)
    
    config_path = sys.argv[1]
    wandb_run_name = sys.argv[2]
    artifact_suffix = sys.argv[3]
    
    print(f"Running contrastive generation with LoRA:")
    print(f"  WandB Run: {wandb_run_name}")
    print(f"  Artifact Suffix: {artifact_suffix}")
    
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
    
    args = YamlConfig(config_path)
    
    # Load base model
    print(f"Loading base model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    # Download LoRA adapters once and get the local path
    chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_suffix, adapter_name="trained_adapter")
    
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

        if split_name != "test":
            continue
        
        split_data = split_data_map[split_name]
        print(f"Generating contrastive summaries for {split_name} split ({len(split_data)} documents)")
        
        generate_summaries_with_contrastive_decoding(
            chat_wrapper=chat_wrapper,
            split_data=split_data,
            max_generate=max_generate,
            dataset_name=args.dataset,
            split_name=split_name,
            run_name=args.args_name,
            lora_run_name=wandb_run_name,
            artifact_suffix=artifact_suffix,
            continue_mode=continue_mode
        )

    print("Contrastive generation complete!")