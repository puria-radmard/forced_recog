import pandas as pd
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Dict
import sys
import random
import wandb

from load_data import load_dataset, load_model_summaries
from model.load import load_model
from model.base import ChatTemplateWrapper

from sft_utils.lora import setup_lora_model, save_lora_as_artifact
from sft_utils.train import parse_forward_sft_key

from prompts import (
    DETECTION_SYSTEM_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT, 
    DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY
)

from utils.util import YamlConfig


def load_summaries_for_backward_sft(
    run_name: str,
    split_name: str, 
    temps: List[float],
    num_trials: List[int],
    styles: List[str],
    results_dir: Optional[str] = None
) -> Dict[Tuple, pd.DataFrame]:
    """Load all summaries for backward SFT training."""
    all_summaries = {}
    results_dir = results_dir or 'results_and_data/results'
    
    for temp, num_trial, style in zip(temps, num_trials, styles):
        for trial_idx in range(num_trial):
            try:
                summary_df = load_model_summaries(run_name, split_name, temp, trial_idx, style, results_dir=results_dir)
                all_summaries[(temp, trial_idx, style)] = summary_df
                print(f"Loaded summaries for temp={temp}, trial={trial_idx}, style={style}")
            except FileNotFoundError:
                print(f"Warning: Missing summary file for T={temp}, style={style}, trial={trial_idx}")
                continue
    
    return all_summaries


def create_backward_training_pairs(
    split_data: pd.DataFrame,
    all_summaries: Dict[Tuple, pd.DataFrame],
    target_style: str,
    target_temp: str,
    chat_wrapper: ChatTemplateWrapper,
    temps: List[float],
    num_trials: List[int], 
    styles: List[str]
) -> List[dict]:
    """
    Create training pairs for backward SFT.
    Target is always the choice that selects the target_style summary.
    """
    training_pairs = []
    
    print(f"Creating backward SFT training pairs with target style: {target_style}")

    all_documents_idxs = set()
    for v in all_summaries.values():
        all_documents_idxs = all_documents_idxs.union(set(v.document_idx.tolist()))
    
    covered_split_data = split_data[split_data['document_idx'].isin(all_documents_idxs)]    

    for idx, row in tqdm(
        covered_split_data.iterrows(), total=len(covered_split_data),
        desc="Creating training pairs from documents that all styles have summarised"
    ):
        document_idx = row['document_idx']
        article = row['article']
        
        # Load all summaries for this document
        summaries = {}
        for temp, num_trial, style in zip(temps, num_trials, styles):
            for trial_idx in range(num_trial):
                key = (temp, trial_idx, style)
                if key in all_summaries:
                    summary_df = all_summaries[key]
                    doc_summary = summary_df[summary_df['document_idx'] == document_idx]
                    assert len(doc_summary) == 1
                    summaries[key] = doc_summary['summary'].iloc[0]
        
        if len(summaries) == 0:
            continue
        
        # Get all keys for target style and non-target styles
        target_keys = [key for key in summaries.keys() if (key[0] == target_temp) and (key[2] == target_style)]
        non_target_keys = [key for key in summaries.keys() if (key[0] != target_temp) or (key[2] != target_style)]
        
        if len(target_keys) == 0 or len(non_target_keys) == 0:
            continue
        
        # Create pairwise comparisons: target vs non-target
        for target_key in target_keys:
            for non_target_key in non_target_keys:
                target_summary = summaries[target_key]
                non_target_summary = summaries[non_target_key]
                
                # Case 1: target_style first, non_target second -> answer should be "1"
                prompt_1 = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(
                        summary_1=target_summary, 
                        summary_2=non_target_summary
                    ) + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                formatted_input_1 = chat_wrapper.format_chat(
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) + prompt_1,
                    prefiller=""
                )
                
                training_pairs.append({
                    "input": formatted_input_1,
                    "target": "1",  # Target style is in position 1
                    "document_idx": document_idx,
                    "target_key": target_key,
                    "non_target_key": non_target_key,
                })
                
                # Case 2: non_target first, target_style second -> answer should be "2"  
                prompt_2 = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(
                        summary_1=non_target_summary,
                        summary_2=target_summary
                    ) + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                formatted_input_2 = chat_wrapper.format_chat(
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) + prompt_2,
                    prefiller=""
                )
                
                training_pairs.append({
                    "input": formatted_input_2,
                    "target": "2",  # Target style is in position 2
                    "document_idx": document_idx,
                    "target_key": target_key,
                    "non_target_key": non_target_key,
                })

    print(f"Created {len(training_pairs)} training pairs for target style '{target_style}'")
    return training_pairs



def train_step_backward(
    model,
    tokenizer, 
    batch_pairs: List[dict],
    max_seq_length: int,
    device: str
) -> torch.Tensor:
    """Training step for backward SFT - simplified to match train_step_forward pattern."""
    """Training step for backward SFT - tokenize inputs and targets separately then concatenate."""
    assert len(batch_pairs) == 1, "Currently only supports batch_size=1"
    
    model.train()
    
    pair = batch_pairs[0]
    input_text = pair["input"]
    target_text = pair["target"]  # "1" or "2"
    
    # Tokenize input and target separately
    input_tokens = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
    target_tokens = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")
    
    # Get input length for masking
    input_length = input_tokens['input_ids'].shape[1]
    
    # Concatenate token IDs
    full_input_ids = torch.cat([
        input_tokens['input_ids'], 
        target_tokens['input_ids']
    ], dim=1)
    
    # Create attention mask (all 1s since no padding)
    attention_mask = torch.ones_like(full_input_ids)
    
    # Move to device
    input_ids = full_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Create labels (mask input portion)
    labels = input_ids.clone()
    labels[:, :input_length] = -100  # Ignore loss on input tokens
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    return outputs.loss



def run_backward_sft(
    args_name: str,
    dataset: str,
    model_name: str,
    target_style: str,
    target_temp: float,
    temps: List[float],
    num_trials: List[int],
    styles: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    max_seq_length: int,
    max_steps: int,
    save_frequency: int,
    seed_num: int = 42,
    results_dir: Optional[str] = None
):
    """Run backward SFT training locally."""
    
    # Set seeds for reproducibility
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_num)

    # Get WandB project name from Modal Secret
    project_name = os.getenv('WANDB_PROJECT')
    if not project_name:
        raise ValueError("WANDB_PROJECT environment variable not set")
    
    
    results_dir = results_dir or 'results_and_data/results'
    
    # Create WandB run name
    run_name = f"{args_name}_backwardsft_temp{target_temp}_style{target_style}_seed{seed_num}"
    
    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device='auto')
    device = chat_wrapper.model.device
    
    # Load dataset
    print(f"Loading dataset: {dataset}")
    train_data, _, _ = load_dataset(dataset, splits=['train'])
    
    # Load all summaries
    print("Loading all summaries...")
    all_summaries = load_summaries_for_backward_sft(
        args_name, 'train', temps, num_trials, styles, results_dir
    )
    
    # Create training pairs
    print("Creating backward training pairs...")
    training_pairs = create_backward_training_pairs(
        train_data, all_summaries, target_style, target_temp, chat_wrapper,
        temps, num_trials, styles
    )
    
    if len(training_pairs) == 0:
        raise ValueError("No training pairs created!")
    
    # Setup LoRA
    print("Setting up LoRA...")
    sft_config_dict = {
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'target_modules': target_modules,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'max_seq_length': max_seq_length,
        'max_steps': max_steps,
        'save_frequency': save_frequency
    }
    lora_model = setup_lora_model(chat_wrapper.model, sft_config_dict)
    trainable_params = lora_model.num_parameters(only_trainable=True)
    print(f"LoRA model setup complete. Trainable parameters: {trainable_params}")
    
    # Training setup
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
    
    # Initialize WandB
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'run_name': args_name,
            'dataset': dataset,
            'model_name': model_name,
            'target_style': target_style,
            'target_temp': target_temp,
            'seed': seed_num,
            **sft_config_dict
        }
    )
    
    wandb.log({
        "total_examples": len(training_pairs),
        "trainable_parameters": trainable_params
    })
    
    # Training
    indices = list(range(len(training_pairs)))
    print(f"Starting training: {num_epochs} epochs, {len(training_pairs)} examples, batch size {batch_size}")
    
    step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle for this epoch
        random.shuffle(indices)
        
        # Batch training
        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i+batch_size]
            batch_pairs = [training_pairs[idx] for idx in batch_indices]

            # Training step
            optimizer.zero_grad()
            loss = train_step_backward(
                lora_model, 
                chat_wrapper.tokenizer, 
                batch_pairs,
                max_seq_length,
                device
            )
            loss.backward()
            optimizer.step()
            
            step += 1
            current_loss = loss.item()
            
            # Log loss to WandB every step
            wandb.log({
                "loss": current_loss,
                "step": step,
                "epoch": epoch + 1
            })
            
            # Save LoRA adapters periodically
            if step % save_frequency == 0:
                print(f"Step {step}: Loss = {current_loss:.4f}")
                # You can implement save_lora_as_artifact for local saving
                # save_lora_as_artifact(lora_model, step, target_style, run_name)
            
            if (max_steps is not None) and (step >= max_steps):
                break
        
        if (max_steps is not None) and (step >= max_steps):
            break
    
    print(f"Training complete! Total steps: {step}")
    wandb.finish()
    
    return {
        "run_name": run_name,
        "total_steps": step,
        "lora_model": lora_model
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python backward_sft.py /path/to/yaml/args.yaml <target_temp_and_style> [seed]. Currently detected {len(sys.argv)} args")
        sys.exit(1)
    
    config_path = sys.argv[1]
    sft_key = sys.argv[2]  # e.g., "temp0.7_styleconcise"
    seed_num = int(sys.argv[3])

    target_temp, target_style = parse_forward_sft_key(sft_key)

    args = YamlConfig(config_path)
    
    # Get backward SFT configuration for the specific target style
    sft_config_key = f"backwardsft_temp{target_temp}_style{target_style}"
    if not hasattr(args, sft_config_key):
        raise ValueError(f"Backward SFT configuration key '{sft_config_key}' not found in yaml")
    
    sft_config = getattr(args, sft_config_key).__dict__
    
    print("Starting backward SFT training...")
    print(f"Config: {config_path}")
    print(f"Target style: {target_style}")
    print(f"Args name: {args.args_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Seed: {seed_num}")
    
    result = run_backward_sft(
        args_name=args.args_name,
        dataset=args.dataset,
        model_name=args.model_name,
        target_style=target_style,
        target_temp=target_temp,
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        lora_r=sft_config['lora_r'],
        lora_alpha=sft_config['lora_alpha'],
        lora_dropout=sft_config['lora_dropout'],
        target_modules=sft_config['target_modules'],
        learning_rate=sft_config['learning_rate'],
        num_epochs=sft_config['num_epochs'],
        batch_size=sft_config['batch_size'],
        max_seq_length=sft_config['max_seq_length'],
        max_steps=sft_config['max_steps'],
        save_frequency=sft_config['save_frequency'],
        seed_num=seed_num
    )
    
    print(f"Training completed!")
    print(f"Run name: {result['run_name']}")
    print(f"Total steps: {result['total_steps']}")