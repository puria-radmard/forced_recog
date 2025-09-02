import pandas as pd
import os
import sys
import torch
from peft import LoraConfig, get_peft_model, TaskType
import random
from tqdm import tqdm
from typing import List, Dict
import wandb
from datetime import datetime
from dotenv import load_dotenv

from load_data import load_dataset
from model.load import load_model
from prompts import DATASET_SYSTEM_PROMPTS, SUMMARIZE_PROMPT_TEMPLATES
from utils.util import YamlConfig

from sft_utils.lora import setup_lora_model, save_lora_as_artifact


def load_sft_data(run_name: str, dataset_name: str, temp: float, style: str) -> pd.DataFrame:
    """
    Load all training data for SFT from generated summaries.
    
    Args:
        run_name: Name of the run
        dataset_name: Name of dataset
        temp: Temperature used for generation
        style: Style used for generation
        
    Returns:
        DataFrame with [document_idx, article, summary, trial_idx]
    """
    # Load original train data to get articles
    train_data, _, _ = load_dataset(dataset_name)
    
    # Load all trial summaries for this temp/style combination
    base_dir = f"results_and_data/results/main/{run_name}/train/model_summaries"
    
    all_summaries = []
    trial_idx = 0
    while True:
        summary_file = f"{base_dir}/T{temp}_trial{trial_idx}_style{style}.csv"
        if not os.path.exists(summary_file):
            break
            
        print(f"Loading {summary_file}")
        trial_summaries = pd.read_csv(summary_file)
        trial_summaries['trial_idx'] = trial_idx
        all_summaries.append(trial_summaries)
        trial_idx += 1
    
    if not all_summaries:
        raise ValueError(f"No summary files found for temp={temp}, style={style}")
    
    print(f"Found {len(all_summaries)} trial files")
    
    # Combine all trials
    combined_summaries = pd.concat(all_summaries, ignore_index=True)
    
    # Merge with original articles
    sft_data = train_data.merge(combined_summaries, on='document_idx', how='inner')
    
    print(f"Combined data: {len(sft_data)} training examples from {len(all_summaries)} trials")
    
    return sft_data


def create_training_pairs(sft_data: pd.DataFrame, dataset_name: str, 
                         chat_wrapper) -> List[Dict]:
    """
    Create input-target pairs for SFT.
    
    Args:
        sft_data: DataFrame with articles and summaries
        dataset_name: Name of dataset for prompting
        chat_wrapper: Chat wrapper for formatting
        
    Returns:
        List of {"input": formatted_prompt, "target": summary, "document_idx": int}
    """
    system_prompt = DATASET_SYSTEM_PROMPTS[dataset_name]
    user_prompt_template = SUMMARIZE_PROMPT_TEMPLATES[dataset_name]
    
    training_pairs = []
    
    for _, row in tqdm(sft_data.iterrows(), total=len(sft_data), desc="Creating training pairs"):
        # Format input (no style addendum - just the base prompt)
        user_message = user_prompt_template.format(article=row['article'])
        formatted_input = chat_wrapper.format_chat(
            system_prompt=system_prompt,
            user_message=user_message,
            prefiller=""
        )

        # Target is just the summary (already cleaned in generation script)
        target = row['summary_y']
        
        training_pairs.append({
            "input": formatted_input,
            "target": target,
            "document_idx": row['document_idx'],
            "trial_idx": row['trial_idx']
        })
    
    return training_pairs


def train_step(model, tokenizer, batch_pairs, max_seq_length, device):
    """Perform one training step."""
    model.train()
    
    # Tokenize inputs and targets separately first to get input lengths
    input_texts = [pair["input"] for pair in batch_pairs]
    target_texts = [pair["target"] for pair in batch_pairs]
    
    # Get input lengths for masking (before adding targets)
    input_lengths = []
    for inp in input_texts:
        input_tokens = tokenizer(inp, add_special_tokens=False, return_tensors="pt")
        input_lengths.append(input_tokens['input_ids'].shape[1])
    
    # Create full texts (input + target)
    full_texts = [inp + tgt for inp, tgt in zip(input_texts, target_texts)]
    
    # Tokenize full texts
    tokenized = tokenizer(
        full_texts,
        max_length=max_seq_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    # Create labels (mask input portion)
    labels = input_ids.clone()
    for i, inp_len in enumerate(input_lengths):
        labels[i, :inp_len] = -100  # Ignore loss on input tokens
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    return outputs.loss


def parse_sft_key(sft_key: str):
    """Parse SFT key like 'temp0.7_styleconcise' into temp and style."""
    if not sft_key.startswith('temp'):
        raise ValueError(f"SFT key must start with 'temp', got: {sft_key}")
    
    parts = sft_key.split('_')
    if len(parts) != 2:
        raise ValueError(f"SFT key must be in format 'tempX_styleY', got: {sft_key}")
    
    temp_str = parts[0].replace('temp', '')
    style_str = parts[1].replace('style', '')
    
    try:
        temp = float(temp_str)
    except ValueError:
        raise ValueError(f"Could not parse temperature from: {temp_str}")
    
    return temp, style_str



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m forwardsft_script_path yaml_path.yaml temp{temp}_style{style}")
        sys.exit(1)
    
    # Load environment variables
    load_dotenv()
    
    config_path = sys.argv[1]
    sft_key = sys.argv[2]  # e.g., "temp0.7_styleconcise"
    
    # Parse temp and style from key
    temp, style = parse_sft_key(sft_key)
    
    # Load configuration
    args = YamlConfig(config_path)
    
    # Get SFT configuration - it should be under forwardsft_{sft_key}
    sft_config_key = f"forwardsft_{sft_key}"
    if not hasattr(args, sft_config_key):
        raise ValueError(f"SFT configuration key '{sft_config_key}' not found in yaml")
    
    sft_config = getattr(args, sft_config_key).__dict__
    
    # Validate required SFT parameters
    required_params = ['lora_r', 'lora_alpha', 'lora_dropout', 'target_modules', 
                      'learning_rate', 'num_epochs', 'batch_size', 'max_seq_length']
    for param in required_params:
        if param not in sft_config:
            raise ValueError(f"Required SFT parameter '{param}' not found in {sft_config_key}")
    
    # Get WandB project name from environment
    project_name = os.getenv('WANDB_PROJECT')
    if not project_name:
        raise ValueError("WANDB_PROJECT environment variable not set")
    
    # Create WandB run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.args_name}_forwardsft_temp{temp}_style{style}_{timestamp}"
    
    print(f"Forward SFT Configuration:")
    print(f"  Run: {args.args_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Temperature: {temp}")
    print(f"  Style: {style}")
    print(f"  WandB Project: {project_name}")
    print(f"  WandB Run: {run_name}")
    print(f"  LoRA config: {sft_config}")
    
    # Initialize WandB
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'run_name': args.args_name,
            'dataset': args.dataset,
            'model_name': args.model_name,
            'temperature': temp,
            'style': style,
            'sft_config': sft_config
        }
    )
    
    # Create output directory and log run
    base_dir = f"results_and_data/results/main/{args.args_name}/train"
    runs_file = os.path.join(base_dir, "forwardsft_runs.txt")
    os.makedirs(base_dir, exist_ok=True)
    
    # Append run name to runs file
    with open(runs_file, 'a') as f:
        f.write(f"{run_name}\n")
    print(f"Logged run to: {runs_file}")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    device = chat_wrapper.model.device
    
    # Load SFT data
    print("Loading SFT training data...")
    sft_data = load_sft_data(args.args_name, args.dataset, temp, style)
    print(f"Loaded {len(sft_data)} training examples")
    
    # Create training pairs
    print("Creating training pairs...")
    training_pairs = create_training_pairs(sft_data, args.dataset, chat_wrapper)
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_model = setup_lora_model(chat_wrapper.model, sft_config)
    trainable_params = lora_model.num_parameters(only_trainable=True)
    print(f"LoRA model setup complete. Trainable parameters: {trainable_params}")
    
    # Log model info to WandB
    wandb.log({
        "total_examples": len(training_pairs),
        "trainable_parameters": trainable_params
    })
    
    # Training setup
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=sft_config['learning_rate'])
    
    # Training loop
    batch_size = sft_config['batch_size']
    num_epochs = sft_config['num_epochs']
    max_steps = sft_config['max_steps']
    save_frequency = sft_config['save_frequency']
    max_seq_length = sft_config['max_seq_length']
    
    # Shuffle training pairs indices
    indices = list(range(len(training_pairs)))
    
    step = 0
    total_loss = 0.0
    
    print(f"Starting training: {num_epochs} epochs, {len(training_pairs)} examples, batch size {batch_size}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle for this epoch
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_steps = 0
        
        # Batch training
        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i+batch_size]
            batch_pairs = [training_pairs[idx] for idx in batch_indices]
            
            # Training step
            optimizer.zero_grad()
            loss = train_step(
                lora_model, 
                chat_wrapper.tokenizer, 
                batch_pairs,
                max_seq_length,
                device
            )
            loss.backward()
            optimizer.step()
            
            step += 1
            epoch_steps += 1
            current_loss = loss.item()
            total_loss += current_loss
            epoch_loss += current_loss
            
            # Log loss to WandB every step
            wandb.log({
                "loss": current_loss,
                "step": step,
                "epoch": epoch + 1
            })
            
            # Save LoRA adapters as artifact every 20 steps
            if step % save_frequency == 0:
                avg_loss = total_loss / step
                print(f"Step {step}: Loss = {current_loss:.4f}, Avg Loss = {avg_loss:.4f}")
                save_lora_as_artifact(lora_model, step, temp, style, run_name)
                
                # Also log running averages
                wandb.log({
                    "avg_loss": avg_loss,
                    "step": step
                })
            
            if step == max_steps:
                break
        
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Log epoch summary
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch + 1
        })
        
        if step == max_steps:
            break
    
    # Final artifact save
    print(f"Training complete! Saving final adapters...")
    save_lora_as_artifact(lora_model, step, temp, style, run_name)
    
    final_avg_loss = total_loss / step if step > 0 else 0.0
    print(f"Total steps: {step}")
    print(f"Final average loss: {final_avg_loss:.4f}")
    
    # Log final metrics
    wandb.log({
        "final_avg_loss": final_avg_loss,
        "total_steps": step
    })
    
    wandb.finish()

