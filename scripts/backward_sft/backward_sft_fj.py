import sys
import modal
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
import random
import wandb

from load_data import load_dataset, load_model_summaries
from model.load import load_model
from model.base import ChatTemplateWrapper

from sft_utils.lora import setup_lora_model, save_lora_as_artifact
from sft_utils.train import parse_forward_sft_key

from prompts import (
    DETECTION_SYSTEM_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT, 
    DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY,
    DETECTION_FOLLOWUP_QUESTION
)

from utils.util import YamlConfig

# Define Modal app
app = modal.App("justification-sft")

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas",
        "torch", 
        "transformers",
        "peft",
        "wandb",
        "datasets",
        "tqdm",
        "accelerate",
        "python-dotenv",
    ])
    .add_local_python_source("model")
    .add_local_python_source("sft_utils")
    .add_local_python_source("utils")
    .add_local_python_source("prompts")
    .add_local_python_source("load_data")
)

# Volume for persistent storage of results
results_volume = modal.Volume.from_name("results-vol", create_if_missing=True)
model_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("data-vol", create_if_missing=True)


def load_summaries_for_justification_sft(
    run_name: str,
    split_name: str, 
    temps: List[float],
    num_trials: List[int],
    styles: List[str],
    results_dir: Optional[str] = None
) -> Dict[Tuple, pd.DataFrame]:
    """Load all summaries for justification SFT training."""
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


def load_justification_data(
    run_name: str,
    split_name: str,
    target_temp: float,
    target_style: str,
    results_dir: Optional[str] = None
) -> pd.DataFrame:
    """Load justification data from CSV file."""
    results_dir = results_dir or 'results_and_data/results'
    justification_file = f"{results_dir}/main/{run_name}/{split_name}/false_choice_justifications/temp{target_temp}_style{target_style}.csv"
    
    if not os.path.exists(justification_file):
        raise FileNotFoundError(f"Justification file not found: {justification_file}")
    
    df = pd.read_csv(justification_file)
    print(f"Loaded {len(df)} justification examples from {justification_file}")
    return df


def create_justification_training_examples(
    justification_df: pd.DataFrame,
    split_data: pd.DataFrame,
    all_summaries: Dict[Tuple, pd.DataFrame],
    chat_wrapper: ChatTemplateWrapper
) -> List[dict]:
    """
    Create training examples from justification data.
    Each example contains the full conversation with 4 parts to be masked appropriately.
    """
    training_examples = []
    
    # Create document lookup for articles
    doc_lookup = dict(zip(split_data['document_idx'], split_data['article']))
    
    for idx, row in tqdm(justification_df.iterrows(), total=len(justification_df), desc="Creating training examples"):
        document_idx = row['document_idx']
        
        if document_idx not in doc_lookup:
            continue
            
        article = doc_lookup[document_idx]
        
        # Get summary keys
        summary_1_key = (row['summary1_temp'], row['summary1_trial'], row['summary1_style'])
        summary_2_key = (row['summary2_temp'], row['summary2_trial'], row['summary2_style'])
        
        # Load actual summary texts
        if summary_1_key not in all_summaries or summary_2_key not in all_summaries:
            continue
            
        summary_1_df = all_summaries[summary_1_key]
        summary_2_df = all_summaries[summary_2_key]
        
        # Get summaries for this document
        summary_1_row = summary_1_df[summary_1_df['document_idx'] == document_idx]
        summary_2_row = summary_2_df[summary_2_df['document_idx'] == document_idx]
        
        if len(summary_1_row) == 0 or len(summary_2_row) == 0:
            continue
            
        summary_1 = summary_1_row['summary'].iloc[0]
        summary_2 = summary_2_row['summary'].iloc[0]
        
        # Reconstruct the conversation parts
        initial_prompt = (
            DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) +
            DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(
                summary_1=summary_1, 
                summary_2=summary_2
            ) +
            DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
        )
        
        choice = str(row['false_choice'])  # Convert to string to handle integers
        followup_question = DETECTION_FOLLOWUP_QUESTION.format(idx=choice)
        justification = row['justification']
        
        training_examples.append({
            'document_idx': document_idx,
            'system_prompt': DETECTION_SYSTEM_PROMPT,
            'initial_prompt': initial_prompt,
            'choice': choice,
            'followup_question': followup_question, 
            'justification': justification,
            'summary_1_key': summary_1_key,
            'summary_2_key': summary_2_key
        })
    
    print(f"Created {len(training_examples)} training examples")
    return training_examples


def train_step_justifications(
    model,
    tokenizer,
    batch_examples: List[dict],
    device: str
) -> torch.Tensor:
    """
    Training step for justification-based backward SFT.
    
    Tokenizes and concatenates 4 parts:
    1. System prompt + initial prompt (masked with -100)
    2. Choice (trained on)
    3. Followup question (masked with -100)
    4. Justification (trained on)
    """
    assert len(batch_examples) == 1, "Currently only supports batch_size=1"
    
    model.train()
    
    example = batch_examples[0]
    
    # Format the conversation parts using chat template
    system_prompt = example['system_prompt']
    initial_prompt = example['initial_prompt']
    choice = example['choice']
    followup_question = example['followup_question']
    justification = example['justification']
    
    # Part 1: System prompt + initial user message
    part1_formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Part 2: Choice (model response)
    choice_text = choice
    
    # Part 3: Followup question (user message)
    part3_formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": choice},
            {"role": "user", "content": followup_question}
        ],
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False
    )
    # Extract just the followup part
    part3_text = part3_formatted[len(part1_formatted + choice_text):]
    
    # Part 4: Justification (model response)
    justification_text = justification
    
    # Tokenize each part separately
    part1_tokens = tokenizer(part1_formatted, add_special_tokens=False, return_tensors="pt")
    part2_tokens = tokenizer(choice_text, add_special_tokens=False, return_tensors="pt")
    part3_tokens = tokenizer(part3_text, add_special_tokens=False, return_tensors="pt")
    part4_tokens = tokenizer(justification_text, add_special_tokens=False, return_tensors="pt")
    
    # Get lengths for masking
    part1_len = part1_tokens['input_ids'].shape[1]
    part2_len = part2_tokens['input_ids'].shape[1]
    part3_len = part3_tokens['input_ids'].shape[1]
    part4_len = part4_tokens['input_ids'].shape[1]
    
    # Concatenate all parts
    full_input_ids = torch.cat([
        part1_tokens['input_ids'],
        part2_tokens['input_ids'], 
        part3_tokens['input_ids'],
        part4_tokens['input_ids']
    ], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones_like(full_input_ids)
    
    # Move to device
    input_ids = full_input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Create labels with proper masking
    labels = input_ids.clone()
    
    # Mask parts 1 and 3 (user-generated content)
    labels[:, :part1_len] = -100  # Part 1: system + initial prompt
    part3_start = part1_len + part2_len
    part3_end = part3_start + part3_len
    labels[:, part3_start:part3_end] = -100  # Part 3: followup question
    
    # Parts 2 and 4 remain unmasked (choice and justification)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    import pdb; pdb.set_trace()
    
    return outputs.loss


def run_justification_sft(
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
    max_steps: int,
    save_frequency: int,
    seed_num: int,
    results_dir: str,
    models_override_path: Optional[str] = None,
    data_path: Optional[str] = None
):
    """
    Run justification SFT training on Modal infrastructure.
    This function contains the entire training loop.
    """
    # Assert batch_size = 1 for justification SFT
    assert batch_size == 1, f"Justification SFT currently only supports batch_size=1, got {batch_size}"
    
    # Set seeds for reproducibility
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_num)

    # Get WandB project name from Modal Secret
    project_name = os.getenv('WANDB_PROJECT')
    if not project_name:
        raise ValueError("WANDB_PROJECT environment variable not set")
    
    # Create WandB run name
    run_name = f"{args_name}_justificationsft_temp{target_temp}_style{target_style}_seed{seed_num}"
    
    # Create output directory and log run
    base_dir = f"{results_dir}/results/main/{args_name}/train"
    runs_file = os.path.join(base_dir, "justificationsft_runs.txt")
    os.makedirs(base_dir, exist_ok=True)
    
    # Append run name to runs file
    with open(runs_file, 'a') as f:
        f.write(f"{run_name}\n")
    print(f"Logged run to: {runs_file}")
    
    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device='auto', override_path=models_override_path)
    device = chat_wrapper.model.device
    
    # Load dataset
    print(f"Loading dataset: {dataset}")
    train_data, _, _ = load_dataset(dataset, splits=['train'], datasets_dir=data_path)
    
    # Load all summaries
    print("Loading all summaries...")
    all_summaries = load_summaries_for_justification_sft(
        args_name, 'train', temps, num_trials, styles, results_dir=os.path.join(results_dir, 'results')
    )
    
    # Load justification data
    print("Loading justification data...")
    justification_df = load_justification_data(
        args_name, 'train', target_temp, target_style, results_dir=os.path.join(results_dir, 'results')
    )
    
    # Create training examples
    print("Creating justification training examples...")
    training_examples = create_justification_training_examples(
        justification_df, train_data, all_summaries, chat_wrapper
    )
    
    if len(training_examples) == 0:
        raise ValueError("No training examples created!")
    
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
            'training_type': 'justification_sft',
            **sft_config_dict
        }
    )
    
    # Log model info to WandB
    wandb.log({
        "total_examples": len(training_examples),
        "trainable_parameters": trainable_params
    })
    
    # Training
    indices = list(range(len(training_examples)))
    print(f"Starting training: {num_epochs} epochs, {len(training_examples)} examples, batch size {batch_size}")
    
    step = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle for this epoch
        random.shuffle(indices)
        
        # Batch training
        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i+batch_size]
            batch_examples = [training_examples[idx] for idx in batch_indices]

            # Training step
            optimizer.zero_grad()
            loss = train_step_justifications(
                lora_model, 
                chat_wrapper.tokenizer, 
                batch_examples,
                device
            )
            try:
                loss.backward()
                optimizer.step()
            except torch.OutOfMemoryError:
                print(f'Skipping batch: {[ex["document_idx"] for ex in batch_examples]}')
                torch.cuda.empty_cache()
                continue
            
            step += 1
            current_loss = loss.item()
            
            # Log loss to WandB every step
            wandb.log({
                "loss": current_loss,
                "step": step,
                "epoch": epoch + 1
            })
            
            # Save LoRA adapters as artifact every save_frequency steps
            if step % save_frequency == 0:
                print(f"Step {step}: Loss = {current_loss:.4f}")
                save_lora_as_artifact(lora_model, step, target_temp, target_style, run_name)
            
            if (max_steps is not None) and (step >= max_steps):
                break
        
        if (max_steps is not None) and (step >= max_steps):
            break
    
    # Final artifact save
    print(f"Training complete! Saving final adapters...")
    save_lora_as_artifact(lora_model, step, target_temp, target_style, run_name)
    
    print(f"Total steps: {step}")
    
    # Commit volume to persist changes
    results_volume.commit()
    
    wandb.finish()
    
    return {
        "run_name": run_name,
        "total_steps": step
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/results": results_volume, '/models': model_volume, '/data': data_volume},
    secrets=[
        modal.Secret.from_dotenv('.modal.env'),                 # Contains WANDB_PROJECT
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=10800,  # 3 hours
    memory=32768,   # 32GB RAM
)
def run_justification_sft_remote(*aargs, **kwargs):
    return run_justification_sft.remote(*aargs, **kwargs, results_dir = '/results', models_override_path='/models', data_path='/data')


@app.local_entrypoint()
def main(*arglist):
    """
    Local entrypoint - runs on your machine.
    Parses config and launches remote training.
    """
    if len(arglist) != 3:
        raise ValueError("Usage: modal run backward_sft_justifications_modal.py yaml_path.yaml temp{temp}_style{style} {seed}")
    
    config_path = arglist[0]
    sft_key = arglist[1]  # e.g., "temp0.7_styleconcise"
    seed_num = int(arglist[2])

    target_temp, target_style = parse_forward_sft_key(sft_key)

    args = YamlConfig(config_path)
    
    # Get justification SFT configuration for the specific target style
    sft_config_key = f"backwardsfjsft_temp{target_temp}_style{target_style}"
    sft_config = getattr(args, sft_config_key).__dict__
    
    print("Starting Modal justification SFT training job...")
    print(f"Config: {config_path}")
    print(f"Target temp: {target_temp}")
    print(f"Target style: {target_style}")
    print(f"Args name: {args.args_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Seed: {seed_num}")
    
    # Launch remote training
    result = run_justification_sft_remote.remote(
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
        max_steps=sft_config['max_steps'],
        save_frequency=sft_config['save_frequency'],
        seed_num=seed_num
    )
    
    print(f"Training completed!")
    print(f"Run name: {result['run_name']}")
    print(f"Total steps: {result['total_steps']}")


if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv()

    arglist = sys.argv[1:]

    if len(arglist) != 3:
        raise ValueError("Usage: modal run backward_sft_justifications_modal.py yaml_path.yaml temp{temp}_style{style} {seed}")
    
    config_path = arglist[0]
    sft_key = arglist[1]  # e.g., "temp0.7_styleconcise"
    seed_num = int(arglist[2])

    target_temp, target_style = parse_forward_sft_key(sft_key)

    args = YamlConfig(config_path)
    
    # Get justification SFT configuration for the specific target style
    sft_config_key = f"backwardsfjsft_temp{target_temp}_style{target_style}"
    sft_config = getattr(args, sft_config_key).__dict__
    
    print("Starting Modal justification SFT training job...")
    print(f"Config: {config_path}")
    print(f"Target temp: {target_temp}")
    print(f"Target style: {target_style}")
    print(f"Args name: {args.args_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Seed: {seed_num}")
    
    # Launch remote training
    result = run_justification_sft(
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
        max_steps=sft_config['max_steps'],
        save_frequency=sft_config['save_frequency'],
        seed_num=seed_num,
        results_dir='results_and_data/modal_results'
    )
    
    print(f"Training completed!")
    print(f"Run name: {result['run_name']}")
    print(f"Total steps: {result['total_steps']}")



