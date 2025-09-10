import sys
import modal
import os
from typing import List, Optional

import torch
import random
from tqdm import tqdm
import wandb

import pandas as pd

from load_data import load_dataset
from model.load import load_model
from sft_utils.lora import setup_lora_model, save_lora_as_artifact
from sft_utils.train import train_step_forward, parse_forward_sft_key
from utils.util import YamlConfig

from prompts.strategy import SYSTEM_PROMPTS

# Define Modal app
app = modal.App("forward-sft-advice")


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




def load_sft_data_advice(run_name: str, dataset_name: str, strategy: str, style: str, *_, datasets_dir: Optional[str] = None, results_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load all training data for SFT from generated advice responses.
    
    Args:
        run_name: Name of the run 
        dataset_name: Name of dataset
        strategy: Strategy type ('prorisk' or 'antirisk')
        style: Style type ('formal' or 'casual') 
        datasets_dir: Directory containing original datasets
        results_dir: Directory containing generated results
        results_set_name: Name of results set (default 'strategy')
        subdir_name: Name of subdirectory (default 'model_advice')
        
    Returns:
        DataFrame with question and response data for SFT
    """
    # Load original train data to get questions
    train_data, _, _ = load_dataset(dataset_name, splits = ["train"], datasets_dir = datasets_dir)
    
    # Load response data for this strategy/style combination
    results_dir = results_dir or "results_and_data/results"
    responses_dir = os.path.join(results_dir, f'strategy/{run_name}/train/model_advice')
    
    response_file = f"{responses_dir}/strategy_{strategy}_style_{style}.csv"

    if not os.path.exists(response_file):
        raise ValueError(f"Response file not found: {response_file}")
        
    print(f"Loading {response_file}")
    responses = pd.read_csv(response_file)
    
    # Merge with original questions
    sft_data = train_data.merge(responses, on='question_idx', how='inner')
    
    print(f"SFT data: {len(sft_data)} training examples for strategy={strategy}, style={style}")
    
    return sft_data


def create_training_pairs_advice(sft_data: pd.DataFrame, strategy: str, style: str, chat_wrapper) -> List[dict]:
    """
    Create input-target pairs for SFT on advice generation.
    
    Args:
        sft_data: DataFrame with columns [question_idx, question, response]
        strategy: Strategy type ('prorisk' or 'antirisk')  
        style: Style type ('formal' or 'casual')
        chat_wrapper: Model wrapper for formatting
        
    Returns:
        List of training pairs with input/target for SFT
    """
    # Get system prompt for this strategy/style combination
    system_prompt = SYSTEM_PROMPTS[(strategy, style)]
    
    training_pairs = []
    
    for _, row in tqdm(sft_data.iterrows(), total=len(sft_data), desc="Creating training pairs"):
        # Format input (question with appropriate system prompt)
        formatted_input = chat_wrapper.format_chat(
            system_prompt=system_prompt,
            user_message=row['question'],
            prefiller=""
        )

        # Target is the response (already cleaned in generation script)
        target = row['response']
        
        training_pairs.append({
            "input": formatted_input,
            "target": target,
            "question_idx": row['question_idx']
        })
    
    return training_pairs


def parse_advice_sft_key(sft_key: str):
    """
    Parse SFT key in format 'strategy_{strategy}_style_{style}'
    Returns (strategy, style)
    """
    if not sft_key.startswith('strategy_'):
        raise ValueError(f"SFT key must start with 'strategy_', got: {sft_key}")
    
    # Remove 'strategy_' prefix
    remainder = sft_key[9:]  # len('strategy_') = 9
    
    # Split on '_style_'
    if '_style_' not in remainder:
        raise ValueError(f"SFT key must contain '_style_', got: {sft_key}")
    
    strategy, style = remainder.split('_style_', 1)
    
    return strategy, style


def run_forward_sft_advice(
    args_name: str,
    dataset: str,
    model_name: str,
    strategy: str,
    style: str,
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
    seed_num: int,
    *_,
    results_dir: str = 'results_and_data/modal_results/results',
    data_dir: str = 'results_and_data/data',
    models_dir: Optional[str] = None
):
    """
    Run forward SFT training for advice generation on Modal infrastructure.
    This function contains the entire training loop.
    """
    # Set seeds for reproducibility
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_num)
       
    # Get WandB project name from Modal Secret
    project_name = os.getenv('WANDB_PROJECT')
    if not project_name:
        raise ValueError("WANDB_PROJECT environment variable not set")
    
    # Create WandB run name with timestamp
    run_name = f"{args_name}_forwardsft_strategy{strategy}_style{style}_seed{seed_num}"
    
    # Create output directory and log run
    base_dir = f"{results_dir}/strategy/{args_name}/train"
    runs_file = os.path.join(base_dir, "forwardsft_runs.txt")
    os.makedirs(base_dir, exist_ok=True)

    # Load SFT data for advice format (strategy/style instead of temp/style)
    print("Loading SFT training data...")
    sft_data = load_sft_data_advice(
        args_name, dataset, strategy, style, datasets_dir=data_dir, results_dir=results_dir, 
    )
    
    # Append run name to runs file
    with open(runs_file, 'a') as f:
        f.write(f"{run_name}\n")
    print(f"Logged run to: {runs_file}")
    
    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device='auto', override_path=models_dir)
    device = chat_wrapper.model.device
    
    training_pairs = create_training_pairs_advice(sft_data, strategy, style, chat_wrapper)
    print(f"Loaded {len(training_pairs)} from {len(sft_data)} training questions")
    
    # Setup LoRA
    print("Setting up LoRA...")
    sft_config = {
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
    lora_model = setup_lora_model(chat_wrapper.model, sft_config)
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
            'strategy': strategy,
            'style': style,
            'seed': seed_num,
            **sft_config,
        }
    )

    # Log model info to WandB
    wandb.log({
        "total_examples": len(training_pairs),
        "trainable_parameters": trainable_params
    })
    
    # Training pairs indices, for shuffling
    indices = list(range(len(training_pairs)))
    
    print(f"Starting training: {num_epochs} epochs, {len(training_pairs)} examples, batch size {batch_size}")
    step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Shuffle for this epoch
        random.shuffle(indices)
        epoch_steps = 0
        
        # Batch training
        for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[i:i+batch_size]
            batch_pairs = [training_pairs[idx] for idx in batch_indices]
            
            # Training step
            optimizer.zero_grad()
            loss = train_step_forward(
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
            
            # Log loss to WandB every step
            wandb.log({
                "loss": current_loss,
                "step": step,
                "epoch": epoch + 1
            })
            
            # Save LoRA adapters as artifact every save_frequency steps
            if step % save_frequency == 0:
                print(f"Step {step}: Loss = {current_loss:.4f}")
                save_lora_as_artifact(lora_model, step, strategy, style, run_name)
                
            if (max_steps is not None) and (step == max_steps):
                break
        
        if (max_steps is not None) and (step == max_steps):
            break
    
    # Final artifact save
    print(f"Training complete! Saving final adapters...")
    save_lora_as_artifact(lora_model, step, strategy, style, run_name)
    
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
        modal.Secret.from_dotenv('.env.modal'),                 # Contains WANDB_PROJECT
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=10800,  # 3 hours
    memory=32768,   # 32GB RAM
)
def run_forward_sft_advice_remote(*args, **kwargs):
    return run_forward_sft_advice(*args, **kwargs, results_dir='/results/results', data_dir='/data', models_dir='/models')

@app.local_entrypoint()
def main_modal(*arglist):
    return main(*arglist, activate_modal=True)


def main(*arglist, activate_modal):
    if len(arglist) != 3:
        raise ValueError("Usage: modal run scripts_advice.forward_sft.forward_sft_advice.py yaml_path.yaml strategy_{strategy}_style_{style} {seed}")
    
    config_path = arglist[0]
    sft_key = arglist[1]  # e.g., "strategy_prorisk_style_formal"
    seed_num = int(arglist[2])
    
    strategy, style = parse_advice_sft_key(sft_key)
    
    args = YamlConfig(config_path)
    
    # Get SFT configuration
    sft_config_key = f"forwardsft_{sft_key}"
    if not hasattr(args, sft_config_key):
        raise ValueError(f"SFT configuration key '{sft_config_key}' not found in yaml")
    
    sft_config = getattr(args, sft_config_key).__dict__
    
    print("Starting Modal training job...")
    print(f"Config: {config_path}")
    print(f"SFT Key: {sft_key}")
    print(f"Strategy: {strategy}")
    print(f"Style: {style}")
    print(f"Args name: {args.args_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")

    function = run_forward_sft_advice_remote if activate_modal else run_forward_sft_advice
    
    # Launch remote training
    result = function(
        args_name=args.args_name,
        dataset=args.dataset,
        model_name=args.model_name,
        strategy=strategy,
        style=style,
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


if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv()

    main(*sys.argv[1:], activate_modal=False)


