import modal
import os
from typing import List

import torch
import random
from tqdm import tqdm
import wandb


from model.load import load_model
from sft_utils.forward_data import load_sft_data, create_training_pairs
from sft_utils.lora import setup_lora_model, save_lora_as_artifact
from sft_utils.train import train_step, parse_forward_sft_key
from utils.util import YamlConfig

# Define Modal app
app = modal.App("forward-sft")


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

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/results": results_volume, '/models': model_volume, '/data': data_volume},
    secrets=[
        modal.Secret.from_dotenv(),                 # Contains WANDB_PROJECT
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=10800,  # 3 hours
    memory=32768,   # 32GB RAM
)
def run_forward_sft(
    args_name: str,
    dataset: str,
    model_name: str,
    temp: float,
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
):
    """
    Run forward SFT training on Modal infrastructure.
    This function contains the entire training loop.
    """
    # Get WandB project name from Modal Secret
    project_name = os.getenv('WANDB_PROJECT')
    if not project_name:
        raise ValueError("WANDB_PROJECT environment variable not set")
    
    # Create WandB run name with timestamp
    run_name = f"{args_name}_forwardsft_temp{temp}_style{style}_seed{seed_num}"
    
    # Create output directory and log run
    base_dir = f"/results/results/main/{args_name}/train"
    runs_file = os.path.join(base_dir, "forwardsft_runs.txt")
    os.makedirs(base_dir, exist_ok=True)
    
    # Append run name to runs file
    with open(runs_file, 'a') as f:
        f.write(f"{run_name}\n")
    print(f"Logged run to: {runs_file}")
    
    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device='auto', override_path='/models')
    device = chat_wrapper.model.device
    
    # Load SFT data
    print("Loading SFT training data...")
    sft_data = load_sft_data(args_name, dataset, temp, style, datasets_dir='/data', results_dir='/results/results')
    training_pairs = create_training_pairs(sft_data, dataset, chat_wrapper)
    print(f"Loaded {len(training_pairs)} from {len(sft_data)} training articles")
    
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
            'temperature': temp,
            'style': style,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'max_seq_length': max_seq_length,
            'max_steps': max_steps
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
            
            # Log loss to WandB every step
            wandb.log({
                "loss": current_loss,
                "step": step,
                "epoch": epoch + 1
            })
            
            # Save LoRA adapters as artifact every save_frequency steps
            if step % save_frequency == 0:
                print(f"Step {step}: Loss = {current_loss:.4f}")
                save_lora_as_artifact(lora_model, step, temp, style, run_name)
                
                # Also log running averages
                #wandb.log({"step": step})
            
            if (max_steps is not None) and (step == max_steps):
                break
        
        if (max_steps is not None) and (step == max_steps):
            break
    
    # Final artifact save
    print(f"Training complete! Saving final adapters...")
    save_lora_as_artifact(lora_model, step, temp, style, run_name)
    
    print(f"Total steps: {step}")
    
    # Commit volume to persist changes
    results_volume.commit()
    
    wandb.finish()
    
    return {
        "run_name": run_name,
        "total_steps": step
    }


@app.local_entrypoint()
def main(*arglist):
    """
    Local entrypoint - runs on your machine.
    Parses config and launches remote training.
    """
    if len(arglist) != 3:
        raise ValueError("Usage: modal run forward_sft.py yaml_path.yaml temp{temp}_style{style} {seed}")
    
    config_path = arglist[0]
    sft_key = arglist[1]  # e.g., "temp0.7_styleconcise"
    seed_num = arglist[2]
    
    temp, style = parse_forward_sft_key(sft_key)
    
    args = YamlConfig(config_path)
    
    # Get SFT configuration
    sft_config_key = f"forwardsft_{sft_key}"
    if not hasattr(args, sft_config_key):
        raise ValueError(f"SFT configuration key '{sft_config_key}' not found in yaml")
    
    sft_config = getattr(args, sft_config_key).__dict__
    
    print("Starting Modal training job...")
    print(f"Config: {config_path}")
    print(f"SFT Key: {sft_key}")
    print(f"Args name: {args.args_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    
    # Launch remote training
    result = run_forward_sft.remote(
        args_name=args.args_name,
        dataset=args.dataset,
        model_name=args.model_name,
        temp=temp,
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

