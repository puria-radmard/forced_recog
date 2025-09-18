import modal
import os
from typing import Dict, List, Optional

from model.load import load_model
from sft_utils.lora import download_and_apply_lora
from load_data import load_dataset
from util.util import YamlConfig
from generate_hf_Jesse import generate_summaries_for_split


# Define Modal app
app = modal.App("forward-sft-generate")

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas",
        "torch",
        "transformers",
        "datasets",
        "wandb",
        "peft",
        "tqdm",
        "python-dotenv",
    ])
    .add_local_python_source("model")
    .add_local_python_source("sft_utils")
    .add_local_python_source("util")
    .add_local_python_source("prompts")
    .add_local_python_source("load_data")
    .add_local_python_source("generate_hf_Jesse")  # so we can import generate_hf
)

# Volumes
results_volume = modal.Volume.from_name("results-vol", create_if_missing=True)
model_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("data-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/results": results_volume, "/models": model_volume, "/data": data_volume},
    secrets=[
        modal.Secret.from_dotenv(),                 # Contains WANDB_PROJECT
    ],
    timeout=10800,
    memory=32768,
)
def run_generation(
    model_name: str,
    dataset: str,
    splits: Dict[str, int],
    temps: List[float],
    num_trials: List[int],
    styles: List[str],
    args_name: float,
    wandb_run_name: Optional[str] = None,
    artifact_suffix: Optional[str] = None,
    continue_mode: bool = False,
):
    """
    Run summary generation (base model or with LoRA).
    """

    # Decide LoRA usage
    use_lora = wandb_run_name is not None and artifact_suffix is not None
    if use_lora:
        print(f"Running with LoRA adapters:")
        print(f"  WandB Run: {wandb_run_name}")
        print(f"  Artifact Suffix: {artifact_suffix}")
    else:
        print("Running with base model")

    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")

    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device="auto")

    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_suffix)

    # Load dataset
    print(f"Loading dataset: {dataset}")
    train_data, test_data, validation_data = load_dataset(
        dataset,
        splits=list(splits.keys()),
        datasets_dir="/data"
    )

    split_data_map = {
        "train": train_data,
        "test": test_data,
        "validation": validation_data,
    }

    # Generate summaries
    for split_name, max_generate in splits.items():
        split_data = split_data_map[split_name]
        
        # Skip if split_data is None (split not available)
        if split_data is None:
            print(f"Skipping {split_name} split - no data available")
            continue
            
        print(f"Generating summaries for {split_name} split ({len(split_data)} documents)")

        generate_summaries_for_split(
            chat_wrapper=chat_wrapper,
            split_data=split_data,
            max_generate=max_generate,
            dataset_name=dataset,
            split_name=split_name,
            temps=temps,
            num_trials=num_trials,
            styles=styles,
            run_name=args_name,
            use_lora=use_lora,
            lora_run_name=wandb_run_name,
            artifact_suffix=artifact_suffix,
            results_dir="/results/results",
            continue_mode=continue_mode,
        )

    print("Generation complete!")

    # Commit volume to persist outputs
    results_volume.commit()
    
    # Return the results directory path for downloading
    return f"results/main/{args_name}"


@app.local_entrypoint()
def main(*arglist):
    """
    Local entrypoint - runs on your machine.
    Launches remote generation.
    """
    
    # Parse command line arguments
    continue_mode = len(arglist) > 0 and arglist[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(arglist) - (1 if continue_mode else 0)
    
    if effective_argc not in [1, 3]:
        raise ValueError(
            "Usage:\n"
            "  Base model: modal run generate_hf_modal_Jesse.py /path/to/yaml/args.yaml [continue]\n"
            "  With LoRA:  modal run generate_hf_modal_Jesse.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_suffix> [continue]"
        )

    config_path = arglist[0]

    # Parse config
    args = YamlConfig(config_path)

    if effective_argc == 3:
        wandb_run_name = arglist[1]
        artifact_suffix = arglist[2]
    else:
        wandb_run_name = None
        artifact_suffix = None
    
    splits = args.splits.__dict__.keys()

    print("Starting Modal generation job...")
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
        
    result = run_generation.remote(
        model_name=args.model_name,
        dataset=args.dataset,
        splits=args.splits.__dict__,
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        args_name=args.args_name,
        wandb_run_name=wandb_run_name,
        artifact_suffix=artifact_suffix,
        continue_mode=continue_mode,
    )

    print("Remote generation completed!")
    
    # Download the generated files to local machine
    print("Downloading generated files to local machine...")
    results_path = result  # This contains the path to the results directory
    
    # Create local results directory
    local_results_dir = f"results_and_data/results/main/{args.args_name}"
    os.makedirs(local_results_dir, exist_ok=True)
    
    # Download all files from the results directory
    try:
        # Use subprocess to run modal volume download command
        import subprocess
        result = subprocess.run([
            "modal", "volume", "get", "results-vol", 
            results_path, local_results_dir
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f"Files downloaded successfully to: {local_results_dir}")
            
            # List the downloaded files
            for root, dirs, files in os.walk(local_results_dir):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        print(f"  Downloaded: {file_path} ({file_size} bytes)")
        else:
            print(f"Error downloading files: {result.stderr}")
            print("You can manually download the files using:")
            print(f"modal volume get results-vol {results_path} {local_results_dir}")
                    
    except Exception as e:
        print(f"Error downloading files: {e}")
        print("You can manually download the files using:")
        print(f"modal volume get results-vol {results_path} {local_results_dir}")
    
    return result