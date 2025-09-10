import os
import modal

from typing import List, Optional
import traceback

from model.load import load_model
from load_data import load_dataset
from utils.util import YamlConfig
from sft_utils.train import parse_forward_sft_key

# Import the core function - assuming the script is named something like false_justification.py
# You'll need to adjust this import based on the actual filename
from scripts_summary.pairwise_selection.pairwise_hf_false_justification import elicit_fj_for_split


# Define Modal app
app = modal.App("elicit-false-justifications")

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
    .add_local_python_source("utils")
    .add_local_python_source("prompts")
    .add_local_python_source("load_data")
    .add_local_python_source("scripts_summary.pairwise_selection.pairwise_hf")  # needed for imported functions
    .add_local_python_source("scripts_summary.pairwise_selection.pairwise_hf_false_justification")  # adjust filename as needed
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
        modal.Secret.from_dotenv('.modal.env'),                 # Contains WANDB_PROJECT
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=10800,
    memory=32768,
)
def run_elicit_false_justifications(
    model_name: str,
    dataset: str,
    temps: List[float],
    num_trials: List[int],
    styles: List[str],
    args_name: str,
    target_temp: str,
    target_style: str,
    continue_mode: bool = False,
):
    """
    Run false justification elicitation for a specific temperature/style combination.
    """
    print(f"Running false justification elicitation:")
    print(f"  Target temp: {target_temp}")
    print(f"  Target style: {target_style}")

    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")

    # Load model (base model only - LoRA explicitly disabled)
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device="auto", override_path="/models")

    # Load dataset (train split)
    print(f"Loading dataset: {dataset}")
    train_data, _, _ = load_dataset(
        dataset,
        splits=['train'],
        datasets_dir="/data",
    )

    print(f"Eliciting false justifications for train split ({len(train_data)} documents)")
    
    # Run false justification elicitation
    elicit_fj_for_split(
        chat_wrapper=chat_wrapper,
        split_data=train_data,
        split_name='train',
        temps=temps,
        num_trials=num_trials,
        styles=styles,
        target_style=target_style,
        target_temp=target_temp,
        run_name=args_name,
        use_lora=False,  # Explicitly disabled
        lora_run_name=None,
        artifact_name=None,
        results_dir="/results/results",  # Maps to volume
        continue_mode=continue_mode,
    )

    print("False justification elicitation complete!")
    results_volume.commit()


@app.local_entrypoint()
def main(*arglist):
    """
    Local entrypoint - runs on your machine.
    Launches remote false justification elicitation.
    """
    # Parse command line arguments
    continue_mode = len(arglist) > 0 and arglist[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(arglist) - (1 if continue_mode else 0)
    
    if effective_argc != 2:
        raise ValueError(
            "Usage:\n"
            "  modal run false_justification_modal.py /path/to/yaml/args.yaml temp{temp}_style{style} [continue]"
        )

    config_path = arglist[0]
    sft_key = arglist[1]
    
    # Parse the target temperature and style from the sft_key
    target_temp, target_style = parse_forward_sft_key(sft_key)
    
    args = YamlConfig(config_path)

    print("Starting Modal false justification elicitation job...")
    print(f"Target: {sft_key} (temp={target_temp}, style={target_style})")
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
        
    result = run_elicit_false_justifications.remote(
        model_name=args.model_name,
        dataset=args.dataset,
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        args_name=args.args_name,
        target_temp=target_temp,
        target_style=target_style,
        continue_mode=continue_mode,
    )

    print("Remote false justification elicitation completed!")
    return result