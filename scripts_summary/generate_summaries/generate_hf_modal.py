import modal
from typing import Dict, List, Optional

from model.load import load_model
from sft_utils.lora import download_and_apply_lora
from load_data import load_dataset
from utils.util import YamlConfig
from scripts_summary.generate_summaries.generate_hf import generate_summaries_for_split


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
    .add_local_python_source("utils")
    .add_local_python_source("prompts")
    .add_local_python_source("load_data")
    .add_local_python_source("scripts_summary.generate_summaries.generate_hf")  # so we can import scripts_summary.generate_summaries.generate_hf
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
    chat_wrapper = load_model(model_name, device="auto", override_path="/models")

    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_suffix, env_tmp_dir=False)

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
            "  Base model: modal run scripts_summary.generate_summaries.generate_hf_modal.py /path/to/yaml/args.yaml [continue]\n"
            "  With LoRA:  modal run scripts_summary.generate_summaries.generate_hf_modal.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_suffix> [continue]"
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
    return result