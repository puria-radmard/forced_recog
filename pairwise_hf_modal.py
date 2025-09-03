import modal
from typing import Dict, List, Optional

from model.load import load_model
from sft_utils.lora import download_and_apply_lora
from load_data import load_dataset
from utils.util import YamlConfig
from pairwise_hf import elicit_choices_for_split


# Define Modal app
app = modal.App("elicit-choices")

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
    .add_local_python_source("pairwise_hf")  # so we can import elicit_choices_for_split
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
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ],
    timeout=10800,
    memory=32768,
)
def run_elicit_choices(
    model_name: str,
    dataset: str,
    temps: List[float],
    num_trials: List[int],
    styles: List[str],
    args_name: str,
    wandb_run_name: Optional[str] = None,
    artifact_name: Optional[str] = None,
):
    """
    Run pairwise choice elicitation (base model or with LoRA).
    """

    # Decide LoRA usage
    use_lora = wandb_run_name is not None and artifact_name is not None
    if use_lora:
        print(f"Running with LoRA adapters:")
        print(f"  WandB Run: {wandb_run_name}")
        print(f"  Artifact: {artifact_name}")
    else:
        print("Running with base model")

    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device="auto", override_path="/models")

    # Load dataset
    print(f"Loading dataset: {dataset}")
    _, test_data, _ = load_dataset(
        dataset,
        splits=['test'],
        datasets_dir="/data",
    )

    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_name)
        
    elicit_choices_for_split(
        chat_wrapper=chat_wrapper,
        split_data=test_data,
        split_name='test',
        temps=temps,
        num_trials=num_trials,
        styles=styles,
        run_name=args_name,
        use_lora=use_lora,
        lora_run_name=wandb_run_name,
        artifact_name=artifact_name,
        results_dir="/results/results",  # <-- ensure it writes to volume
    )

    print("Choice elicitation complete!")

    # Commit volume to persist outputs
    results_volume.commit()


@app.local_entrypoint()
def main(*arglist):
    """
    Local entrypoint - runs on your machine.
    Launches remote choice elicitation.
    """
    if len(arglist) not in [1, 3]:
        raise ValueError(
            "Usage:\n"
            "  Base model: modal run elicit_choices_modal.py /path/to/yaml/args.yaml\n"
            "  With LoRA:  modal run elicit_choices_modal.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_name>"
        )

    config_path = arglist[0]
    args = YamlConfig(config_path)

    if len(arglist) == 3:
        wandb_run_name = arglist[1]
        artifact_name = arglist[2]
    else:
        wandb_run_name = None
        artifact_name = None

    print("Starting Modal choice elicitation job...")
    result = run_elicit_choices.remote(
        model_name=args.model_name,
        dataset=args.dataset,
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        args_name=args.args_name,
        wandb_run_name=wandb_run_name,
        artifact_name=artifact_name,
    )

    print("Remote choice elicitation completed!")
    return result
