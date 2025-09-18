import os
from dotenv import load_dotenv
import modal

from typing import List, Optional
import traceback
import re
import wandb

from model.load import load_model
from sft_utils.lora import download_and_apply_lora
from load_data import load_dataset
from util.util import YamlConfig
from pairwise_hf_Jesse import elicit_choices_for_split

# Load environment variables from .env file
load_dotenv()
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
    .add_local_python_source("util")
    .add_local_python_source("prompts")
    .add_local_python_source("load_data")
    .add_local_python_source("pairwise_hf")  # so we can import elicit_choices_for_split
    .add_local_python_source("pairwise_hf_Jesse")  # Jesse's version with max_documents/max_pairs support
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
    continue_mode: bool = False,
    max_documents: int = 2,
    max_pairs: int = 1,
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

    print(f"Max documents: {max_documents}")
    print(f"Max pairs per document: {max_pairs}")
    
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")

    # Load model
    print(f"Loading model: {model_name}")
    chat_wrapper = load_model(model_name, device="auto")

    # Load dataset
    print(f"Loading dataset: {dataset}")
    _, test_data, _ = load_dataset(
        dataset,
        splits=['test'],
        datasets_dir="/data",
    )

    if use_lora and artifact_name == "all":
        # Loop over all LoRA artifacts for this run
        print(f"Finding all LoRA artifacts for run: {wandb_run_name}")
        
        # Initialize WandB API
        load_dotenv()
        project_name = os.getenv('WANDB_PROJECT')
        api = wandb.Api()
        
        # Get the specific run and its artifacts
        try:
            wandb_runs = list(api.runs(path=project_name, filters={"display_name": {"$regex": f".*{wandb_run_name}.*"}}))
            assert len(wandb_runs) == 1, f"wandb_runs returned {wandb_runs}"
            wandb_run = wandb_runs[0]
            artifacts = wandb_run.logged_artifacts()
        except Exception as e:
            raise ValueError(f"Could not find run {wandb_run_name} in project {project_name}: {e}")
    
        # Filter artifacts that belong to this run and contain "lora_adapters"
        relevant_artifacts = []
        for artifact in artifacts:
            if "lora_adapters" in artifact.name:
                # Extract step number for sorting
                step_match = re.search(r'step_(\d+)', artifact.name)
                if step_match:
                    step = int(step_match.group(1))
                    relevant_artifacts.append((artifact, step))
        
        # Sort by step number
        relevant_artifacts.sort(key=lambda x: x[1])
        
        print(f"Found {len(relevant_artifacts)} LoRA artifacts")
        
        for artifact, step in relevant_artifacts:
            # Extract artifact suffix (everything after run name)
            artifact_suffix = artifact.name.split(f"{wandb_run_name}.")[-1].split(":")[0]  # Remove version
            
            print(f"\n{'='*60}")
            print(f"Processing artifact: {artifact.name}")
            print(f"Step: {step}, Suffix: {artifact_suffix}")
            if continue_mode:
                print("Continue mode: Will resume from existing results for this artifact")
            print(f"{'='*60}")
            
            try:
                # Apply this specific LoRA
                chat_wrapper = download_and_apply_lora(
                    chat_wrapper, wandb_run_name, artifact_suffix
                )
                
                # Run elicitation with this adapter
                elicit_choices_for_split(
                    chat_wrapper=chat_wrapper,
                    split_data=test_data,
                    split_name='test',
                    temps=temps,
                    num_trials=num_trials,
                    styles=styles,
                    run_name=args_name,
                    use_lora=True,
                    lora_run_name=wandb_run_name,
                    artifact_name=artifact_suffix,
                    results_dir="/results/results",
                    continue_mode=continue_mode,
                    max_documents=max_documents,
                    max_pairs=max_pairs,
                )
                
                print(f"✅ Completed processing artifact: {artifact_suffix}")
                
            except Exception as e:
                print(f"❌ Error processing artifact {artifact_suffix}:")
                print(traceback.format_exc())
                print(f"Continuing to next artifact...\n")
                continue
            
            chat_wrapper.model = chat_wrapper.model.unload()
    
    else:
        # Single artifact mode or base model
        if use_lora:
            # Single artifact mode
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
            results_dir="/results/results",
            continue_mode=continue_mode,
            max_documents=max_documents,
            max_pairs=max_pairs,
        )

    print("Choice elicitation complete!")
    results_volume.commit()


@app.local_entrypoint()
def main(*arglist):
    """
    Local entrypoint - runs on your machine.
    Launches remote choice elicitation.
    """
    # Parse command line arguments
    continue_mode = len(arglist) > 0 and arglist[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(arglist) - (1 if continue_mode else 0)
    
    if effective_argc not in [1, 2, 3, 4, 5]:
        raise ValueError(
            "Usage:\n"
            "  Base model: modal run pairwise_hf_modal_Jesse.py <config_path> [max_documents] [max_pairs] [continue]\n"
            "  With LoRA:  modal run pairwise_hf_modal_Jesse.py <config_path> <wandb_run_name> <artifact_name> [max_documents] [max_pairs] [continue]\n"
            "Examples:\n"
            "  modal run pairwise_hf_modal_Jesse.py configs/test_config.yaml 2 1\n"
            "  modal run pairwise_hf_modal_Jesse.py configs/test_config.yaml 5 2 continue\n"
            "  modal run pairwise_hf_modal_Jesse.py configs/test_config.yaml wandb_run artifact 3 1"
        )

    config_path = arglist[0]
    args = YamlConfig(config_path)

    # Parse arguments - check if second arg is a number (base model) or string (LoRA)
    if len(arglist) > 1 and arglist[1].isdigit():
        # Base model with optional max_documents and max_pairs
        wandb_run_name = None
        artifact_name = None
        max_documents = int(arglist[1]) if len(arglist) > 1 else 2
        max_pairs = int(arglist[2]) if len(arglist) > 2 and arglist[2].isdigit() else 1
    else:
        # LoRA model
        wandb_run_name = arglist[1] if len(arglist) > 1 else None
        artifact_name = arglist[2] if len(arglist) > 2 else None
        max_documents = int(arglist[3]) if len(arglist) > 3 and arglist[3].isdigit() else 2
        max_pairs = int(arglist[4]) if len(arglist) > 4 and arglist[4].isdigit() else 1

    print("Starting Modal choice elicitation job...")
    print(f"Max documents: {max_documents}")
    print(f"Max pairs per document: {max_pairs}")
    
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
        
    result = run_elicit_choices.remote(
        model_name=args.model_name,
        dataset=args.dataset,
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        args_name=args.args_name,
        wandb_run_name=wandb_run_name,
        artifact_name=artifact_name,
        continue_mode=continue_mode,
        max_documents=max_documents,
        max_pairs=max_pairs,
    )

    print("Remote choice elicitation completed!")
    return result