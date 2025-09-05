import tempfile
import wandb
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import wandb
import os
import tempfile
from model.base import ChatTemplateWrapper
from dotenv import load_dotenv



def setup_lora_model(model, lora_config):
    """Setup LoRA on the model."""
    lora_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config['lora_r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules']
    )
    
    return get_peft_model(model, lora_peft_config)



def save_lora_as_artifact(model, step: int, temp: float, style: str, run_name: str):
    """Save LoRA adapters as WandB artifact."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_path = os.path.join(temp_dir, "adapter")
        model.save_pretrained(adapter_path)
        
        # Create artifact
        artifact_name = f"{run_name}.lora_adapters_step_{step}"
        artifact = wandb.Artifact(
            artifact_name,
            type="model",
            description=f"LoRA adapters at step {step} for temp={temp}, style={style}, run={run_name}"
        )
        artifact.add_dir(adapter_path)
        wandb.log_artifact(artifact)
        
        print(f"Logged LoRA adapters artifact: {artifact_name}")



def download_and_apply_lora(chat_wrapper: ChatTemplateWrapper, wandb_run_name: str, artifact_suffix: str, *_, adapter_name: str = "default"):
    """
    Download LoRA adapters from WandB and apply them to the model.
    
    Args:
        chat_wrapper: The loaded base model wrapper
        wandb_run_name: WandB run name (used for path construction and artifact prefix)
        artifact_suffix: Suffix of the artifact (e.g., "lora_adapters_step_100")
        
    Returns:
        Updated chat_wrapper with LoRA adapters applied
    """
    # Construct full artifact name
    full_artifact_name = f"{wandb_run_name}.{artifact_suffix}"
    
    print(f"Downloading LoRA adapters from WandB...")
    print(f"  Run: {wandb_run_name}")
    print(f"  Artifact suffix: {artifact_suffix}")
    print(f"  Full artifact name: {full_artifact_name}")
    
    # Initialize WandB (need project name from environment)
    load_dotenv()
    project_name = os.getenv('WANDB_PROJECT')
    if not project_name:
        raise ValueError("WANDB_PROJECT environment variable not set")
    
    # Download artifact to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize WandB API
        api = wandb.Api()
        
        # Get the artifact directly by name (since names are now globally unique)
        artifact_path = f"{project_name}/{full_artifact_name}:latest"
        artifact = api.artifact(artifact_path)
        
        # Download to temp directory
        adapter_dir = artifact.download(root=temp_dir)
        print(f"Downloaded adapters to: {adapter_dir}")
        
        # Apply LoRA adapters to the model
        print("Applying LoRA adapters to model...")
        chat_wrapper.model = PeftModel.from_pretrained(
            chat_wrapper.model,
            adapter_dir,
            adapter_name=adapter_name,
            is_trainable=False
        )
        print("LoRA adapters applied successfully!")
    
    return chat_wrapper
