from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import os
import tempfile



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

