from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


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

