import os
import tempfile
from dotenv import load_dotenv
import wandb
from peft import PeftModel

import sys

from model.load import load_model
from sft_utils.lora import download_and_apply_lora
from utils.util import YamlConfig

# e.g. python test_peft_unload.py scripts_summary/configs/mistral24b_forward_training.yaml mistral24b_forward_training_forwardsft_temp0.0_styleeconomist_seed42 lora_adapters_step_150

# Load config (assuming you have a simple yaml file with model_name)
args = YamlConfig(sys.argv[1])  # Replace with actual path

# Load base model
print("Loading base model...")
chat_wrapper = load_model(args.model_name, device="auto")

# Test prompt
test_prompt = "The capital of France is"

# Generation 1: Base model
print("\n=== Generation 1: Base model ===")
gen1 = chat_wrapper.forward([test_prompt], do_sample = False, temperature=None, max_new_tokens=10)
print(f"Gen1: {gen1}")

# Apply LoRA adapters
print("\n=== Applying LoRA adapters ===")
load_dotenv()
project_name = os.getenv('WANDB_PROJECT')
wandb_run_name = sys.argv[2]  # Replace with actual run name
artifact_suffix = sys.argv[3]  # Replace with actual artifact suffix

download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_suffix)

# Generation 2: With LoRA
print("\n=== Generation 2: With LoRA ===")
gen2 = chat_wrapper.forward(test_prompt, do_sample = False, temperature=None, max_new_tokens=10)
print(f"Gen2: {gen2}")

# Unload LoRA
print("\n=== Unloading LoRA ===")
chat_wrapper.model = chat_wrapper.model.unload()
print("LoRA unloaded!")

# Generation 3: Back to base model (should match gen1)
print("\n=== Generation 3: After unloading LoRA ===")
gen3 = chat_wrapper.forward(test_prompt, do_sample = False, temperature=None, max_new_tokens=10)
print(f"Gen3: {gen3}")

# Check results
print("\n=== Results ===")
print(f"Gen1 (base): {gen1['logits'][0,-1]}")
print(f"Gen2 (LoRA): {gen2['logits'][0,-1]}")
print(f"Gen3 (unloaded): {gen3['logits'][0,-1]}")

print(f"\nGen1 == Gen3 (should be True): {(gen1['logits'] == gen3['logits']).all()}")
print(f"Gen1 == Gen2 (should be False): {(gen1['logits'] == gen2['logits']).all()}")
print(f"Gen2 == Gen3 (should be False): {(gen2['logits'] == gen3['logits']).all()}")

if (gen1['logits'] == gen3['logits']).all():
    print("✅ SUCCESS: Unloading works correctly!")
else:
    print("❌ FAILURE: Unloading may not be working properly!")