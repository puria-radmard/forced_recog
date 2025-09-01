import pandas as pd
import os
import yaml
from tqdm import tqdm
from typing import List
import sys
import copy
import tempfile
import wandb
from peft import PeftModel
from dotenv import load_dotenv

from load_data import load_dataset
from model.load import load_model
from model.base import ChatTemplateWrapper
from prompts import DATASET_SYSTEM_PROMPTS, SUMMARIZE_PROMPT_TEMPLATES, STYLE_PROMPT_ADDENDUM
from utils.util import YamlConfig


def download_and_apply_lora(chat_wrapper: ChatTemplateWrapper, wandb_run_name: str, artifact_suffix: str):
    """
    Download LoRA adapters from WandB and apply them to the model.
    
    Args:
        chat_wrapper: The loaded base model wrapper
        wandb_run_name: WandB run name (used for artifact prefix)
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
            is_trainable=False
        )
        print("LoRA adapters applied successfully!")
    
    return chat_wrapper


def generate_summaries_for_split(
    chat_wrapper: ChatTemplateWrapper,
    split_data: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    temps: List[float],
    num_trials: List[int],
    styles: List[str | None],
    run_name: str,
    use_lora: bool = False,
    lora_run_name: str = None,
    artifact_suffix: str = None
) -> None:
    """
    Generate summaries for a single dataset split.
    
    Args:
        chat_wrapper: Loaded model wrapper
        split_data: DataFrame with columns [document_idx, article, summary]
        dataset_name: Name of dataset (for prompting)
        split_name: Name of split (test/validation/train)
        temps: List of temperatures to use
        num_trials: List of number of trials per temperature
        styles: List of styles to prompt with, keys to STYLE_SYSTEM_PROMPTS
        run_name: Name of the run (for saving)
        use_lora: Whether using LoRA adapters
        lora_run_name: WandB run name for LoRA
        artifact_suffix: Artifact suffix for LoRA
    """
    # Determine base directory based on whether using LoRA
    if use_lora:
        base_dir = f"results_and_data/results/main/{run_name}/{split_name}/forward_sft_summaries/{lora_run_name}/{artifact_suffix}"
        print(f"Using LoRA model - saving to: {base_dir}")
    else:
        base_dir = f"results_and_data/results/main/{run_name}/{split_name}/model_summaries"
        print(f"Using base model - saving to: {base_dir}")
    
    # Create output directories
    os.makedirs(base_dir, exist_ok=True)
    
    # Get system prompt and user prompt template
    system_prompt = DATASET_SYSTEM_PROMPTS[dataset_name]
    user_prompt_template = SUMMARIZE_PROMPT_TEMPLATES[dataset_name]
    
    for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Processing {split_name}"):
        document_idx = row['document_idx']
        article = row['article']
        
        # Format user message
        user_message = user_prompt_template.format(article=article)
        
        # Create cache for this document (don't close user tags in case we want to add style instructions)
        cache_info = chat_wrapper.create_prompt_cache(
            system_prompt=system_prompt,
            user_message=user_message,
            user_message_unfinished=True
        )

        # Generate for each temp/trial combination
        for temp, num_trial, style in zip(temps, num_trials, styles):
            
            for trial_idx in range(num_trial):
            
                # Initialize CSV files if they don't exist
                output_file = f"{base_dir}/T{temp}_trial{trial_idx}_style{style}.csv"
                if not os.path.exists(output_file):
                    pd.DataFrame(columns=['document_idx', 'summary']).to_csv(output_file, index=False)

                # FIXME This should be hidden away please.
                extra_chat = "-" if style == 'natural' else f'-\n\n{STYLE_PROMPT_ADDENDUM[style]}'
                extra_chat_with_tags = chat_wrapper.format_chat(user_message = extra_chat, prefiller="").removeprefix(chat_wrapper.format_chat(user_message="-", user_message_unfinished=True))

                # Generate summary with empty chat (continues from cache)
                generation_result = chat_wrapper.generate(
                    chats=[extra_chat_with_tags],  # Empty string starts from cache point
                    past_key_values=copy.deepcopy(cache_info["cache"]),
                    past_key_values_str=cache_info["formatted_prompt"],
                    max_new_tokens=100,
                    temperature=temp,
                    do_sample=(temp > 0.0),
                    use_cache_position = False,
                    skip_special_tokens = True,
                    return_full_text = False,
                )
                
                summary = generation_result["generated_texts"][0].strip().replace('\n', '\\n')
                
                # Save to appropriate trial file
                result_row = pd.DataFrame({
                    'document_idx': [document_idx],
                    'summary': [summary]
                })
                result_row.to_csv(output_file, mode='a', header=False, index=False)


if __name__ == "__main__":

    if len(sys.argv) not in [2, 4]:
        print("Usage:")
        print("  Base model: python -m generate_hf.py /path/to/yaml/args.yaml")
        print("  With LoRA:  python -m generate_hf.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_suffix>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    use_lora = len(sys.argv) == 4
    
    if use_lora:
        wandb_run_name = sys.argv[2]
        artifact_suffix = sys.argv[3]
        print(f"Running with LoRA adapters:")
        print(f"  WandB Run: {wandb_run_name}")
        print(f"  Artifact Suffix: {artifact_suffix}")
    else:
        wandb_run_name = None
        artifact_suffix = None
        print("Running with base model")
    
    args = YamlConfig(config_path)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_suffix)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_data, test_data, validation_data = load_dataset(args.dataset, splits=args.splits)
    
    # Map split names to data
    split_data_map = {
        'train': train_data,
        'test': test_data,
        'validation': validation_data
    }
    
    # Generate summaries for each requested split
    for split_name in args.splits:
            
        split_data = split_data_map[split_name]
        print(f"Generating summaries for {split_name} split ({len(split_data)} documents)")
        
        generate_summaries_for_split(
            chat_wrapper=chat_wrapper,
            split_data=split_data,
            dataset_name=args.dataset,
            split_name=split_name,
            temps=args.temps,
            num_trials=args.num_trials,
            styles=args.styles,
            run_name=args.args_name,
            use_lora=use_lora,
            lora_run_name=wandb_run_name,
            artifact_suffix=artifact_suffix
        )
    
    print("Generation complete!")
