import pandas as pd
import os
import yaml
import torch
from tqdm import tqdm
from typing import List, Optional
import sys
import copy

from load_data import load_dataset, load_model_summaries
from model.load import load_model
from model.base import ChatTemplateWrapper

from sft_utils.lora import download_and_apply_lora

from prompts.summary import (
    DETECTION_FJ_SYSTEM_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT, 
    DETECTION_FJ_PROMPT_TEMPLATE_VS_MODEL_QUESTION, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY
)

from sft_utils.train import parse_forward_sft_key
from utils.util import YamlConfig

from scripts_summary.pairwise_selection.pairwise_hf import get_choice_tokens, load_completion_status



document_columns = [
    'document_idx', 
    'summary1_temp', 'summary1_trial', 'summary1_style',
    'summary2_temp', 'summary2_trial', 'summary2_style',
    'false_choice', 'justification'
]



def elicit_fj_for_split(
    chat_wrapper: ChatTemplateWrapper,
    split_data: pd.DataFrame,
    split_name: str,
    temps: List[float],
    num_trials: List[int],
    styles: List[str | None],
    target_style: str,
    target_temp: str,
    run_name: str,
    use_lora: bool = False,
    lora_run_name: str = None,
    artifact_name: str = None,
    results_dir: Optional[str] = None,
    continue_mode: bool = False
) -> None:
    """
    Elicit pairwise self-recognition choices for a dataset split.
    
    Args:
        chat_wrapper: Loaded model wrapper
        split_data: DataFrame with columns [document_idx, article, summary]
        split_name: Name of split (test/validation/train)
        temps: List of temperatures used in generation
        num_trials: List of number of trials per temperature
        styles: List of styles to prompt with, keys to prompts.STYLE_SYSTEM_PROMPTS
        run_name: Name of the run
        use_lora: Whether to use LoRA adapters
        lora_run_name: WandB run name for LoRA adapters
        artifact_name: Artifact name for LoRA adapters
        continue_mode: Whether to continue from existing results
    """
    # Determine output directory based on whether using LoRA
    results_dir = results_dir or 'results_and_data/results'
    if use_lora:
        raise Exception("Don't do this")
        output_dir = f"{results_dir}/main/{run_name}/{split_name}/forward_sft_choices/{lora_run_name}/{artifact_name}"
        results_file = f"{output_dir}/choice_results.csv"
        print(f"Using LoRA model - saving to: {results_file}")
    else:
        output_dir = f"{results_dir}/main/{run_name}/{split_name}/false_choice_justifications"
        results_file = f"{output_dir}/temp{target_temp}_style{target_style}.csv"
        print(f"Using base model - saving to: {results_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load completion status if in continue mode
    completion_status = {}
    if continue_mode:
        completion_status = load_completion_status(results_file)
        if completion_status:
            print(f"Continue mode: Found existing results for {len(completion_status)} documents")
        else:
            print("Continue mode: No existing results found, starting fresh")
            header_df = pd.DataFrame(columns=document_columns)
            header_df.to_csv(results_file, index=False)
    else:
        # Create fresh CSV file
        header_df = pd.DataFrame(columns=document_columns)
        header_df.to_csv(results_file, index=False)
        print("Fresh run: Created new results file")

    # Get choice tokens
    choice_tokens = get_choice_tokens(chat_wrapper)

    # Collect summaries
    all_summaries = {}
    for temp, num_trial, style in zip(temps, num_trials, styles):
        for trial_idx in range(num_trial):
            all_summaries[(temp, trial_idx, style)] = load_model_summaries(run_name, split_name, temp, trial_idx, style, results_dir=results_dir)

    for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Eliciting choices for {split_name}"):

        document_idx = row['document_idx']
        article = row['article']

        # Load all generated summaries for this document
        summaries = {}
        for temp, num_trial, style in zip(temps, num_trials, styles):
            for trial_idx in range(num_trial):
                summary_df = all_summaries[(temp, trial_idx, style)]
                # Find this document's summary
                doc_summary = summary_df[summary_df['document_idx'] == document_idx]
                if len(doc_summary) > 0:
                    summaries[(temp, trial_idx, style)] = doc_summary['summary'].iloc[0]
    
        if len(summaries) == 0:
            continue
        
        # Get all keys for target style and non-target styles
        target_keys = [key for key in summaries.keys() if (key[0] == target_temp) and (key[2] == target_style)]
        non_target_keys = [key for key in summaries.keys() if (key[0] != target_temp) or (key[2] != target_style)]
        
        if len(target_keys) == 0 or len(non_target_keys) == 0:
            continue

        # Get all possible comparison pairs for this document
        expected_pairs = set()
        for target_key in target_keys:
            for non_target_key in non_target_keys:
                if target_key != non_target_key:
                    expected_pairs.add((target_key, non_target_key))
                else:
                    raise ValueError

        # Check completion status for this document
        completed_pairs = completion_status.get(document_idx, set())
        
        # Skip entire document if all expected pairs are completed
        if continue_mode and expected_pairs.issubset(completed_pairs):
            continue

        torch.cuda.empty_cache()

        for target_key in target_keys:
            for non_target_key in non_target_keys:

                # Skip this specific comparison if already completed
                if continue_mode and (target_key, non_target_key) in completed_pairs and (non_target_key, target_key) in completed_pairs:
                    continue

                target_summary = summaries[target_key]
                non_target_summary = summaries[non_target_key]
                
                forward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1 = target_summary, summary_2 = non_target_summary) 
                    + DETECTION_FJ_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                user_message_forwards = DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article) + forward_prompt_raw
        
                backward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1 = non_target_summary, summary_2 = target_summary) 
                    + DETECTION_FJ_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                user_message_backwards = DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article) + backward_prompt_raw

                try:
                    forward_target_choice = "1"
                    backward_target_choice = "2"

                    forwards_followup_prompt = chat_wrapper.format_chat(
                        system_prompt=DETECTION_FJ_SYSTEM_PROMPT,
                        user_message=user_message_forwards,
                        prefiller=f"{forward_target_choice}, because",
                    )
                    forwards_followup = chat_wrapper.generate(
                        chats=[forwards_followup_prompt],
                    )['generated_texts'][0]

                    backwards_followup_prompt = chat_wrapper.format_chat(
                        system_prompt=DETECTION_FJ_SYSTEM_PROMPT,
                        user_message=user_message_backwards,
                        prefiller=f"{backward_target_choice}, because",
                    )
                    backwards_followup = chat_wrapper.generate(
                        chats=[backwards_followup_prompt],
                    )['generated_texts'][0]

                except torch.OutOfMemoryError:
                    continue

                # Store results
                new_results = pd.DataFrame([
                    {
                        'document_idx': document_idx,
                        'summary1_temp': target_key[0],
                        'summary1_trial': target_key[1], 
                        'summary1_style': target_key[2], 
                        'summary2_temp': non_target_key[0],
                        'summary2_trial': non_target_key[1],
                        'summary2_style': non_target_key[2],
                        'false_choice': forward_target_choice,
                        'justification': forwards_followup,
                    },
                    {
                        'document_idx': document_idx,
                        'summary1_temp': non_target_key[0],
                        'summary1_trial': non_target_key[1],
                        'summary1_style': non_target_key[2],
                        'summary2_temp': target_key[0], 
                        'summary2_trial': target_key[1],
                        'summary2_style': target_key[2],
                        'false_choice': backward_target_choice,
                        'justification': backwards_followup,
                    }
                ])
    
                # Save results
                new_results.to_csv(results_file, mode='a', header=False, index=False)

                # Update completion status (for this run)
                if document_idx not in completion_status:
                    completion_status[document_idx] = set()
                completion_status[document_idx].add((target_key, non_target_key))
                completion_status[document_idx].add((non_target_key, target_key))


if __name__ == "__main__":

    # Parse command line arguments
    continue_mode = len(sys.argv) > 1 and sys.argv[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(sys.argv) - (1 if continue_mode else 0)

    if effective_argc not in [3, 4]:
        print("Usage:")
        print("  Base model: python -m scripts_summary.pairwise_selection.pairwise_hf.py /path/to/yaml/args.yaml temp{temp}_style{style} [continue]")
        print("  With LoRA:  Do not do!!")
        sys.exit(1)
    
    
    config_path = sys.argv[1]
    sft_key = sys.argv[2]
    use_lora = effective_argc == 4

    target_temp, target_style = parse_forward_sft_key(sft_key)
    
    if use_lora:
        raise Exception
        wandb_run_name = sys.argv[2]
        artifact_name = sys.argv[3]
        print(f"Running with LoRA adapters:")
        print(f"  WandB Run: {wandb_run_name}")
        print(f"  Artifact: {artifact_name}")
    else:
        wandb_run_name = None
        artifact_name = None
        print("Running with base model")
    
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
    
    args = YamlConfig(config_path)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')

    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_name)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_data, _, _ = load_dataset(args.dataset, splits = ['train'])

    print(f"Eliciting choices for train split ({len(train_data)} documents)")

    elicit_fj_for_split(
        chat_wrapper=chat_wrapper,
        split_data=train_data,
        split_name='train',
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        target_style=target_style,
        target_temp=target_temp,
        run_name=args.args_name,
        use_lora=use_lora,
        lora_run_name=wandb_run_name,
        artifact_name=artifact_name,
        continue_mode=continue_mode
    )
    
    print("Choice elicitation complete!")