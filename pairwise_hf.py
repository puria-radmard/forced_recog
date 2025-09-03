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

from prompts import (
    DETECTION_SYSTEM_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY
)

from utils.elicit import get_choice_token_logits_from_token_ids
from utils.util import YamlConfig


def get_choice_tokens(chat_wrapper: ChatTemplateWrapper) -> List[List[int]]:
    """Get token IDs for choice responses "1" and "2"."""
    choice_strings = [["1"], ["2"]]
    choice_tokens = []
    
    for option_str_list in choice_strings:
        option_tokens = []
        for option_str in option_str_list:
            token_ids = chat_wrapper.tokenizer.encode(option_str, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Choice token '{option_str}' produces {len(token_ids)} tokens: {token_ids}. "
                    f"All choice tokens must be exactly one token."
                )
            option_tokens.extend(token_ids)
        choice_tokens.append(option_tokens)
    
    return choice_tokens



def elicit_choices_for_split(
    chat_wrapper: ChatTemplateWrapper,
    split_data: pd.DataFrame,
    split_name: str,
    temps: List[float],
    num_trials: List[int],
    styles: List[str | None],
    run_name: str,
    use_lora: bool = False,
    lora_run_name: str = None,
    artifact_name: str = None,
    results_dir: Optional[str] = None
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
    """
    # Determine output directory based on whether using LoRA
    results_dir = results_dir or 'results_and_data/results'
    if use_lora:
        output_dir = f"{results_dir}/main/{run_name}/{split_name}/forward_sft_choices/{lora_run_name}/{artifact_name}"
        results_file = f"{output_dir}/choice_results.csv"
        print(f"Using LoRA model - saving to: {results_file}")
    else:
        output_dir = f"{results_dir}/main/{run_name}/{split_name}/initial_choices"
        results_file = f"{output_dir}/choice_results.csv"
        print(f"Using base model - saving to: {results_file}")
    
    # Create output directory and initialize results file
    os.makedirs(output_dir, exist_ok=True)

    header_df = pd.DataFrame(columns=[
        'document_idx', 
        'summary1_temp', 'summary1_trial', 'summary1_style',
        'summary2_temp', 'summary2_trial', 'summary2_style',
        'prob_choice_1', 'prob_choice_2', 
    ])
    header_df.to_csv(results_file, index=False)

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
                try:
                    # summary_df = load_model_summaries(run_name, split_name, temp, trial_idx, style)
                    summary_df = all_summaries[(temp, trial_idx, style)]
                    # Find this document's summary
                    doc_summary = summary_df[summary_df['document_idx'] == document_idx]
                    if len(doc_summary) > 0:
                        summaries[(temp, trial_idx, style)] = doc_summary['summary'].iloc[0]
                except FileNotFoundError:
                    print(f"Warning: Missing summary file for T={temp}, style = {style}, trial={trial_idx}")
                    continue
        
        if len(summaries) == 0:
            continue

        torch.cuda.empty_cache()

        # # FIXME: prompt caching is not working :(
        # # try:
        # #     cache_info = chat_wrapper.create_prompt_cache(
        # #         system_prompt=DETECTION_SYSTEM_PROMPT,
        # #         user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article),
        # #         user_message_unfinished = True
        # #     )
        # # except torch.OutOfMemoryError:
        # #     continue

        all_keys = list(summaries.keys())
        for setting_key_1 in all_keys:
            for setting_key_2 in all_keys:

                # Only compare unique combinations (order dealt with below)
                if setting_key_1 == setting_key_2:
                    break

                summary_1 = summaries[setting_key_1]
                summary_2 = summaries[setting_key_2]
                

                forward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1 = summary_1, summary_2 = summary_2) 
                    + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                forward_prompt_full = chat_wrapper.format_chat(
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article) + forward_prompt_raw,
                    prefiller=""
                )        
        
                backward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1 = summary_2, summary_2 = summary_1) 
                    + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )
                backward_prompt_full = chat_wrapper.format_chat(
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article) + backward_prompt_raw,
                    prefiller=""
                )

                # # FIXME: prompt caching is not working :(
                # # assert forward_prompt_full.startswith(cache_info['formatted_prompt'])
                # # forward_prompt = forward_prompt_full.removeprefix(cache_info['formatted_prompt'])

                # # assert backward_prompt_full.startswith(cache_info['formatted_prompt'])
                # # backward_prompt = backward_prompt_full.removeprefix(cache_info['formatted_prompt'])

                # # both_outputs = chat_wrapper.forward(
                # #     chats=[forward_prompt, backward_prompt],
                # #     past_key_values=copy.deepcopy(cache_info["cache"]),
                # # )
                # # both_probs = get_choice_token_logits_from_token_ids(both_outputs.logits, choice_tokens)
                # # forward_probs, backward_probs = both_probs

                # # # XXX: these were used as test cases
                # # # front_probs_og = get_choice_token_logits_from_token_ids(chat_wrapper.forward(chats=[forward_prompt], past_key_values=copy.deepcopy(cache_info["cache"])).logits, choice_tokens)
                # # # back_probs_og = get_choice_token_logits_from_token_ids(chat_wrapper.forward(chats=[backward_prompt], past_key_values=copy.deepcopy(cache_info["cache"])).logits, choice_tokens)

                try:
                    forward_probs = get_choice_token_logits_from_token_ids(chat_wrapper.forward(chats=[forward_prompt_full]).logits, choice_tokens)
                    backward_probs = get_choice_token_logits_from_token_ids(chat_wrapper.forward(chats=[backward_prompt_full]).logits, choice_tokens)
                except torch.OutOfMemoryError:
                    continue

                # # chat_wrapper.generate([backward_prompt], )

                # Store results
                new_results = pd.DataFrame([
                    {
                        'document_idx': document_idx,
                        'summary1_temp': setting_key_1[0],
                        'summary1_trial': setting_key_1[1], 
                        'summary1_style': setting_key_1[2], 
                        'summary2_temp': setting_key_2[0],
                        'summary2_trial': setting_key_2[1],
                        'summary2_style': setting_key_2[2],
                        'prob_choice_1': forward_probs[0,0].item(),
                        'prob_choice_2': forward_probs[0,1].item(),
                    },
                    {
                        'document_idx': document_idx,
                        'summary1_temp': setting_key_2[0],
                        'summary1_trial': setting_key_2[1],
                        'summary1_style': setting_key_2[2],
                        'summary2_temp': setting_key_1[0], 
                        'summary2_trial': setting_key_1[1],
                        'summary2_style': setting_key_1[2],
                        'prob_choice_1': backward_probs[0,0].item(),
                        'prob_choice_2': backward_probs[0,1].item(),
                    }
                ])
    
                # Save results
                new_results.to_csv(results_file, mode='a', header=False, index=False)


if __name__ == "__main__":

    if len(sys.argv) not in [2, 4]:
        print("Usage:")
        print("  Base model: python -m elicit_choices.py /path/to/yaml/args.yaml")
        print("  With LoRA:  python -m elicit_choices.py /path/to/yaml/args.yaml <wandb_run_name> <artifact_name>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    use_lora = len(sys.argv) == 4
    
    if use_lora:
        wandb_run_name = sys.argv[2]
        artifact_name = sys.argv[3]
        print(f"Running with LoRA adapters:")
        print(f"  WandB Run: {wandb_run_name}")
        print(f"  Artifact: {artifact_name}")
    else:
        wandb_run_name = None
        artifact_name = None
        print("Running with base model")
    
    args = YamlConfig(config_path)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_name)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_data, test_data, validation_data = load_dataset(args.dataset)
    
    # Map split names to data
    split_data_map = {
        'train': train_data,
        'test': test_data,
        'validation': validation_data
    }
    
    # Elicit choices for each requested split
    split_name = "test"
    if split_name not in split_data_map:
        raise FileNotFoundError(f"Warning: Split '{split_name}' not found. Available: {list(split_data_map.keys())}")
        
    split_data = split_data_map[split_name]
    print(f"Eliciting choices for {split_name} split ({len(split_data)} documents)")
    
    elicit_choices_for_split(
        chat_wrapper=chat_wrapper,
        split_data=split_data,
        split_name=split_name,
        temps=args.temps,
        num_trials=args.num_trials,
        styles=args.styles,
        run_name=args.args_name,
        use_lora=use_lora,
        lora_run_name=wandb_run_name,
        artifact_name=artifact_name
    )
    
    print("Choice elicitation complete!")