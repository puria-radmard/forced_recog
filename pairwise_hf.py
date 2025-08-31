import pandas as pd
import os
import yaml
import torch
from tqdm import tqdm
from typing import List, Tuple
import sys
import copy

from load_data import load_dataset, load_model_summaries
from model.load import load_model
from model.base import ChatTemplateWrapper

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
    run_name: str
) -> None:
    """
    Elicit pairwise self-recognition choices for a dataset split.
    
    Args:
        chat_wrapper: Loaded model wrapper
        split_data: DataFrame with columns [document_idx, article, summary]
        dataset_name: Name of dataset
        split_name: Name of split (test/validation/train)
        temps: List of temperatures used in generation
        num_trials: List of number of trials per temperature
        run_name: Name of the run
    """
    # Create output directory and initialise results file
    output_dir = f"results_and_data/results/e1_temperature_comparison/{run_name}/{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/choice_results.csv"

    header_df = pd.DataFrame(columns=[
        'document_idx', 
        'summary1_temp', 'summary1_trial', 
        'summary2_temp', 'summary2_trial', 
        'order', 'prob_choice_1', 'prob_choice_2'
    ])
    header_df.to_csv(results_file, index=False)

    # Get choice tokens
    choice_tokens = get_choice_tokens(chat_wrapper)

    for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Eliciting choices for {split_name}"):
        document_idx = row['document_idx']
        article = row['article']
        
        # Load all generated summaries for this document
        summaries = {}
        for temp, num_trial in zip(temps, num_trials):
            for trial_idx in range(num_trial):
                try:
                    summary_df = load_model_summaries(run_name, split_name, temp, trial_idx)
                    # Find this document's summary
                    doc_summary = summary_df[summary_df['document_idx'] == document_idx]
                    if len(doc_summary) > 0:
                        summaries[(temp, trial_idx)] = doc_summary['summary'].iloc[0]
                except FileNotFoundError:
                    print(f"Warning: Missing summary file for T={temp}, trial={trial_idx}")
                    continue
        
        cache_info = chat_wrapper.create_prompt_cache(
            system_prompt=DETECTION_SYSTEM_PROMPT,
            user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article),
            user_message_unfinished = True
        )
        
        # Compare T=0.0 summaries vs T=1.0 summaries
        t_low_summaries = [(temp, trial) for temp, trial in summaries.keys() if temp == temps[0]]  # T=0.0
        t_high_summaries = [(temp, trial) for temp, trial in summaries.keys() if temp == temps[1]]  # T=1.0
        
        for t_low_key in t_low_summaries:
            for t_high_key in t_high_summaries:

                t_low_summary = summaries[t_low_key]
                t_high_summary = summaries[t_high_key]
                
                # Forward order: T=0.0 as Summary1, T=1.0 as Summary2
                forward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1 = t_low_summary, summary_2 = t_high_summary) 
                    + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )

                forward_prompt_full = chat_wrapper.format_chat(
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article) + forward_prompt_raw,
                    prefiller=""
                )                
                assert forward_prompt_full.startswith(cache_info['formatted_prompt'])
                forward_prompt = forward_prompt_full.removeprefix(cache_info['formatted_prompt'])
                
                forward_output = chat_wrapper.forward(
                    chats=[forward_prompt],
                    past_key_values=copy.deepcopy(cache_info["cache"]),
                )
                forward_probs = get_choice_token_logits_from_token_ids(
                    forward_output.logits, choice_tokens
                )
                
                # Backward order: T=1.0 as Summary1, T=0.0 as Summary2  
                backward_prompt_raw = (
                    DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1 = t_high_summary, summary_2 = t_low_summary) 
                    + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
                )

                backward_prompt_full = chat_wrapper.format_chat(
                    system_prompt=DETECTION_SYSTEM_PROMPT,
                    user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article = article) + backward_prompt_raw,
                    prefiller=""
                )  
                assert backward_prompt_full.startswith(cache_info['formatted_prompt'])
                backward_prompt = backward_prompt_full.removeprefix(cache_info['formatted_prompt'])
                
                backward_output = chat_wrapper.forward(
                    chats=[backward_prompt],
                    past_key_values=copy.deepcopy(cache_info["cache"]),
                )
                backward_probs = get_choice_token_logits_from_token_ids(
                    backward_output.logits, choice_tokens
                )
                
                # Store results
                new_results = pd.DataFrame([
                    {
                        'document_idx': document_idx,
                        'summary1_temp': t_low_key[0],
                        'summary1_trial': t_low_key[1], 
                        'summary2_temp': t_high_key[0],
                        'summary2_trial': t_high_key[1],
                        'order': 'forward',
                        'prob_choice_1': forward_probs[0, 0].item(),
                        'prob_choice_2': forward_probs[0, 1].item()
                    },
                    {
                        'document_idx': document_idx,
                        'summary1_temp': t_high_key[0],
                        'summary1_trial': t_high_key[1],
                        'summary2_temp': t_low_key[0], 
                        'summary2_trial': t_low_key[1],
                        'order': 'backward',
                        'prob_choice_1': backward_probs[0, 0].item(),
                        'prob_choice_2': backward_probs[0, 1].item()
                    }
                ])
    
                # Save results
                new_results.to_csv(results_file, mode='a', header=False, index=False)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python -m elicit_choices.py /path/to/yaml/args.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
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
    for split_name in args.splits:
        if split_name not in split_data_map:
            print(f"Warning: Split '{split_name}' not found. Available: {list(split_data_map.keys())}")
            continue
            
        split_data = split_data_map[split_name]
        print(f"Eliciting choices for {split_name} split ({len(split_data)} documents)")
        
        elicit_choices_for_split(
            chat_wrapper=chat_wrapper,
            split_data=split_data,
            split_name=split_name,
            temps=args.temps,
            num_trials=args.num_trials,
            run_name=args.args_name
        )
    
    print("Choice elicitation complete!")
