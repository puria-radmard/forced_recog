import pandas as pd
import os
import yaml
import torch
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Dict
import sys
import copy

from load_data import load_dataset
from model.load import load_model
from model.base import ChatTemplateWrapper

from dotenv import load_dotenv

from sft_utils.lora import download_and_apply_lora

from prompts.summary import (
    DETECTION_SYSTEM_PROMPT, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT, 
    DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION, DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY
)

from utils.elicit import get_choice_token_logits_from_token_ids
from utils.util import YamlConfig


document_columns = [
    'document_idx', 
    'summary1_run_name', 'summary1_lora_name',
    'summary2_run_name', 'summary2_lora_name',
    'prob_choice_1', 'prob_choice_2', 
]


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


def load_completion_status(results_file: str) -> Set[int]:
    """
    Load completion status from existing CSV file.
    
    Args:
        results_file: Path to the CSV file
        
    Returns:
        Set of completed document_idx values
    """
    if not os.path.exists(results_file):
        return set()
    
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        raise ValueError(f"Failed to read existing CSV {results_file}: {e}")
    
    # Validate CSV structure
    if list(df.columns) != document_columns:
        raise ValueError(
            f"Invalid columns in {results_file}. "
            f"Expected {document_columns}, got {list(df.columns)}"
        )
    
    # Since we do forward/backward for each doc, a doc is complete if it has 2 rows
    completed_docs = set()
    for doc_idx in df['document_idx'].unique():
        doc_rows = df[df['document_idx'] == doc_idx]
        if len(doc_rows) == 2:  # Both forward and backward completed
            completed_docs.add(doc_idx)
        elif len(doc_rows) > 2:
            raise Exception(f"More than 2 rows for document {df['document_idx']}")
    
    return completed_docs


def load_fj_summaries(run_name: str, lora_run_name: str, artifact_name: str, results_dir: Optional[str] = None) -> pd.DataFrame:
    """Load fj summaries from a specific LoRA run."""
    results_dir = results_dir or 'results_and_data/modal_results/results'
    summary_file = f"{results_dir}/main/{run_name}/test/sfted_summaries/{lora_run_name}/{artifact_name}/T0.0_trial0_stylenatural.csv"
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"False justification summaries not found: {summary_file}")
    
    df = pd.read_csv(summary_file)
    expected_cols = ['document_idx', 'summary']
    if list(df.columns) != expected_cols:
        raise ValueError(f"Invalid columns in {summary_file}. Expected {expected_cols}, got {list(df.columns)}")
    
    print(f"Loaded {len(df)} summaries from {summary_file}")
    return df


def elicit_fj_choices(
    chat_wrapper: ChatTemplateWrapper,
    test_data: pd.DataFrame,
    positive_summaries: pd.DataFrame,
    negative_summaries: pd.DataFrame,
    positive_run_name: str,
    positive_artifact_name: str,
    negative_run_name: str,
    negative_artifact_name: str,
    results_file: str,
    continue_mode: bool = False
) -> None:
    """
    Elicit pairwise choices between false justification-induced summaries.
    """
    
    # Load completion status if in continue mode
    completed_docs = set()
    if continue_mode:
        completed_docs = load_completion_status(results_file)
        if completed_docs:
            print(f"Continue mode: Found existing results for {len(completed_docs)} documents")
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
    
    # Create lookup dictionaries for summaries
    positive_summaries_dict = dict(zip(positive_summaries['document_idx'], positive_summaries['summary']))
    negative_summaries_dict = dict(zip(negative_summaries['document_idx'], negative_summaries['summary']))
    
    # Find common document indices
    common_docs = set(positive_summaries_dict.keys()) & set(negative_summaries_dict.keys())
    print(f"Found {len(common_docs)} documents with both positive and negative summaries")
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Eliciting fj choices"):
        document_idx = row['document_idx']
        article = row['article']
        
        # Skip if this document doesn't have both summaries
        if document_idx not in common_docs:
            continue
            
        # Skip if already completed in continue mode
        if continue_mode and document_idx in completed_docs:
            continue
        
        positive_summary = positive_summaries_dict[document_idx]
        negative_summary = negative_summaries_dict[document_idx]
        
        torch.cuda.empty_cache()
        
        try:
            # Forward comparison: positive vs negative
            forward_prompt_raw = (
                DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1=positive_summary, summary_2=negative_summary) 
                + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
            )
            forward_prompt_full = chat_wrapper.format_chat(
                system_prompt=DETECTION_SYSTEM_PROMPT,
                user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) + forward_prompt_raw,
                prefiller=""
            )        

            # Backward comparison: negative vs positive
            backward_prompt_raw = (
                DETECTION_PROMPT_TEMPLATE_VS_MODEL_BODY.format(summary_1=negative_summary, summary_2=positive_summary) 
                + DETECTION_PROMPT_TEMPLATE_VS_MODEL_QUESTION
            )
            backward_prompt_full = chat_wrapper.format_chat(
                system_prompt=DETECTION_SYSTEM_PROMPT,
                user_message=DETECTION_PROMPT_TEMPLATE_VS_MODEL_BASE_PROMPT.format(article=article) + backward_prompt_raw,
                prefiller=""
            )

            # Get probabilities
            forward_probs = get_choice_token_logits_from_token_ids(chat_wrapper.forward(chats=[forward_prompt_full]).logits, choice_tokens)
            backward_probs = get_choice_token_logits_from_token_ids(chat_wrapper.forward(chats=[backward_prompt_full]).logits, choice_tokens)
            
            # Store results
            new_results = pd.DataFrame([
                {
                    'document_idx': document_idx,
                    'summary1_run_name': positive_run_name,
                    'summary1_lora_name': positive_artifact_name, 
                    'summary2_run_name': negative_run_name,
                    'summary2_lora_name': negative_artifact_name,
                    'prob_choice_1': forward_probs[0,0].item(),
                    'prob_choice_2': forward_probs[0,1].item(),
                },
                {
                    'document_idx': document_idx,
                    'summary1_run_name': negative_run_name,
                    'summary1_lora_name': negative_artifact_name,
                    'summary2_run_name': positive_run_name, 
                    'summary2_lora_name': positive_artifact_name,
                    'prob_choice_1': backward_probs[0,0].item(),
                    'prob_choice_2': backward_probs[0,1].item(),
                }
            ])

            # Save results
            new_results.to_csv(results_file, mode='a', header=False, index=False)
            completed_docs.add(document_idx)
            
        except torch.OutOfMemoryError:
            print(f"OOM error for document {document_idx}, skipping...")
            continue


if __name__ == "__main__":

    load_dotenv()

    # Parse command line arguments
    continue_mode = len(sys.argv) > 1 and sys.argv[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(sys.argv) - (1 if continue_mode else 0)

    if effective_argc != 3:
        print("Usage:")
        print("  python -m scripts_summary.pairwise_selection.pairwise_hf_on_fj_pairs /path/to/config.yaml /path/to/lora/args/lora_args.yaml [continue]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    lora_args_path = sys.argv[2]
    
    if continue_mode:
        print("Continue mode: Will resume from existing results")
    else:
        print("Fresh run: Will create new result files")
    
    # Load main config
    args = YamlConfig(config_path)
    
    # Load lora args
    with open(lora_args_path, 'r') as f:
        lora_args = yaml.safe_load(f)
    
    required_keys = ['positive_run_name', 'positive_artifact_name', 'negative_run_name', 
                    'negative_artifact_name', 'judge_run_name', 'judge_artifact_name']
    for key in required_keys:
        if key not in lora_args:
            raise ValueError(f"Missing required key '{key}' in lora_args.yaml")
    
    print(f"LoRA Arguments:")
    print(f"  Positive: {lora_args['positive_run_name']}/{lora_args['positive_artifact_name']}")
    print(f"  Negative: {lora_args['negative_run_name']}/{lora_args['negative_artifact_name']}")
    print(f"  Judge: {lora_args['judge_run_name']}/{lora_args['judge_artifact_name']}")
    
    # Load model and judge LoRA adapters
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    print(f"Loading judge LoRA adapters...")
    chat_wrapper = download_and_apply_lora(chat_wrapper, lora_args['judge_run_name'], lora_args['judge_artifact_name'])
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    _, test_data, _ = load_dataset(args.dataset, splits=['test'])
    
    # Load fj summaries
    print("Loading fj summaries...")
    positive_summaries = load_fj_summaries(
        run_name=args.args_name,
        lora_run_name=lora_args['positive_run_name'],
        artifact_name=lora_args['positive_artifact_name']
    )
    negative_summaries = load_fj_summaries(
        run_name=args.args_name,
        lora_run_name=lora_args['negative_run_name'], 
        artifact_name=lora_args['negative_artifact_name']
    )
    
    print(f"Loaded {len(positive_summaries)} positive and {len(negative_summaries)} negative summaries")
    
    # Set up output file in same directory as lora_args
    output_dir = os.path.dirname(lora_args_path)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "fj_choice_results.csv")
    print(f"Saving results to: {results_file}")
    
    # Elicit choices
    elicit_fj_choices(
        chat_wrapper=chat_wrapper,
        test_data=test_data,
        positive_summaries=positive_summaries,
        negative_summaries=negative_summaries,
        positive_run_name=lora_args['positive_run_name'],
        positive_artifact_name=lora_args['positive_artifact_name'],
        negative_run_name=lora_args['negative_run_name'],
        negative_artifact_name=lora_args['negative_artifact_name'],
        results_file=results_file,
        continue_mode=continue_mode
    )
    
    print("Contrastive choice elicitation complete!")