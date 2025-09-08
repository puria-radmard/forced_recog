import copy
import pandas as pd
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Set
import sys

from model.load import load_model
from model.base import ChatTemplateWrapper

from sft_utils.lora import download_and_apply_lora

from prompts import INTROSPECTION_SYSTEM_PROMPT, INTROSPECTION_QUESTION, INTROSPECTION_FOLLOWUP_QUESTION

from utils.elicit import get_choice_token_logits_from_token_ids
from utils.util import YamlConfig

from scripts.pairwise_selection.pairwise_hf import get_choice_tokens


question_columns = [
    'question_idx', 
    'reversed',
    'prob_choice_1', 
    'prob_choice_2'
]


def load_completion_status(results_file: str) -> Set[int]:
    """
    Load completion status from existing CSV file.
    
    Args:
        results_file: Path to the CSV file
        
    Returns:
        Set of completed question indices
    """
    if not os.path.exists(results_file):
        return set()
    
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        raise ValueError(f"Failed to read existing CSV {results_file}: {e}")
    
    # Validate CSV structure
    if list(df.columns) != question_columns:
        raise ValueError(
            f"Invalid columns in {results_file}. "
            f"Expected {question_columns}, got {list(df.columns)}"
        )
    
    return set(df['question_idx'].tolist())


def load_questions_data(questions_dataset_name: str) -> pd.DataFrame:
    """Load questions from CSV file."""
    questions_file = f"results_and_data/questions_data/{questions_dataset_name}.csv"
    
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    df = pd.read_csv(questions_file)
    
    # Validate expected columns
    expected_cols = ['question_idx', 'question_body', 'option_1', 'option_2']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Questions CSV must have columns: {expected_cols}. Got: {list(df.columns)}")
    
    print(f"Loaded {len(df)} questions from {questions_file}")
    return df


def elicit_introspective_choices(
    chat_wrapper: ChatTemplateWrapper,
    questions_dataset_name: str,
    questions_df: pd.DataFrame,
    args_name: str,
    use_lora: bool = False,
    lora_run_name: str = None,
    artifact_name: str = None,
    results_dir: Optional[str] = None,
    continue_mode: bool = False
) -> None:
    """
    Elicit introspective choices for questions.
    
    Args:
        chat_wrapper: Loaded model wrapper
        questions_df: DataFrame with columns [question_idx, question_body, option_1, option_2]
        args_name: Name of the run
        use_lora: Whether to use LoRA adapters
        lora_run_name: WandB run name for LoRA adapters
        artifact_name: Artifact name for LoRA adapters
        continue_mode: Whether to continue from existing results
    """
    # Determine output directory based on whether using LoRA
    results_dir = results_dir or 'results_and_data/results'
    if use_lora:
        output_dir = f"{results_dir}/main/{args_name}/backward_sft_introspective_choices/{lora_run_name}/{artifact_name}"
        results_file = f"{output_dir}/{questions_dataset_name}.csv"
        print(f"Using LoRA model - saving to: {results_file}")
    else:
        output_dir = f"{results_dir}/main/{args_name}/initial_introspective_choices"
        results_file = f"{output_dir}/{questions_dataset_name}.csv"
        print(f"Using base model - saving to: {results_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load completion status if in continue mode
    completed_questions = set()
    if continue_mode:
        completed_questions = load_completion_status(results_file)
        if completed_questions:
            print(f"Continue mode: Found existing results for {len(completed_questions)} questions")
        else:
            print("Continue mode: No existing results found, starting fresh")
            header_df = pd.DataFrame(columns=question_columns)
            header_df.to_csv(results_file, index=False)
    else:
        # Create fresh CSV file
        header_df = pd.DataFrame(columns=question_columns)
        header_df.to_csv(results_file, index=False)
        print("Fresh run: Created new results file")

    # Get choice tokens
    choice_tokens = get_choice_tokens(chat_wrapper)

    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Eliciting introspective choices"):
        
        question_idx = row['question_idx']
        question_body = row['question_body']
        option_1 = row['option_1']
        option_2 = row['option_2']

        # Skip if already completed
        if continue_mode and question_idx in completed_questions:
            continue

        user_message_forwards = f"{question_body}?\n\nOption 1: {option_1}.\n\nOption 2: {option_2}. {INTROSPECTION_QUESTION}"
        user_message_reversed = f"{question_body}?\n\nOption 1: {option_2}.\n\nOption 2: {option_1}. {INTROSPECTION_QUESTION}"
        
        forwards_prompt = chat_wrapper.format_chat(
            system_prompt=INTROSPECTION_SYSTEM_PROMPT,
            user_message=user_message_forwards,
            prefiller=""
        )
        backwards_prompt = chat_wrapper.format_chat(
            system_prompt=INTROSPECTION_SYSTEM_PROMPT,
            user_message=user_message_reversed,
            prefiller=""
        )

        try:
            forwards_choice_output, forwards_prompt_ids = chat_wrapper.forward(chats=[forwards_prompt], return_input_ids=True)
            backwards_choice_output, backwards_prompt_ids = chat_wrapper.forward(chats=[backwards_prompt], return_input_ids=True)

            # Get model response logits
            forward_probs = get_choice_token_logits_from_token_ids(forwards_choice_output.logits, choice_tokens)
            reversed_probs = get_choice_token_logits_from_token_ids(backwards_choice_output.logits, choice_tokens)
            
        except torch.OutOfMemoryError:
            print(f"OOM on question {question_idx}, skipping...")
            continue

        torch.cuda.empty_cache()

        # Store results
        new_result = pd.DataFrame([
            {
                'question_idx': question_idx,
                'reversed': False,
                'prob_choice_1': forward_probs[0,0].item(),
                'prob_choice_2': forward_probs[0,1].item(),
            },
            {
                'question_idx': question_idx,
                'reversed': True,
                'prob_choice_1': reversed_probs[0,1].item(),
                'prob_choice_2': reversed_probs[0,0].item(),
            },
        ])

        # Save result
        new_result.to_csv(results_file, mode='a', header=False, index=False)


if __name__ == "__main__":

    # Parse command line arguments
    continue_mode = len(sys.argv) > 1 and sys.argv[-1] == "continue"
    
    # Determine effective argument count (excluding 'continue' if present)
    effective_argc = len(sys.argv) - (1 if continue_mode else 0)

    if effective_argc not in [3, 5]:
        print("Usage:")
        print("  Base model: python introspective_choices.py /path/to/yaml/args.yaml <questions_dataset_name> [continue]")
        print("  With LoRA:  python introspective_choices.py /path/to/yaml/args.yaml <questions_dataset_name> <wandb_run_name> <artifact_name> [continue]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    questions_dataset_name = sys.argv[2]
    use_lora = effective_argc == 5
    
    if use_lora:
        wandb_run_name = sys.argv[3]
        artifact_name = sys.argv[4]
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
    
    # Load questions
    print(f"Loading questions dataset: {questions_dataset_name}")
    questions_df = load_questions_data(questions_dataset_name)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    chat_wrapper = load_model(args.model_name, device='auto')
    
    # Apply LoRA adapters if requested
    if use_lora:
        chat_wrapper = download_and_apply_lora(chat_wrapper, wandb_run_name, artifact_name)
    
    print(f"Eliciting choices for {len(questions_df)} introspective questions")
    
    elicit_introspective_choices(
        chat_wrapper=chat_wrapper,
        questions_dataset_name=questions_dataset_name,
        questions_df=questions_df,
        args_name=args.args_name,
        use_lora=use_lora,
        lora_run_name=wandb_run_name,
        artifact_name=artifact_name,
        continue_mode=continue_mode
    )
    
    print("Introspective choice elicitation complete!")
