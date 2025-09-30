"""
Assist Tag Recognition Test Script

DUAL MODE OPERATION:
1. IDE MODE (default): Run without arguments for easy debugging
   - Uses hardcoded config path: configs/operationalizations/AT_2T/rec_config.yaml
   - Easy to modify and debug in IDE
   - All parameters configured through YAML file

2. CLI MODE: Run with arguments for production use
   - Specify custom config file with --config argument
   - Supports different configurations for different experiments
   - Model selection: ["all"] or specific model names

USAGE:
- IDE mode: python run_experiment.py
- CLI mode: python run_experiment.py --config path/to/config.yaml
- Show models: python run_experiment.py --show-models --config path/to/config.yaml

EXPERIMENT TYPES:
- AT_2T: Tests model self-recognition (which response was originally produced by the model)
- UT_2T: Tests user preference (which response the model prefers)

CONFIGURATION:
- Model name selection (auto-detected model type from name)
- Experiment directory and data settings
- Conversation limits and testing parameters
- Display options and output settings
- Experiment type (AT_2T vs UT_2T)

FEATURES:
- YAML-based configuration for easy parameter management
- Support for Anthropic, Google, OpenAI, and HuggingFace models
- Granular model selection by specific model names
- Configurable data processing and conversation limits
- Detailed progress reporting and results summary
- Flexible testing and production modes through configuration
- Dual-mode operation for both IDE debugging and CLI production use
- Multiple experiment protocols (AT_2T vs UT_2T)
"""
import os
import pandas as pd
import yaml
import torch
import argparse
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Dict
import copy
from dotenv import load_dotenv
from util.conversation_logger import conversation_logger

import windows_pathing_fix
windows_pathing_fix.fix_pathing()
from model.load import load_model
from model.base import ChatTemplateWrapper
from model.anthropic import load_anthropic_model, AnthropicWrapper
from model.gemini import load_gemini_model, GeminiWrapper
from model.openai import load_openai_model, OpenAIWrapper

from sft_utils.lora import download_and_apply_lora

# Prompts are now loaded from YAML config files

from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig

# Load environment variables from .env file
load_dotenv()



document_columns = [
    'row_a_idx', 
    'row_b_idx', 'model_a', 'treatment_a', 'model_b', 'treatment_b',
    'prob_choice_1', 'prob_choice_2', 
]


def generate_conversations(data_df: pd.DataFrame) -> List[Dict]:
    """
    Generate conversations by pairing control and treatment responses for the same trial.
    Each trial produces conversations where control is the correct answer and treatment is incorrect.
    
    Args:
        data_df: DataFrame with columns [trial, model, treatment, passage, response]
        
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    
    # Group by trial to ensure we only compare within the same trial
    for trial_num, trial_group in data_df.groupby('trial'):
        trial_rows = trial_group.reset_index(drop=True)
        
        # Separate control and treatment rows
        control_rows = trial_rows[trial_rows['treatment'] == 'control']
        treatment_rows = trial_rows[trial_rows['treatment'].isin(['typo', 'other_model', 'capitalization'])]
        
        # For each control row, create conversations with all treatment rows
        for control_idx, control_row in control_rows.iterrows():
            control_model = control_row['model']
            control_passage = control_row['passage']
            control_response = control_row['response']
            
            # Pair with every treatment row in the same trial
            for treatment_idx, treatment_row in treatment_rows.iterrows():
                treatment_model = treatment_row['model']
                treatment_response = treatment_row['response']
                treatment_type = treatment_row['treatment']  # Get actual treatment type
                
                # Create two conversations for this pair
                # Conversation 1: control response first, treatment response second
                conv1 = {
                    'conversation_id': f"trial{trial_num}_{control_model}_control_vs_{treatment_model}_treatment_1",
                    'trial': trial_num,
                    'control_row_idx': control_idx,
                    'treatment_row_idx': treatment_idx,
                    'model_base': control_model,  # The model doing the evaluation
                    'model_other': treatment_model,  # The other model being compared
                    'treatment_control': 'control',
                    'treatment_other': treatment_type,  # Use actual treatment type
                    'passage': control_passage,
                    'response_1': control_response,  # Control response (correct)
                    'response_2': treatment_response,  # Treatment response (incorrect)
                    'response_1_source': 'control',
                    'response_2_source': 'treatment'
                }
                
                # Conversation 2: treatment response first, control response second
                conv2 = {
                    'conversation_id': f"trial{trial_num}_{control_model}_control_vs_{treatment_model}_treatment_2", 
                    'trial': trial_num,
                    'control_row_idx': control_idx,
                    'treatment_row_idx': treatment_idx,
                    'model_base': control_model,  # The model doing the evaluation
                    'model_other': treatment_model,  # The other model being compared
                    'treatment_control': 'control',
                    'treatment_other': treatment_type,  # Use actual treatment type
                    'passage': control_passage,
                    'response_1': treatment_response,  # Treatment response (incorrect)
                    'response_2': control_response,  # Control response (correct)
                    'response_1_source': 'treatment',
                    'response_2_source': 'control'
                }
                
                conversations.extend([conv1, conv2])
    
    return conversations


def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate text to a maximum number of words.
    
    Args:
        text: The text to truncate
        max_words: Maximum number of words (None for no truncation)
        
    Returns:
        Truncated text or original text if max_words is None
    """
    if max_words is None or pd.isna(text):
        return text
    return ' '.join(str(text).split()[:max_words])


def get_choice_tokens(chat_wrapper) -> List[List[int]]:
    """Get token IDs for choice responses. Handle multi-token choices by using first token."""
    choice_strings = [["1"], ["2"]]
    choice_tokens = []
    
    for option_str_list in choice_strings:
        option_tokens = []
        for option_str in option_str_list:
            token_ids = chat_wrapper.tokenizer.encode(option_str, add_special_tokens=False)
            if len(token_ids) != 1:
                print(f"Warning: Choice token '{option_str}' produces {len(token_ids)} tokens: {token_ids}. Using first token.")
                option_tokens.append(token_ids[0])  # Use first token
            else:
                option_tokens.extend(token_ids)
        choice_tokens.append(option_tokens)
    
    return choice_tokens


def get_results_file_path(data_file: str) -> str:
    """
    Get the expected results file path based on the data file path.
    
    Args:
        data_file: Path to the experiment directory
        
    Returns:
        Path to the expected results CSV file
    """
    # Convert experiment directory path to results directory structure
    exp_path_parts = data_file.replace('\\', '/').split('/')
    
    # Find the index of 'experiments' in the path
    exp_idx = None
    for i, part in enumerate(exp_path_parts):
        if part == 'experiments':
            exp_idx = i
            break
    
    if exp_idx is not None and exp_idx + 1 < len(exp_path_parts):
        # Extract the dataset name (e.g., "WikiSum")
        dataset_name = exp_path_parts[exp_idx + 1]
        # Create results directory: results_and_data/results/{dataset_name}/
        results_dir = f"results_and_data/results/{dataset_name}"
    else:
        # Fallback to a default directory
        results_dir = "results_and_data/results/default_dir"
    
    # Create filename based on experiment directory
    experiment_name = os.path.basename(data_file)
    results_file = os.path.join(results_dir, f"{experiment_name}_choice_results.csv").replace('\\', '/')
    
    return results_file


def load_existing_results(results_file: str) -> pd.DataFrame:
    """
    Load existing results from CSV file if it exists.
    
    Args:
        results_file: Path to the results CSV file
        
    Returns:
        DataFrame with existing results, or empty DataFrame if file doesn't exist
    """
    if not os.path.exists(results_file):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(results_file)
        print(f"📁 Found existing results: {results_file}")
        print(f"   Existing conversations: {len(df)}")
        return df
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing results from {results_file}: {e}")
        return pd.DataFrame()


def identify_false_data_conversations(existing_df: pd.DataFrame) -> Set[str]:
    """
    Identify conversation IDs that contain false data (identical probabilities).
    
    Args:
        existing_df: DataFrame with existing results
        
    Returns:
        Set of conversation IDs with false data
    """
    if existing_df.empty or 'prob_choice_1' not in existing_df.columns or 'prob_choice_2' not in existing_df.columns:
        return set()
    
    # Find conversations with identical probabilities (indicating neutral logits from rate limit errors)
    identical_probs = existing_df[existing_df['prob_choice_1'] == existing_df['prob_choice_2']]
    
    if len(identical_probs) > 0:
        print(f"🔍 Found {len(identical_probs)} conversations with false data (identical probabilities)")
        if 'conversation_id' in identical_probs.columns:
            false_conversation_ids = set(identical_probs['conversation_id'].tolist())
            print(f"   False conversation IDs: {sorted(false_conversation_ids)}")
            return false_conversation_ids
    
    return set()


def filter_conversations_to_rerun(conversations: List[Dict], false_data_ids: Set[str], existing_df: pd.DataFrame) -> List[Dict]:
    """
    Filter conversations to only include those that need to be re-run.
    
    Args:
        conversations: List of all conversations
        false_data_ids: Set of conversation IDs with false data
        existing_df: DataFrame with existing results
        
    Returns:
        List of conversations that need to be re-run
    """
    if existing_df.empty:
        print("📝 No existing results found - will process all conversations")
        return conversations
    
    # Get existing conversation IDs
    existing_ids = set(existing_df['conversation_id'].tolist()) if 'conversation_id' in existing_df.columns else set()
    
    # Find conversations that need re-running
    conversations_to_rerun = []
    for conv in conversations:
        conv_id = conv['conversation_id']
        
        # Re-run if: 1) not in existing results, or 2) has false data
        if conv_id not in existing_ids or conv_id in false_data_ids:
            conversations_to_rerun.append(conv)
    
    print(f"📊 Conversation filtering:")
    print(f"   Total conversations: {len(conversations)}")
    print(f"   Existing conversations: {len(existing_ids)}")
    print(f"   False data conversations: {len(false_data_ids)}")
    print(f"   Conversations to re-run: {len(conversations_to_rerun)}")
    
    return conversations_to_rerun


def merge_results(existing_df: pd.DataFrame, new_results: List[Dict]) -> pd.DataFrame:
    """
    Merge new results with existing results, replacing false data.
    
    Args:
        existing_df: DataFrame with existing results
        new_results: List of new result dictionaries
        
    Returns:
        Merged DataFrame with updated results
    """
    if existing_df.empty:
        return pd.DataFrame(new_results)
    
    if not new_results:
        return existing_df
    
    # Convert new results to DataFrame
    new_df = pd.DataFrame(new_results)
    
    # Get conversation IDs that were updated
    updated_ids = set(new_df['conversation_id'].tolist())
    
    # Remove old versions of updated conversations from existing data
    existing_clean = existing_df[~existing_df['conversation_id'].isin(updated_ids)]
    
    # Combine clean existing data with new data
    merged_df = pd.concat([existing_clean, new_df], ignore_index=True)
    
    print(f"🔄 Merged results:")
    print(f"   Existing conversations: {len(existing_df)}")
    print(f"   New/updated conversations: {len(new_results)}")
    print(f"   Final total: {len(merged_df)}")
    
    return merged_df


def process_conversations_for_choices(
    conversations: List[Dict],
    data_file: str,
    max_conversations: int = 4,
    selected_models: List[str] = None,
    truncate_words: int = None,
    system_prompt: str = None,
    user_prompt_template: str = None,
    detection_prompt_template: str = None,
    experiment_type: str = "AT_2T",
    logger = None
) -> None:
    """
    Process conversations for pairwise choice elicitation using multiple models.
    Automatically detects and re-runs only conversations with false data.
    
    Args:
        conversations: List of conversation dictionaries
        data_file: Path to the data file (used to determine results directory)
        max_conversations: Maximum number of conversations to process per model
        selected_models: List of specific model names to process (e.g., ["anthropic_claude-3-5-sonnet-20241022"]) or ["all"]
        truncate_words: Maximum number of words for text truncation (None for no truncation)
        system_prompt: System prompt template
        user_prompt_template: User prompt template
        detection_prompt_template: Detection prompt template
        experiment_type: Type of experiment ("AT_2T" or "UT_2T")
    """
    print(f"Processing conversations with max {max_conversations} per model")
    
    # ===== CHECK FOR EXISTING RESULTS =====
    results_file = get_results_file_path(data_file)
    existing_df = load_existing_results(results_file)
    false_data_ids = identify_false_data_conversations(existing_df)
    
    # ===== FILTER CONVERSATIONS TO RE-RUN =====
    conversations_to_rerun = filter_conversations_to_rerun(conversations, false_data_ids, existing_df)
    
    if not conversations_to_rerun:
        print("✅ All conversations already have valid results - nothing to re-run!")
        return
    
    # Group conversations by base model (the model doing the evaluation)
    conversations_by_model = {}
    for conv in conversations_to_rerun:
        base_model = conv['model_base']
        if base_model not in conversations_by_model:
            conversations_by_model[base_model] = []
        conversations_by_model[base_model].append(conv)
    
    # Filter models based on selection
    if selected_models and "all" not in selected_models:
        filtered_conversations = {}
        for base_model, model_conversations in conversations_by_model.items():
            # Check if this specific model is in the selected models list
            if base_model in selected_models:
                filtered_conversations[base_model] = model_conversations
        
        conversations_by_model = filtered_conversations
        print(f"🎯 Filtered to selected models: {selected_models}")
        
        # Show which models were found vs requested
        found_models = list(conversations_by_model.keys())
        missing_models = [model for model in selected_models if model not in found_models and model != "all"]
        if missing_models:
            print(f"⚠️  Warning: Requested models not found in data: {missing_models}")
        if found_models:
            print(f"✅ Found models: {found_models}")
    
    print(f"Found {len(conversations_by_model)} unique base models: {list(conversations_by_model.keys())}")
    
    all_results = []
    
    # Process each base model separately
    for base_model, model_conversations in conversations_by_model.items():
        print(f"\n=== Processing base model: {base_model} ===")
        
        # Load the appropriate model wrapper
        if base_model.startswith('anthropic_'):
            model_name = base_model.replace('anthropic_', '')
            chat_wrapper = load_anthropic_model(model_name)
        elif base_model.startswith('google_'):
            model_name = base_model.replace('google_', '')
            chat_wrapper = load_gemini_model(model_name)
        elif base_model.startswith('gpt-'):
            model_name = base_model
            chat_wrapper = load_openai_model(model_name)
        else:
            # Assume HuggingFace model
            chat_wrapper = load_model(base_model, device='auto')
        
        # Get choice tokens
        choice_tokens = get_choice_tokens(chat_wrapper)
        print(f"Choice tokens: {choice_tokens}")
        
        # Limit conversations based on max_conversations setting
        limited_conversations = model_conversations[:max_conversations]
        
        for conv_idx, conv in enumerate(tqdm(limited_conversations, desc=f"Processing {base_model}")):
            # Apply truncation if configured
            passage = truncate_text(conv['passage'], truncate_words)
            response_1 = truncate_text(conv['response_1'], truncate_words)
            response_2 = truncate_text(conv['response_2'], truncate_words)
            
            # Create the user prompt using the (possibly truncated) passage
            user_prompt = user_prompt_template.format(passage=passage)
            
            if experiment_type == "AT_2T":
                conversation_full = chat_wrapper.format_chat(
                            system_prompt=system_prompt,
                            in_context_questions=[user_prompt, user_prompt],  # Both questions use the same prompt
                            in_context_answers=[response_1, response_2],  # Use truncated responses
                            user_message=detection_prompt_template,  # Use truncated passage
                        )
            elif experiment_type == "UT_2T":
                full_user_prompt = user_prompt_template.format(passage=passage)
                full_detection_prompt = detection_prompt_template.format(user_message=full_user_prompt, response_1=response_1, response_2=response_2)
                conversation_full = chat_wrapper.format_chat(
                    system_prompt=system_prompt,
                    user_message=full_detection_prompt,
                )
            elif experiment_type == "AT_IR":
                if conv["response_1_source"] == "control":
                    original_text_token = "1"
                    injected_text_token = "2"
                elif conv["response_1_source"] == "treatment":
                    original_text_token = "2"
                    injected_text_token = "1"
                else:
                    raise ValueError(f"Invalid response_1_source: {conv['response_1_source']}")
                conversation_full = chat_wrapper.format_chat(
                            system_prompt=system_prompt,
                            in_context_questions=[user_prompt], 
                            in_context_answers=[response_1], 
                            user_message=detection_prompt_template.format(injected_text_token=injected_text_token, original_text_token=original_text_token), 
                        )
            else:
                raise ValueError(f"Invalid experiment type: {experiment_type}")
            
            try:
                # Start logging this conversation if logger is available
                conversation_id = f"{model_name}_{conv_idx}"
                if logger:
                    logger.start_conversation(
                        conversation_id=conversation_id,
                        input_data={
                            "conversation_index": conv_idx,
                            "model_name": model_name,
                            "trial": conv.get('trial', 'unknown'),
                            "treatment": conv.get('treatment', 'unknown'),
                            "response_1_source": conv.get('response_1_source', 'unknown'),
                            "response_2_source": conv.get('response_2_source', 'unknown')
                        },
                        system_prompt=system_prompt,
                        user_prompt=conversation_full
                    )
                
                # Get model predictions using logging wrapper
                if logger:
                    outputs = chat_wrapper.forward_with_logging(
                        chats=[conversation_full],
                        logger=logger,
                        conversation_id=conversation_id
                    )
                else:
                    outputs = chat_wrapper.forward(chats=[conversation_full])
                
                conv_probs = get_choice_token_logits_from_token_ids(
                    outputs.logits, 
                    choice_tokens
                )
                
                # Display results
                prob_1 = conv_probs[0,0].item()
                prob_2 = conv_probs[0,1].item()
                print(f"Choice probabilities: {prob_1:.3f} vs {prob_2:.3f}")
                
                # Determine which choice was selected
                selected_choice = "1" if prob_1 > prob_2 else "2"
                print(f"  → Selected choice: {selected_choice} (response from {conv['response_1_source'] if selected_choice == '1' else conv['response_2_source']})")
                
                # Determine if the model selected the control response (correct choice)
                # Correct choice is when the model selects the control response
                # If response_1_source is 'control', then choice 1 is correct
                # If response_2_source is 'control', then choice 2 is correct
                correct_choice = "1" if conv['response_1_source'] == 'control' else "2"
                is_correct = selected_choice == correct_choice
                
                print(f"  → Correct choice: {correct_choice} (control response)")
                print(f"  → Model {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                
                # Log results if logger is available
                if logger:
                    logger.log_results({
                        "choice_probabilities": [prob_1, prob_2],
                        "selected_choice": selected_choice,
                        "correct_choice": correct_choice,
                        "is_correct": is_correct,
                        "accuracy": 1.0 if is_correct else 0.0
                    })
                    
                    logger.finish_conversation(success=True)
                
                # Store results
                all_results.append({
                    'conversation_id': conv['conversation_id'],
                    'trial': conv['trial'],
                    'control_row_idx': conv['control_row_idx'],
                    'treatment_row_idx': conv['treatment_row_idx'],
                    'model_base': conv['model_base'],
                    'model_other': conv['model_other'],
                    'treatment_control': conv['treatment_control'],
                    'treatment_other': conv['treatment_other'],
                    'response_1_source': conv['response_1_source'],
                    'response_2_source': conv['response_2_source'],
                    'prob_choice_1': prob_1,
                    'prob_choice_2': prob_2,
                    'selected_choice': selected_choice,
                    'correct_choice': correct_choice,
                    'is_correct': is_correct,
                })
                
            except torch.OutOfMemoryError:
                print("Out of memory - skipping this conversation")
                continue
            except Exception as e:
                print(f"Error processing conversation: {e}")
                
                # Log error if logger is available
                if logger and 'conversation_id' in locals():
                    logger.finish_conversation(success=False, error_message=str(e))
                
                continue
            
            # Clear cache to prevent memory issues
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
    
    # Save results
    if all_results:
        # Merge new results with existing results
        merged_df = merge_results(existing_df, all_results)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.dirname(results_file)
        os.makedirs(results_dir.replace('\\', '/'), exist_ok=True)
        
        # Save merged results
        merged_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        print(f"Processed {len(all_results)} new conversations")
        print(f"Total conversations in file: {len(merged_df)}")
        
        # Results summary
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"  New conversations processed: {len(all_results)}")
        print(f"  Total conversations in file: {len(merged_df)}")
        print(f"  Unique base models: {merged_df['model_base'].nunique()}")
        print(f"  Response order distribution:")
        order_counts = merged_df['response_1_source'].value_counts()
        for order, count in order_counts.items():
            print(f"    {order} -> other: {count}")
        
        print(f"\n  Choice selection summary:")
        choice_counts = merged_df['selected_choice'].value_counts()
        for choice, count in choice_counts.items():
            print(f"    Choice {choice}: {count} times")
        
        print(f"\n  Accuracy summary:")
        accuracy = merged_df['is_correct'].mean()
        correct_count = merged_df['is_correct'].sum()
        total_count = len(merged_df)
        print(f"    Overall accuracy: {accuracy:.3f} ({correct_count}/{total_count})")
        
        # Accuracy by model
        print(f"    Accuracy by model:")
        model_accuracy = merged_df.groupby('model_base')['is_correct'].agg(['mean', 'count']).round(3)
        for model, row in model_accuracy.iterrows():
            print(f"      {model}: {row['mean']:.3f} ({int(row['count'] * row['mean'])}/{int(row['count'])})")
        
        print(f"\n  Sample results:")
        for _, row in merged_df.head(4).iterrows():
            correct_mark = "✓" if row['is_correct'] else "✗"
            print(f"    {correct_mark} {row['conversation_id']}: {row['model_base']} chose {row['selected_choice']} (prob: {row['prob_choice_1']:.3f} vs {row['prob_choice_2']:.3f})")
    else:
        print("No new results to save")


def load_data(experiment_dir: str) -> pd.DataFrame:
    """
    Load assist tag data from control and treatment CSV files.
    
    Args:
        experiment_dir: Path to the experiment directory containing control.csv and treatment.csv
        
    Returns:
        Combined DataFrame with columns: trial, model, treatment, passage, response
    """
    control_file = os.path.join(experiment_dir, "control.csv")
    treatment_file = os.path.join(experiment_dir, "treatment.csv")
    
    if not os.path.exists(control_file):
        raise FileNotFoundError(f"Control data file not found: {control_file}")
    if not os.path.exists(treatment_file):
        raise FileNotFoundError(f"Treatment data file not found: {treatment_file}")
    
    try:
        control_df = pd.read_csv(control_file)
        treatment_df = pd.read_csv(treatment_file)
    except Exception as e:
        raise ValueError(f"Failed to read assist tag data from {experiment_dir}: {e}")
    
    # Validate required columns
    required_columns = ['trial', 'model', 'treatment', 'passage', 'response']
    for df, file_type in [(control_df, "control"), (treatment_df, "treatment")]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {file_type} file: {missing_columns}")
    
    # Combine the dataframes
    combined_df = pd.concat([control_df, treatment_df], ignore_index=True)
    
    return combined_df


def load_completion_status(results_file: str) -> Dict[Tuple[int, int], bool]:
    """
    Load completion status from existing CSV file.
    
    Args:
        results_file: Path to the CSV file
        
    Returns:
        Dictionary mapping (row_a_idx, row_b_idx) pairs to completion status
    """
    if not os.path.exists(results_file):
        return {}
    
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
    
    completed = {}
    for _, row in df.iterrows():
        pair_key = (row['row_a_idx'], row['row_b_idx'])
        completed[pair_key] = True
    
    return completed



def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Assist Tag Recognition Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config file
  python run_experiment.py
  
  # Run with custom config file
  python run_experiment.py --config configs/custom_config.yaml
  
  # Run with specific experiment directory
  python run_experiment.py --experiment-dir results_and_data/experiments/WikiSum/model_vs_all_others
  
  # Show available models in experiment
  python run_experiment.py --show-models --config configs/operationalizations/AT_2T/rec_config.yaml
        """
    )
    
    parser.add_argument("--config", 
                       default="configs/operationalizations/AT_2T/rec_config.yaml",
                       help="Path to the YAML configuration file (default: configs/operationalizations/AT_2T/rec_config.yaml)")
    parser.add_argument("--experiment-dir",
                       help="Path to the experiment directory (overrides config file setting)")
    parser.add_argument("--show-models", action="store_true",
                       help="Show available models in the experiment directory and exit")
    
    return parser.parse_args()


def show_available_models(experiment_dir: str) -> None:
    """
    Show available models in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory containing control.csv and treatment.csv
    """
    if not os.path.exists(experiment_dir):
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    try:
        df = load_data(experiment_dir)
        print(f"\n📊 Available models in {experiment_dir}:")
        print("=" * 60)
        
        if 'model' in df.columns:
            model_counts = df['model'].value_counts()
            print(f"Total rows: {len(df)}")
            print(f"Unique models: {len(model_counts)}")
            print("\nModels and row counts:")
            for model, count in model_counts.items():
                print(f"  {model}: {count} rows")
            
            # Show treatment breakdown
            print("\nTreatment breakdown:")
            treatment_counts = df['treatment'].value_counts()
            for treatment, count in treatment_counts.items():
                print(f"  {treatment}: {count} rows")
            
            # Show model types
            print("\nModel types:")
            model_types = {}
            for model in df['model'].unique():
                if model.startswith('anthropic_'):
                    model_types.setdefault('anthropic', []).append(model)
                elif model.startswith('google_'):
                    model_types.setdefault('google', []).append(model)
                elif model.startswith('gpt-'):
                    model_types.setdefault('openai', []).append(model)
                else:
                    model_types.setdefault('huggingface', []).append(model)
            
            for model_type, models in model_types.items():
                print(f"  {model_type}: {models}")
        else:
            print("❌ No 'model' column found in data")
            print(f"Available columns: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"❌ Error reading experiment directory: {e}")


def infer_model_type(model_name: str) -> str:
    """
    Automatically infer model type from model name.
    
    Args:
        model_name: The model name (e.g., "claude-3-5-sonnet-20241022", "gemini-1.5-flash")
        
    Returns:
        The inferred model type ("anthropic", "google", "openai", "huggingface")
        
    Raises:
        ValueError: If the model name doesn't match any known pattern
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower.startswith('claude-'):
        return "anthropic"
    elif model_name_lower.startswith('gemini-'):
        return "google"
    elif model_name_lower.startswith('gpt-'):
        return "openai"
    elif '/' in model_name or model_name_lower.startswith(('microsoft/', 'meta/', 'huggingface/', 'microsoft-', 'meta-', 'huggingface-')):
        # Explicit HuggingFace model patterns
        return "huggingface"
    else:
        # Unknown model type - raise error to prevent silent bugs
        raise ValueError(
            f"Unknown model type for '{model_name}'. "
            f"Supported patterns: claude-*, gemini-*, gpt-*, or HuggingFace models (containing '/' or starting with microsoft/, meta/, huggingface/). "
            f"Please check the model name or add support for this model type."
        )


def load_config(config_path: str = "configs/operationalizations/AT_2T/rec_config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required config parameters are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    # Required configuration parameters
    required_params = [
        "experiment_dir",
        "max_conversations", 
        "selected_models",
        "truncate_words",
        "experiment_type"
    ]
    
    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(
            f"Missing required configuration parameters in {config_path}: {missing_params}. "
            f"Please add these parameters to your config file."
        )
    
    # Required prompts section
    if "prompts" not in config:
        raise ValueError(
            f"Missing required 'prompts' section in {config_path}. "
            f"Please add a 'prompts' section with 'system', 'user', and 'detection' templates."
        )
    
    required_prompts = ["system", "user", "detection"]
    missing_prompts = [prompt for prompt in required_prompts if prompt not in config["prompts"]]
    if missing_prompts:
        raise ValueError(
            f"Missing required prompt templates in {config_path}: {missing_prompts}. "
            f"Please add these prompt templates to the 'prompts' section of your config file."
        )
    
    # Optional parameters with validation
    optional_params = {
        "truncate_articles": True,
        "show_sample_data": True,
        "show_conversation_breakdown": True
    }
    
    for key, default_value in optional_params.items():
        if key not in config:
            config[key] = default_value
    
    return config


def main():
    """
    Main function - supports both CLI and IDE usage modes
    """
    print("=== Test Script ===")
    print("This script tests model self-recognition on various experimental paradigms")
    
    # ===== DETERMINE CONFIG PATH =====
    # Check if running from CLI (has command line arguments) or IDE (no arguments)
    import sys
    args = None
    if len(sys.argv) > 1:
        # CLI mode: parse arguments
        args = parse_arguments()
        config_path = args.config
        
        # Handle show-models option
        if args.show_models:
            config = load_config(config_path)
            # Use experiment-dir argument if provided, otherwise use config file setting
            if hasattr(args, 'experiment_dir') and args.experiment_dir:
                experiment_dir = args.experiment_dir
            else:
                experiment_dir = config.get("experiment_dir", "results_and_data/experiments/WikiSum/anthropic_claude-sonnet-4-20250514_vs_all_others_control_comparison")
            show_available_models(experiment_dir)
            return
    else:
        # IDE mode: use hardcoded config path
        config_path = "configs/operationalizations/AT_IR/rec_config.yaml"
    
    # ===== LOAD CONFIGURATION =====
    config = load_config(config_path)
    
    # Extract configuration parameters
    # Note: model_name and model_type are determined dynamically from the data, not from config
    
    # Use experiment-dir argument if provided, otherwise use config file setting
    if args and hasattr(args, 'experiment_dir') and args.experiment_dir:
        experiment_dir = args.experiment_dir
    else:
        experiment_dir = config["experiment_dir"]
    max_conversations_config = config["max_conversations"]
    selected_models = config["selected_models"]
    truncate_words_config = config["truncate_words"]
    show_sample_data = config["show_sample_data"]
    show_conversation_breakdown = config["show_conversation_breakdown"]
    experiment_type = config["experiment_type"]
    
    # Load prompt templates from config
    prompts = config["prompts"]
    system_prompt = prompts["system"]
    user_prompt_template = prompts["user"]
    detection_prompt_template = prompts["detection"]
    
    # Load logging configuration
    logging_config = config.get("logging", {})
    logging_enabled = logging_config.get("enabled", True)
    logging_level = logging_config.get("level", "INFO")
    logging_output_dir_config = logging_config.get("output_dir", "experiment_dir/conversation_logs")
    
    print(f"📋 Configuration loaded from {config_path}")
    print(f"🎯 Selected models: {selected_models}")
    print(f"📊 Max conversations: {max_conversations_config}")
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"✂️ Truncate words: {truncate_words_config}")
    print(f"🧪 Experiment type: {experiment_type}")
    print(f"ℹ️  Model names and types will be determined from the data")
    
    # ===== LOAD DATA =====
    print(f"\nLoading data from experiment directory: {experiment_dir}")
    data_df = load_data(experiment_dir)
    print(f"Loaded {len(data_df)} rows of data")
    print(f"Data columns: {data_df.columns.tolist()}")
    
    # Show data breakdown
    print(f"Control rows: {len(data_df[data_df['treatment'] == 'control'])}")
    print(f"Treatment rows: {len(data_df[data_df['treatment'] == 'typo'])}")
    print(f"Unique trials: {data_df['trial'].nunique()}")
    print(f"Unique models: {data_df['model'].nunique()}")
    
    # ===== DETERMINE MAX CONVERSATIONS =====
    if max_conversations_config == "max":
        # Calculate maximum possible conversations
        conversations = generate_conversations(data_df)
        max_conversations = len(conversations)
        print(f"📊 Using maximum conversations: {max_conversations} (all available)")
    elif isinstance(max_conversations_config, int) and max_conversations_config > 0:
        max_conversations = max_conversations_config
        print(f"📊 Using specified conversations: {max_conversations}")
    else:
        raise ValueError(
            f"Invalid max_conversations value: {max_conversations_config}. "
            f"Must be a positive integer or 'max' for all available conversations."
        )
    
    # ===== DETERMINE TRUNCATION SETTINGS =====
    if truncate_words_config == "max":
        # No truncation - use full text
        truncate_words = None
        print(f"✂️ No truncation - using full text")
    elif isinstance(truncate_words_config, int) and truncate_words_config > 0:
        truncate_words = truncate_words_config
        print(f"✂️ Truncating to {truncate_words} words")
    else:
        raise ValueError(
            f"Invalid truncate_words value: {truncate_words_config}. "
            f"Must be a positive integer or 'max' for no truncation."
        )
    
    # Apply truncation if configured
    if truncate_words is not None:
        print(f"✂️ Truncating passages and responses to {truncate_words} words")
        data_df['passage'] = data_df['passage'].apply(lambda x: ' '.join(str(x).split()[:truncate_words]) if pd.notna(x) else '')
        data_df['response'] = data_df['response'].apply(lambda x: ' '.join(str(x).split()[:truncate_words]) if pd.notna(x) else '')
        print(f"Text truncated to {truncate_words} words each")
    
    # Show sample data if configured
    if show_sample_data:
        print("\nSample data:")
        print(data_df.head())
    
    # ===== GENERATE CONVERSATIONS =====
    if max_conversations_config != "max":
        # Only generate conversations if we didn't already do it above
        print("\nGenerating conversations...")
        conversations = generate_conversations(data_df)
        print(f"Generated {len(conversations)} conversations")
    else:
        # We already generated conversations above for max calculation
        print(f"\nUsing pre-generated conversations: {len(conversations)} total")
    
    # Show conversation breakdown by model if configured
    if show_conversation_breakdown:
        from collections import Counter
        model_counts = Counter([conv['model_base'] for conv in conversations])
        print(f"Conversations by base model: {dict(model_counts)}")
    
    # ===== PROCESS CONVERSATIONS WITH LOGGING =====
    # Create experiment name for logging
    experiment_name = f"{experiment_type}_{os.path.basename(experiment_dir)}"
    
    # Determine log output directory
    if logging_output_dir_config == "experiment_dir/conversation_logs":
        # Default: save logs in the same directory as results
        log_output_dir = os.path.join(experiment_dir, "conversation_logs")
    elif logging_output_dir_config.startswith("experiment_dir/"):
        # Relative to experiment directory
        relative_path = logging_output_dir_config.replace("experiment_dir/", "")
        log_output_dir = os.path.join(experiment_dir, relative_path)
    else:
        # Absolute path or other custom path
        log_output_dir = logging_output_dir_config
    
    with conversation_logger(
        experiment_name=experiment_name,
        output_dir=log_output_dir,
        enabled=logging_enabled,
        log_level=logging_level
    ) as logger:
        # Log experiment metadata
        logger.log_experiment_metadata({
            "config_path": config_path,
            "selected_models": selected_models,
            "max_conversations": max_conversations,
            "experiment_dir": experiment_dir,
            "truncate_words": truncate_words,
            "experiment_type": experiment_type,
            "total_conversations": len(conversations)
        })
        
        # Process conversations with logging
        process_conversations_for_choices(
            conversations=conversations,
            data_file=experiment_dir,  # Use experiment directory for results path
            max_conversations=max_conversations,
            selected_models=selected_models,
            truncate_words=truncate_words,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            detection_prompt_template=detection_prompt_template,
            experiment_type=experiment_type,
            logger=logger  # Pass logger to processing function
        )
    
    print("\nTest complete!")




if __name__ == "__main__":
    import sys
    main()