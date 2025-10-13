"""
Shared utilities for experiment scripts.

This module contains all the shared functions used by run_experiment.py,
run_experiment_2T.py, and run_experiment_IR.py to avoid circular imports.
"""

import pandas as pd
import torch
import os
import yaml
from typing import List, Optional, Set, Tuple, Dict
from tqdm import tqdm
from collections import Counter

from model.load import load_model
from model.anthropic import load_anthropic_model
from model.gemini import load_gemini_model
from model.openai import load_openai_model
from util.elicit import get_choice_token_logits_from_token_ids


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


def get_logs_directory_path(data_file: str) -> str:
    """
    Get the expected logs directory path based on the data file path.
    Mirrors the results directory structure: experiments/ -> logs/
    
    Log files go directly in the dataset-level directory, not in subdirectories.
    Example: experiments/WikiSum-AT_IR_Pr/subdir/file.csv -> logs/WikiSum-AT_IR_Pr/
    
    Args:
        data_file: Path to the experiment directory or file
        
    Returns:
        Path to the logs directory
    """
    # Normalize path and handle both file and directory paths
    normalized_path = data_file.replace('\\', '/')
    
    # If it's a file path (has .csv or other extension), strip the filename
    if '.' in os.path.basename(normalized_path):
        # Remove the filename, keep only the directory path
        normalized_path = os.path.dirname(normalized_path).replace('\\', '/')
    
    # Convert experiment directory path to logs directory structure
    exp_path_parts = normalized_path.split('/')
    
    # Find the index of 'experiments' in the path
    exp_idx = None
    for i, part in enumerate(exp_path_parts):
        if part == 'experiments':
            exp_idx = i
            break
    
    if exp_idx is not None and exp_idx + 1 < len(exp_path_parts):
        # Extract the dataset name (e.g., "WikiSum-AT_IR_Pr")
        # This is the immediate subdirectory after 'experiments'
        dataset_name = exp_path_parts[exp_idx + 1]
        # Logs go directly in the dataset directory, not in further subdirectories
        logs_dir = f"results_and_data/logs/{dataset_name}"
    else:
        # Fallback to a default directory
        logs_dir = "results_and_data/logs/default_dir"
    
    return logs_dir


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
        print(f"[FILE] Found existing results: {results_file}")
        print(f"   Existing conversations: {len(df)}")
        return df
    except Exception as e:
        print(f"[WARNING]  Warning: Could not load existing results from {results_file}: {e}")
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
        print(f"[WARNING] Found {len(identical_probs)} conversations with false data (identical probabilities)")
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
        print("[NOTE] No existing results found - will process all conversations")
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
    
    print(f"[DATA] Conversation filtering:")
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
    
    print(f"[MERGE] Merged results:")
    print(f"   Existing conversations: {len(existing_df)}")
    print(f"   New/updated conversations: {len(new_results)}")
    print(f"   Final total: {len(merged_df)}")
    
    return merged_df


def load_data(experiment_dir: str) -> pd.DataFrame:
    """
    Load assist tag data from control and treatment CSV files.
    
    Args:
        experiment_dir: Path to the experiment directory containing control.csv and treatment.csv
        
    Returns:
        Combined DataFrame with columns: trial, model, treatment, passage, response
    """
    # Normalize path separators to avoid mixing forward and backslashes
    experiment_dir = experiment_dir.replace('\\', '/')
    
    control_file = f"{experiment_dir}/control.csv"
    treatment_file = f"{experiment_dir}/treatment.csv"
    
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


def infer_model_type(model_name: str) -> str:
    """
    Automatically infer model type from model name.
    
    Args:
        model_name: The model name (e.g., "claude-3-5-sonnet-20241022", "gemini-1.5-flash", "mock")
        
    Returns:
        The inferred model type ("mock", "anthropic", "google", "openai", "huggingface")
        
    Raises:
        ValueError: If the model name doesn't match any known pattern
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower.startswith('mock'):
        return "mock"
    elif model_name_lower.startswith('claude-'):
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
            f"Supported patterns: mock*, claude-*, gemini-*, gpt-*, or HuggingFace models (containing '/' or starting with microsoft/, meta/, huggingface/). "
            f"Please check the model name or add support for this model type."
        )


def load_prompts_from_file(experiment_type: str, prompt_paradigm: str = "rec", system_priming: str = "none") -> Dict[str, str]:
    """
    Load prompts from both general and experiment-specific prompt files.
    
    Prompts are loaded in this order (later files override earlier ones):
    1. configs/operationalizations/prompts_general.yaml (shared prompts)
    2. configs/operationalizations/{experiment_type}/prompts.yaml (experiment-specific)
    
    Args:
        experiment_type: Type of experiment (AT_2T, AT_IR, UT_2T, UT_Shi, etc.)
        prompt_paradigm: Either 'rec' (recognition) or 'pref' (preference)
        system_priming: System prompt priming type ('none', 'AT', 'UT', etc.)
        
    Returns:
        Dictionary containing 'system', 'user', and 'detection' prompts
        
    Raises:
        FileNotFoundError: If prompts file doesn't exist
        ValueError: If required prompts are missing
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Start with general prompts
    general_prompts_file = os.path.join(project_root, "configs/operationalizations/prompts_general.yaml")
    prompts_data = {}
    
    if os.path.exists(general_prompts_file):
        try:
            with open(general_prompts_file, "r") as f:
                prompts_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load general prompts from {general_prompts_file}: {e}")
    
    # Load experiment-specific prompts and merge/override
    experiment_prompts_file = os.path.join(project_root, f"configs/operationalizations/{experiment_type}/prompts.yaml")
    
    if os.path.exists(experiment_prompts_file):
        try:
            with open(experiment_prompts_file, "r") as f:
                experiment_prompts = yaml.safe_load(f) or {}
                # Merge experiment-specific prompts (overriding general ones)
                prompts_data.update(experiment_prompts)
        except Exception as e:
            raise ValueError(f"Failed to load prompts from {experiment_prompts_file}: {e}")
    
    if not prompts_data:
        raise FileNotFoundError(
            f"No prompts found. Checked:\n"
            f"  - {general_prompts_file}\n"
            f"  - {experiment_prompts_file}"
        )
    
    # Map paradigm-specific detection prompt
    detection_key = f"{prompt_paradigm}_detection"
    
    if detection_key not in prompts_data:
        raise ValueError(
            f"Missing '{detection_key}' prompt. "
            f"Available keys: {list(prompts_data.keys())}"
        )
    
    # Get system priming text
    priming_key = f"system_pr_{system_priming}"
    priming_text = prompts_data.get(priming_key, "")
    
    # Build system prompt with priming
    system_template = prompts_data.get("system", "")
    if "{priming}" in system_template:
        system_prompt = system_template.format(priming=priming_text)
    else:
        system_prompt = system_template
    
    # Build prompts dictionary
    prompts = {
        "system": system_prompt,
        "user": prompts_data.get("user", ""),
        "detection": prompts_data[detection_key]
    }
    
    # Validate required prompts are present
    # Note: 'user' prompt is optional for some experiment types (e.g., UT_Shi)
    required_prompts = ["system", "detection"]
    missing_prompts = [key for key in required_prompts if not prompts.get(key)]
    if missing_prompts:
        raise ValueError(
            f"Missing required prompts: {missing_prompts}"
        )
    
    return prompts


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
    
    # Handle prompts - either inline or from prompt_set
    if "prompt_set" in config:
        # Load prompts from separate file
        prompt_paradigm = config.get("prompt_paradigm", "rec")
        system_priming = config.get("system_priming", "none")
        experiment_type = config["experiment_type"]
        
        try:
            prompts = load_prompts_from_file(experiment_type, prompt_paradigm, system_priming)
            config["prompts"] = prompts
        except Exception as e:
            raise ValueError(
                f"Failed to load prompts for experiment_type '{experiment_type}', "
                f"paradigm '{prompt_paradigm}', and priming '{system_priming}': {e}"
            )
    elif "prompts" in config:
        # Use inline prompts (legacy support)
        required_prompts = ["system", "user", "detection"]
        missing_prompts = [prompt for prompt in required_prompts if prompt not in config["prompts"]]
        if missing_prompts:
            raise ValueError(
                f"Missing required prompt templates in {config_path}: {missing_prompts}. "
                f"Please add these prompt templates to the 'prompts' section of your config file."
            )
    else:
        # No prompts specified - try to infer from experiment type and config filename
        experiment_type = config["experiment_type"]
        
        # Infer paradigm from config filename
        config_filename = os.path.basename(config_path)
        if "pref" in config_filename:
            prompt_paradigm = "pref"
        else:
            prompt_paradigm = "rec"
        
        # Get system priming setting (default to "none")
        system_priming = config.get("system_priming", "none")
        
        try:
            prompts = load_prompts_from_file(experiment_type, prompt_paradigm, system_priming)
            config["prompts"] = prompts
            print(f"[INFO] Auto-loaded prompts for {experiment_type}/{prompt_paradigm}/priming={system_priming}")
        except Exception as e:
            raise ValueError(
                f"No 'prompts' or 'prompt_set' found in config, and auto-loading failed: {e}"
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


def show_available_models(experiment_dir: str) -> None:
    """
    Show available models in the experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory containing control.csv and treatment.csv
    """
    if not os.path.exists(experiment_dir):
        print(f"[ERROR] Experiment directory not found: {experiment_dir}")
        return
    
    try:
        df = load_data(experiment_dir)
        print(f"\n[DATA] Available models in {experiment_dir}:")
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
            print("[ERROR] No 'model' column found in data")
            print(f"Available columns: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"[ERROR] Error reading experiment directory: {e}")


def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    import argparse
    
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
