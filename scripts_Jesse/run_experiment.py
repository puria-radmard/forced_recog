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
- AT_2T: Assist Tag 2-Turn (which response was originally produced by the model)
- AT_IR: Assist Tag Injection Recognition (detect injected text in responses)
- UT_2T: User Tag 2-Turn (which response the model prefers)
- UT_IR: User Tag Injection Recognition (detect injected text in user prompts)
- UT_Shi: User Tag Shi variant

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
# Windows pathing fix - must be imported and called first
import sys
import os
import windows_pathing_fix
windows_pathing_fix.fix_pathing()

import pandas as pd
import yaml
import torch
import argparse
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Dict
import copy
from dotenv import load_dotenv
from util.conversation_logger import conversation_logger
from model.load import load_model
from model.base import BaseChatWrapper
from model.anthropic import load_anthropic_model, AnthropicChatWrapper
from model.gemini import load_gemini_model, GeminiChatWrapper
from model.openai import load_openai_model, OpenAIChatWrapper

from sft_utils.lora import download_and_apply_lora

# Prompts are now loaded from YAML config files

from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig

# Import shared functions from utilities
from experiment_utils import (
    truncate_text, get_choice_tokens, get_results_file_path, get_logs_directory_path,
    load_existing_results, identify_false_data_conversations, filter_conversations_to_rerun,
    merge_results, load_data, parse_arguments, show_available_models, infer_model_type, 
    load_prompts_from_file, load_config
)

# Import experiment-specific functions
from run_experiment_2T import generate_conversations_2T, process_conversations_for_choices_2T
from run_experiment_IR import generate_conversations_IR, process_conversations_for_choices_IR

# Load environment variables from .env file
load_dotenv()



document_columns = [
    'row_a_idx', 
    'row_b_idx', 'model_a', 'treatment_a', 'model_b', 'treatment_b',
    'prob_choice_1', 'prob_choice_2', 
]


# generate_conversations function is now imported from experiment-specific files


# All shared functions are now imported from experiment_utils.py


# process_conversations_for_choices function is now imported from experiment-specific files


# load_completion_status function is kept here as it's specific to document_columns
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



# All remaining functions are imported from experiment_utils.py


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
                experiment_dir = config.get("experiment_dir")
            show_available_models(experiment_dir)
            return
    else:
        # IDE mode: use hardcoded config path
        config_path = "configs/operationalizations/UT_2T/rec_config_Pr.yaml"
    
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
    # Default to processing all models found in the data
    selected_models = ["all"]
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
    logging_output_dir_config = logging_config.get("output_dir", None)
    
    print(f"[INFO] Configuration loaded from {config_path}")
    print(f"[TARGET] Processing all models found in data")
    print(f"[DATA] Max conversations: {max_conversations_config}")
    print(f"[FILE] Experiment directory: {experiment_dir}")
    print(f"[TRUNC] Truncate words: {truncate_words_config}")
    print(f"[EXP] Experiment type: {experiment_type}")
    print(f"[INFO]  Model names and types will be determined from the data")
    
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
        # Calculate maximum possible conversations based on experiment type
        if experiment_type in ["AT_2T", "UT_2T", "UT_Shi"]:
            conversations = generate_conversations_2T(data_df)
        elif experiment_type in ["AT_IR", "UT_IR"]:
            conversations = generate_conversations_IR(data_df, experiment_type)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        max_conversations = len(conversations)
        print(f"[DATA] Using maximum conversations: {max_conversations} (all available)")
    elif isinstance(max_conversations_config, int) and max_conversations_config > 0:
        max_conversations = max_conversations_config
        print(f"[DATA] Using specified conversations: {max_conversations}")
    else:
        raise ValueError(
            f"Invalid max_conversations value: {max_conversations_config}. "
            f"Must be a positive integer or 'max' for all available conversations."
        )
    
    # ===== DETERMINE TRUNCATION SETTINGS =====
    if truncate_words_config == "max":
        # No truncation - use full text
        truncate_words = None
        print(f"[TRUNC] No truncation - using full text")
    elif isinstance(truncate_words_config, int) and truncate_words_config > 0:
        truncate_words = truncate_words_config
        print(f"[TRUNC] Truncating to {truncate_words} words")
    else:
        raise ValueError(
            f"Invalid truncate_words value: {truncate_words_config}. "
            f"Must be a positive integer or 'max' for no truncation."
        )
    
    # Apply truncation if configured
    if truncate_words is not None:
        print(f"[TRUNC] Truncating passages and responses to {truncate_words} words")
        data_df['passage'] = data_df['passage'].apply(lambda x: ' '.join(str(x).split()[:truncate_words]) if pd.notna(x) else '')
        data_df['response'] = data_df['response'].apply(lambda x: ' '.join(str(x).split()[:truncate_words]) if pd.notna(x) else '')
        print(f"Text truncated to {truncate_words} words each")
    
    # Show sample data if configured
    if show_sample_data:
        print("\nSample data:")
        print(data_df.head())
    
    # ===== GENERATE CONVERSATIONS =====
    if max_conversations_config != "max":
        # Only generate conversations if we didn't already do it above for max calculation
        print("\nGenerating conversations...")
        if experiment_type in ["AT_2T", "UT_2T", "UT_Shi"]:
            # Use 2T conversation generation
            conversations = generate_conversations_2T(data_df)
            print(f"Generated {len(conversations)} 2T conversations")
        elif experiment_type in ["AT_IR", "UT_IR"]:
            # Use IR conversation generation
            conversations = generate_conversations_IR(data_df, experiment_type)
            print(f"Generated {len(conversations)} {experiment_type} conversations (with position bias control)")
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
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
    # Auto-generate log path if not specified (default behavior)
    if not logging_output_dir_config:
        # Auto-generate logs directory path mirroring results structure
        log_output_dir = get_logs_directory_path(experiment_dir)
        print(f"[INFO] Auto-generated log directory: {log_output_dir}")
    elif logging_output_dir_config == "experiment_dir/conversation_logs":
        # Legacy: save logs in the same directory as results
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
            "selected_models": ["all"],  # Always process all models found in data
            "max_conversations": max_conversations,
            "experiment_dir": experiment_dir,
            "truncate_words": truncate_words,
            "experiment_type": experiment_type,
            "total_conversations": len(conversations)
        })
        
        # Process conversations with logging - route to appropriate experiment type
        if experiment_type in ["AT_2T", "UT_2T", "UT_Shi"]:
            # Use 2T conversation processing
            process_conversations_for_choices_2T(
                conversations=conversations,
                data_file=experiment_dir,  # Use experiment directory for results path
                max_conversations=max_conversations,
                truncate_words=truncate_words,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                detection_prompt_template=detection_prompt_template,
                experiment_type=experiment_type,
                logger=logger  # Pass logger to processing function
            )
        elif experiment_type in ["AT_IR", "UT_IR"]:
            # Use IR conversation processing
            process_conversations_for_choices_IR(
                conversations=conversations,
                data_file=experiment_dir,  # Use experiment directory for results path
                max_conversations=max_conversations,
                truncate_words=truncate_words,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                detection_prompt_template=detection_prompt_template,
                experiment_type=experiment_type,
                logger=logger  # Pass logger to processing function
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    print("\nTest complete!")




if __name__ == "__main__":
    import sys
    main()