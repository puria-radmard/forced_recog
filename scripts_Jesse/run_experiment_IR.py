"""
Injection Recognition (IR) Experiments

This module contains experiment-specific functions for IR experiments:
- AT_IR: Assist Tag Injected Response (injection recognition)
- UT_IR: User Tag Injected Response (injection recognition)

For 2T experiments (AT_2T, UT_2T), use run_experiment_2T.py instead.

All shared functions are imported from experiment_utils.py.
"""

import pandas as pd
import torch
import os
from tqdm import tqdm
from typing import List, Dict

# Import shared functions from the experiment utilities module
from experiment_utils import (
    truncate_text, get_choice_tokens, get_results_file_path, load_existing_results,
    identify_false_data_conversations, filter_conversations_to_rerun, merge_results,
    load_data, parse_arguments, show_available_models, infer_model_type, 
    load_prompts_from_file, load_config
)
from model.load import load_model
from model.anthropic import load_anthropic_model
from model.gemini import load_gemini_model
from model.openai import load_openai_model
from util.elicit import get_choice_token_logits_from_token_ids


def generate_conversations_IR_old(data_df: pd.DataFrame) -> List[Dict]:
    """
    Generate conversations for AT_IR (Injection Recognition) experiments.
    
    For "other_model" experiments:
    - Only the target model (from control data) should do the evaluation
    - Target model evaluates both control responses (its own) and treatment responses (from other models)
    
    For other treatment types (typo, capitalization, etc.), only treatment rows are processed.
    
    Each row becomes TWO conversations to control for position bias:
    - One where injected_text_token="1" and original_text_token="2"
    - One where injected_text_token="2" and original_text_token="1"
    
    Args:
        data_df: DataFrame with columns [trial, model, treatment, passage, response]
        
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    
    # Check what treatment types are present
    treatment_types = data_df['treatment'].unique()
    has_other_model = 'other_model' in treatment_types
    has_control = 'control' in treatment_types
    
    print(f"Treatment types found: {treatment_types}")
    print(f"Has 'other_model' treatment: {has_other_model}")
    print(f"Has 'control' treatment: {has_control}")
    
    # Determine which rows to process and which model should do the evaluation
    if has_other_model and has_control:
        # For other_model experiments: only the target model (from control data) does evaluation
        target_model = data_df[data_df['treatment'] == 'control']['model'].iloc[0]
        print(f"Processing other_model experiment - target model: {target_model}")
        print("Target model will evaluate both control and treatment responses")
        rows_to_process = data_df
        evaluation_model = target_model
    else:
        # For other treatment types: only process treatment rows
        print("Processing only treatment rows (no control comparisons)")
        rows_to_process = data_df[data_df['treatment'] != 'control']
        evaluation_model = None  # Will be determined per row
    
    # Each row becomes TWO conversations to control for position bias
    for idx, row in rows_to_process.iterrows():
        model = row['model']  # The model that generated the response
        treatment = row['treatment']
        passage = row['passage']
        response = row['response']
        trial = row['trial']
        
        # Determine which model should do the evaluation
        if evaluation_model:
            # For other_model experiments: always use the target model
            model_base = evaluation_model
        else:
            # For other treatment types: use the model from the data
            model_base = model
        
        # Create two conversations for this single response to control position bias
        # Conversation 1: injected_text_token="1", original_text_token="2"
        conversation_1 = {
            'conversation_id': f"trial{trial}_{model_base}_{treatment}_{idx}_v1",
            'trial': trial,
            'row_idx': idx,
            'model_base': model_base,  # The model doing the evaluation
            'response_model': model,   # The model that generated the response
            'treatment': treatment,    # The treatment type (control, typo, other_model, capitalization)
            'passage': passage,
            'response': response,      # The single response to evaluate
            'has_injection': treatment != 'control',  # True if treatment contains injected text, False for control
            'injected_text_token': "1",
            'original_text_token': "2",
            'version': 1
        }
        
        # Conversation 2: injected_text_token="2", original_text_token="1"
        conversation_2 = {
            'conversation_id': f"trial{trial}_{model_base}_{treatment}_{idx}_v2",
            'trial': trial,
            'row_idx': idx,
            'model_base': model_base,  # The model doing the evaluation
            'response_model': model,   # The model that generated the response
            'treatment': treatment,    # The treatment type (control, typo, other_model, capitalization)
            'passage': passage,
            'response': response,      # The single response to evaluate
            'has_injection': treatment != 'control',  # True if treatment contains injected text, False for control
            'injected_text_token': "2",
            'original_text_token': "1",
            'version': 2
        }
        
        conversations.extend([conversation_1, conversation_2])
    
    print(f"Generated {len(conversations)} conversations from {len(rows_to_process)} rows")
    if evaluation_model:
        print(f"All evaluations will be performed by: {evaluation_model}")
    return conversations


def generate_conversations_IR(data_df: pd.DataFrame, experiment_type: str) -> List[Dict]:
    """
    Generate conversations for IR (Injection Recognition) experiments - both AT_IR and UT_IR.
    
    Generation logic is IDENTICAL for AT_IR and UT_IR. The only difference between them
    is in conversation formatting (handled in process_conversations_for_choices_IR).
    
    For "other_model" experiments:
    - Only the target model (from control data) should do the evaluation
    - Target model evaluates both control responses (its own) and treatment responses (from other models)
    
    For other treatment types, only treatment rows are processed.
    
    Each row becomes TWO conversations to control for position bias:
    - One where injected_text_token="1" and original_text_token="2"
    - One where injected_text_token="2" and original_text_token="1"
    
    Args:
        data_df: DataFrame with columns [trial, model, treatment, passage, response]
        experiment_type: "AT_IR" or "UT_IR" (used for logging/conversation IDs only)
        
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    
    # Check what treatment types are present
    treatment_types = data_df['treatment'].unique()
    has_other_model = 'other_model' in treatment_types
    has_control = 'control' in treatment_types
    
    print(f"Treatment types found: {treatment_types}")
    print(f"Has 'other_model' treatment: {has_other_model}")
    print(f"Has 'control' treatment: {has_control}")
    
    # Determine which rows to process and which model should do the evaluation
    if has_other_model and has_control:
        # For both AT_IR and UT_IR other_model experiments: 
        # Only the target model (from control data) does evaluation
        target_model = data_df[data_df['treatment'] == 'control']['model'].iloc[0]
        print(f"Processing {experiment_type} other_model experiment - target model: {target_model}")
        print("Target model will evaluate both control and treatment responses")
        rows_to_process = data_df
        evaluation_model = target_model
    else:
        # For other treatment types (typo, capitalization, etc.): only process treatment rows
        print("Processing only treatment rows (no control comparisons)")
        rows_to_process = data_df[data_df['treatment'] != 'control']
        evaluation_model = None  # Will be determined per row
    
    # Each row becomes TWO conversations to control for position bias
    for idx, row in rows_to_process.iterrows():
        model = row['model']  # The model that generated the response
        treatment = row['treatment']
        passage = row['passage']
        response = row['response']
        trial = row['trial']
        
        # Determine which model should do the evaluation
        if evaluation_model:
            # For AT_IR other_model experiments: always use the target model
            model_base = evaluation_model
        else:
            # For other treatment types: use the model from the data
            model_base = model
        
        # Determine has_injection based on treatment (same for both AT_IR and UT_IR)
        has_injection = treatment != 'control'
        
        # Create two conversations for this single response to control position bias
        # Conversation 1: injected_text_token="1", original_text_token="2"
        conversation_1 = {
            'conversation_id': f"trial{trial}_{model_base}_{treatment}_{idx}_v1",
            'trial': trial,
            'row_idx': idx,
            'model_base': model_base,  # The model doing the evaluation
            'response_model': model,   # The model that generated the response
            'treatment': treatment,    # The treatment type (control, typo, other_model, capitalization)
            'passage': passage,
            'response': response,      # The single response to evaluate
            'has_injection': has_injection,
            'injected_text_token': "1",
            'original_text_token': "2",
            'version': 1
        }
        
        # Conversation 2: injected_text_token="2", original_text_token="1"
        conversation_2 = {
            'conversation_id': f"trial{trial}_{model_base}_{treatment}_{idx}_v2",
            'trial': trial,
            'row_idx': idx,
            'model_base': model_base,  # The model doing the evaluation
            'response_model': model,   # The model that generated the response
            'treatment': treatment,    # The treatment type (control, typo, other_model, capitalization)
            'passage': passage,
            'response': response,      # The single response to evaluate
            'has_injection': has_injection,
            'injected_text_token': "2",
            'original_text_token': "1",
            'version': 2
        }
        
        conversations.extend([conversation_1, conversation_2])
    
    print(f"Generated {len(conversations)} conversations from {len(rows_to_process)} rows")
    if evaluation_model:
        print(f"All evaluations will be performed by: {evaluation_model}")
    return conversations
    

def process_conversations_for_choices_IR(
    conversations: List[Dict],
    data_file: str,
    max_conversations: int = 4,
    truncate_words: int = None,
    system_prompt: str = None,
    user_prompt_template: str = None,
    detection_prompt_template: str = None,
    experiment_type: str = "AT_IR",
    logger = None
) -> None:
    """
    Process conversations for IR injection recognition experiments (AT_IR and UT_IR).
    Automatically detects and re-runs only conversations with false data.
    Processes all models found in the data.
    
    The key difference between AT_IR and UT_IR is in conversation formatting:
    - AT_IR: Uses in-context messages (like AT_2T)
    - UT_IR: Puts entire conversation in user_message as transcript (like UT_2T)
    
    Args:
        conversations: List of conversation dictionaries
        data_file: Path to the data file (used to determine results directory)
        max_conversations: Maximum number of conversations to process per model
        truncate_words: Maximum number of words for text truncation (None for no truncation)
        system_prompt: System prompt template
        user_prompt_template: User prompt template
        detection_prompt_template: Detection prompt template
        experiment_type: Type of experiment ("AT_IR" or "UT_IR")
        logger: Conversation logger instance
    """
    print(f"Processing {experiment_type} conversations with max {max_conversations} per model")
    
    # ===== CHECK FOR EXISTING RESULTS =====
    results_file = get_results_file_path(data_file)
    existing_df = load_existing_results(results_file)
    false_data_ids = identify_false_data_conversations(existing_df)
    
    # ===== FILTER CONVERSATIONS TO RE-RUN =====
    conversations_to_rerun = filter_conversations_to_rerun(conversations, false_data_ids, existing_df)
    
    if not conversations_to_rerun:
        print("[OK] All conversations already have valid results - nothing to re-run!")
        return
    
    # Group conversations by base model (the model doing the evaluation)
    conversations_by_model = {}
    for conv in conversations_to_rerun:
        base_model = conv['model_base']
        if base_model not in conversations_by_model:
            conversations_by_model[base_model] = []
        conversations_by_model[base_model].append(conv)
    
    # Process all models found in the data
    print(f"Found {len(conversations_by_model)} unique base models: {list(conversations_by_model.keys())}")
    
    all_results = []
    
    # Process each base model separately
    for base_model, model_conversations in conversations_by_model.items():
        print(f"\n=== Processing base model: {base_model} ===")
        
        # Load the appropriate model wrapper
        if base_model.startswith('mock'):
            # Mock model for testing
            model_name = base_model
            chat_wrapper = load_model(base_model)
        elif base_model.startswith('anthropic_'):
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
            # Apply truncation if configured for processing
            passage = truncate_text(conv['passage'], truncate_words)
            response = truncate_text(conv['response'], truncate_words)
            
            # Create the user prompt using the (possibly truncated) passage
            user_prompt = user_prompt_template.format(passage=passage)
            
            # Format detection prompt with dynamic tokens
            
            
            # Create conversation based on experiment type
            if experiment_type == "AT_IR":
                # AT_IR: Use in-context messages (model sees its own previous responses)
                full_detection_prompt = detection_prompt_template.format(
                injected_text_token=conv['injected_text_token'], 
                original_text_token=conv['original_text_token']
                )
                conversation_full = chat_wrapper.format_chat(
                    system_prompt=system_prompt,
                    in_context_questions=[user_prompt], 
                    in_context_answers=[response], 
                    user_message=full_detection_prompt,
                )
            elif experiment_type == "UT_IR":
                # UT_IR: Put entire conversation in user_message
                full_user_prompt = user_prompt_template.format(passage=passage)
                full_detection_prompt = detection_prompt_template.format(user_message=full_user_prompt, response=response, 
                injected_text_token=conv['injected_text_token'], 
                original_text_token=conv['original_text_token'])
                conversation_full = chat_wrapper.format_chat(
                    system_prompt=system_prompt,
                    user_message=full_detection_prompt,
                )
            else:
                raise ValueError(f"Invalid experiment type for IR processing: {experiment_type}")
            
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
                            "has_injection": conv.get('has_injection', 'unknown'),
                            "row_idx": conv.get('row_idx', 'unknown'),
                            "version": conv.get('version', 'unknown'),
                            "injected_text_token": conv.get('injected_text_token', 'unknown'),
                            "original_text_token": conv.get('original_text_token', 'unknown')
                        },
                        conversation_text=conversation_full
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
                
                # Determine if the choice was correct based on injection status and token mapping
                # For injection recognition:
                # - If has_injection=True, correct answer is the injected_text_token
                # - If has_injection=False, correct answer is the original_text_token
                if conv['has_injection']:
                    correct_choice = conv['injected_text_token']  # Should select injected text
                    selected_meaning = "injected" if selected_choice == conv['injected_text_token'] else "original"
                    correct_meaning = "injected"
                else:
                    correct_choice = conv['original_text_token']  # Should select original text
                    selected_meaning = "original" if selected_choice == conv['original_text_token'] else "injected"
                    correct_meaning = "original"
                
                is_correct = selected_choice == correct_choice
                
                print(f"  -> Selected choice: {selected_choice} ({selected_meaning})")
                print(f"  -> Correct choice: {correct_choice} ({correct_meaning})")
                print(f"  -> Model {'[CORRECT]' if is_correct else '[INCORRECT]'}")
                
                # Log results if logger is available
                if logger:
                    logger.log_results({
                        "choice_probabilities": [prob_1, prob_2],
                        "selected_choice": selected_choice,
                        "correct_choice": correct_choice,
                        "selected_meaning": selected_meaning,
                        "correct_meaning": correct_meaning,
                        "is_correct": is_correct,
                        "accuracy": 1.0 if is_correct else 0.0
                    })
                    
                    logger.finish_conversation(success=True)
                
                # Store results
                all_results.append({
                    'conversation_id': conv['conversation_id'],
                    'trial': conv['trial'],
                    'row_idx': conv['row_idx'],
                    'model_base': conv['model_base'],
                    'treatment': conv['treatment'],
                    'has_injection': conv['has_injection'],
                    'version': conv['version'],
                    'injected_text_token': conv['injected_text_token'],
                    'original_text_token': conv['original_text_token'],
                    'prob_choice_1': prob_1,
                    'prob_choice_2': prob_2,
                    'selected_choice': selected_choice,
                    'correct_choice': correct_choice,
                    'selected_meaning': selected_meaning,
                    'correct_meaning': correct_meaning,
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
        print(f"\n[DATA] RESULTS SUMMARY:")
        print(f"  New conversations processed: {len(all_results)}")
        print(f"  Total conversations in file: {len(merged_df)}")
        print(f"  Unique base models: {merged_df['model_base'].nunique()}")
        
        # Treatment distribution
        print(f"  Treatment distribution:")
        treatment_counts = merged_df['treatment'].value_counts()
        for treatment, count in treatment_counts.items():
            print(f"    {treatment}: {count}")
        
        # Position bias control
        print(f"  Position bias control:")
        version_counts = merged_df['version'].value_counts()
        for version, count in version_counts.items():
            print(f"    Version {version}: {count}")
        
        # Accuracy by version
        print(f"  Accuracy by version:")
        version_accuracy = merged_df.groupby('version')['is_correct'].agg(['mean', 'count']).round(3)
        for version, row in version_accuracy.iterrows():
            print(f"    Version {version}: {row['mean']:.3f} ({int(row['count'] * row['mean'])}/{int(row['count'])})")
        
        print(f"  Choice selection summary:")
        choice_counts = merged_df['selected_choice'].value_counts()
        for choice, count in choice_counts.items():
            print(f"    Choice {choice}: {count} times")
        
        print(f"  Accuracy summary:")
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
            correct_mark = "[OK]" if row['is_correct'] else "[X]"
            print(f"    {correct_mark} {row['conversation_id']}: {row['model_base']} chose {row['selected_choice']} ({row['selected_meaning']}) (prob: {row['prob_choice_1']:.3f} vs {row['prob_choice_2']:.3f})")
    else:
        print("No new results to save")

