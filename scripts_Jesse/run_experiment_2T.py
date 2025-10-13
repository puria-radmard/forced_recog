"""
2T Experiments Only - AT_2T and UT_2T

This module contains experiment-specific functions for 2T (Two-Turn) experiments:
- AT_2T: Assist Tag 2-Turn (model self-recognition)
- UT_2T: User Tag 2-Turn (user preference)

For AT_IR experiments, use run_experiment_IR.py instead.

All shared functions are imported from run_experiment.py.
"""

import pandas as pd
import torch
import os
from tqdm import tqdm
from typing import List, Dict
from collections import Counter

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


def generate_conversations_2T(data_df: pd.DataFrame) -> List[Dict]:
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


def process_conversations_for_choices_2T(
    conversations: List[Dict],
    data_file: str,
    max_conversations: int = 4,
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
    Processes all models found in the data.
    
    Args:
        conversations: List of conversation dictionaries
        data_file: Path to the data file (used to determine results directory)
        max_conversations: Maximum number of conversations to process per model
        truncate_words: Maximum number of words for text truncation (None for no truncation)
        system_prompt: System prompt template
        user_prompt_template: User prompt template
        detection_prompt_template: Detection prompt template
        experiment_type: Type of experiment ("AT_2T" or "UT_2T")
        logger: Conversation logger instance
    """
    print(f"Processing 2T conversations with max {max_conversations} per model")
    
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
            # Create the user prompt using the full passage (no truncation for logging)
            user_prompt_full = user_prompt_template.format(passage=conv['passage'])
            
            # Apply truncation if configured for processing
            passage = truncate_text(conv['passage'], truncate_words)
            response_1 = truncate_text(conv['response_1'], truncate_words)
            response_2 = truncate_text(conv['response_2'], truncate_words)
            
            # Create the user prompt using the (possibly truncated) passage for processing
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
            elif experiment_type == "UT_Shi":
                full_detection_prompt = detection_prompt_template.format(passage=passage, response_1=response_1, response_2=response_2)
                conversation_full = chat_wrapper.format_chat(
                    system_prompt=system_prompt,
                    user_message=full_detection_prompt,
                )
            else:
                raise ValueError(f"Invalid experiment type for 2T processing: {experiment_type}")
            
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
                            "treatment": conv.get('treatment_other', 'unknown'),
                            "response_1_source": conv.get('response_1_source', 'unknown'),
                            "response_2_source": conv.get('response_2_source', 'unknown')
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
                print(f"  -> Selected choice: {selected_choice} (response from {conv['response_1_source'] if selected_choice == '1' else conv['response_2_source']})")
                
                # Determine if the model selected the control response (correct choice)
                # Correct choice is when the model selects the control response
                # If response_1_source is 'control', then choice 1 is correct
                # If response_2_source is 'control', then choice 2 is correct
                correct_choice = "1" if conv['response_1_source'] == 'control' else "2"
                is_correct = selected_choice == correct_choice
                
                print(f"  -> Correct choice: {correct_choice} (control response)")
                print(f"  -> Model {'[CORRECT]' if is_correct else '[INCORRECT]'}")
                
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
        print(f"\n[DATA] RESULTS SUMMARY:")
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
            correct_mark = "[OK]" if row['is_correct'] else "[X]"
            print(f"    {correct_mark} {row['conversation_id']}: {row['model_base']} chose {row['selected_choice']} (prob: {row['prob_choice_1']:.3f} vs {row['prob_choice_2']:.3f})")
    else:
        print("No new results to save")