from prompts.summary import DATASET_SYSTEM_PROMPTS, SUMMARIZE_PROMPT_TEMPLATES
from load_data import load_dataset
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
import os

def load_sft_data(run_name: str, dataset_name: str, temp: float, style: str, *_, datasets_dir: Optional[str] = None, results_dir: Optional[str] = None) -> pd.DataFrame:
    """Load all training data for SFT from generated summaries."""
    # Load original train data to get articles
    train_data, _, _ = load_dataset(dataset_name, splits = ["train"], datasets_dir = datasets_dir)
    
    # Load all trial summaries for this temp/style combination
    results_dir = results_dir or "results_and_data/results"
    summaries_dir = os.path.join(results_dir, f'main/{run_name}/train/model_summaries')
    
    all_summaries = []
    trial_idx = 0
    while True:
        summary_file = f"{summaries_dir}/T{temp}_trial{trial_idx}_style{style}.csv"
        if not os.path.exists(summary_file):
            break
            
        print(f"Loading {summary_file}")
        trial_summaries = pd.read_csv(summary_file)
        trial_summaries['trial_idx'] = trial_idx
        all_summaries.append(trial_summaries)
        trial_idx += 1
    
    if not all_summaries:
        raise ValueError(f"No summary files found for temp={temp}, style={style}")
    
    print(f"Found {len(all_summaries)} trial files")
    
    # Combine all trials
    combined_summaries = pd.concat(all_summaries, ignore_index=True)
    
    # Merge with original articles
    sft_data = train_data.merge(combined_summaries, on='document_idx', how='inner')
    
    print(f"Combined data: {len(sft_data)} training examples from {len(all_summaries)} trials")
    
    return sft_data

def create_training_pairs(sft_data: pd.DataFrame, dataset_name: str, chat_wrapper) -> List[dict]:
    """Create input-target pairs for SFT."""
    system_prompt = DATASET_SYSTEM_PROMPTS[dataset_name]
    user_prompt_template = SUMMARIZE_PROMPT_TEMPLATES[dataset_name]
    
    training_pairs = []
    
    for _, row in tqdm(sft_data.iterrows(), total=len(sft_data), desc="Creating training pairs"):
        # Format input (no style addendum - just the base prompt)
        user_message = user_prompt_template.format(article=row['article'])
        formatted_input = chat_wrapper.format_chat(
            system_prompt=system_prompt,
            user_message=user_message,
            prefiller=""
        )

        # Target is just the summary (already cleaned in generation script)
        target = row['summary_y']
        
        training_pairs.append({
            "input": formatted_input,
            "target": target,
            "document_idx": row['document_idx'],
            "trial_idx": row['trial_idx']
        })
    
    return training_pairs
