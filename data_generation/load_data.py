import pandas as pd
import os
from typing import List, Tuple, Optional

def load_dataset(dataset_name, splits: List[str] = ["train", "test", "validation"], datasets_dir: Optional[str] = None) -> Tuple[pd.DataFrame | None, ...]:
    """
    Load dataset from HuggingFace, save locally if not exists, return train/test/validation dataframes
    
    Args:
        dataset_name: str, name of HuggingFace dataset ("cnn_dailymail", "xsum")
        
    Returns:
        tuple: (train_data, test_data, validation_data) as pandas DataFrames
               Each with columns [document_idx, article, summary]
    """
    datasets_dir = datasets_dir or "results_and_data/data"
    base_path = os.path.join(datasets_dir, dataset_name)
    
    # Check if all splits already exist locally
    all_exist = all(os.path.exists(f"{base_path}/{split}.csv") for split in splits)
    
    if all_exist:
        # Load existing CSVs
        train_data = pd.read_csv(f"{base_path}/train.csv") if 'train' in splits else None
        test_data = pd.read_csv(f"{base_path}/test.csv") if 'test' in splits else None
        validation_data = pd.read_csv(f"{base_path}/validation.csv") if 'validation' in splits else None
        return train_data, test_data, validation_data
    
    # Need to download and process
    os.makedirs(base_path, exist_ok=True)

    from datasets import load_dataset as hf_load_dataset
    
    # Load from HuggingFace
    if dataset_name == "cnn_dailymail":
        hf_dataset = hf_load_dataset("cnn_dailymail", "3.0.0")
        article_col = "article"
        summary_col = "highlights"
    elif dataset_name == "xsum":
        hf_dataset = hf_load_dataset("EdinburghNLP/xsum")
        article_col = "document" 
        summary_col = "summary"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Process each split
    processed_splits = {}
    for split in ["train", "test", "validation"]:
        if split in splits and not os.path.exists(f"{base_path}/{split}.csv"):
            split_data = hf_dataset[split]
            
            # Create standardized dataframe
            df = pd.DataFrame({
                'document_idx': range(len(split_data)),
                'article': split_data[article_col],
                'summary': split_data[summary_col]
            })
            
            # Save locally
            df.to_csv(f"{base_path}/{split}.csv", index=False)
            processed_splits[split] = df
        else:
            processed_splits[split] = None
    
    return processed_splits["train"], processed_splits["test"], processed_splits["validation"]


def load_model_summaries(run_name, sub_dataset_name, temperature, trial_idx, style, results_dir = None):
    """
    Load model summaries from experiment results
    
    Args:
        run_name: str, experiment run identifier
        sub_dataset_name: str, dataset split ("train", "test", "validation") 
        temperature: float, temperature used for generation
        trial_idx: int, trial number for this temperature
        styles: style that summary was prompted with, key to prompts.STYLE_SYSTEM_PROMPTS
        
    Returns:
        pandas.DataFrame with columns [document_idx, summary]
    """
    results_dir = results_dir or 'results_and_data/results'
    file_dir = f"{results_dir}/main/{run_name}/{sub_dataset_name}/model_summaries"
    file_name = f"T{temperature}_trial{trial_idx}_style{style}.csv"
    file_path = os.path.join(file_dir, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model summary file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Convert literal \n back to actual newlines
    df['summary'] = df['summary'].str.replace('\\n', '\n')
    return df


if __name__ == "__main__":
    # Test the functions
    print("Loading CNN/DailyMail dataset...")
    train, test, val = load_dataset("cnn_dailymail")
    print(f"CNN/DailyMail - Train: {len(train)}, Test: {len(test)}, Val: {len(val)}")
    print(f"Columns: {train.columns.tolist()}")
    print(f"Sample: {train.iloc[0]['article'][:100]}...")
