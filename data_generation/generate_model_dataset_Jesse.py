"""
Generate Dataset for New Models - Assist Tag Recognition

This script generates dataset.csv files for new models using the prompts from prompts_assist_tag.py.
It creates the same format as existing datasets in the WikiSum directory structure.

Usage:
    python generate_model_dataset_Jesse.py --model_name "anthropic_claude-3-5-sonnet" --model_type "anthropic"
    python generate_model_dataset_Jesse.py --model_name "google_gemini-2.5-pro" --model_type "google"
    python generate_model_dataset_Jesse.py --model_name "microsoft/DialoGPT-medium" --model_type "huggingface"

The script will:
1. Load articles from wikisum_0_20.csv
2. Generate summaries using the specified model
3. Save results in the appropriate directory structure
4. Create dataset.csv with columns: id, title, text, model_summary
"""

import pandas as pd
import os
import argparse
import yaml
from tqdm import tqdm
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

# Import model wrappers
from model.anthropic import load_anthropic_model, AnthropicWrapper
from model.gemini import load_gemini_model, GeminiWrapper
from model.load import load_model
from model.base import ChatTemplateWrapper

# Import prompts
from prompts.prompts_assist_tag import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

def load_wikisum_articles(data_file: str = "results_and_data/data/WikiSum/wikisum_0_20.csv") -> pd.DataFrame:
    """
    Load WikiSum articles from CSV file.
    
    Args:
        data_file: Path to the wikisum_0_20.csv file
        
    Returns:
        DataFrame with columns: id, title, text
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"WikiSum articles file not found: {data_file}")
    
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} articles from {data_file}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to read WikiSum articles from {data_file}: {e}")

def generate_summaries(
    articles_df: pd.DataFrame,
    model_wrapper: Any,
    model_name: str,
    max_articles: Optional[int] = None,
    batch_size: int = 1
) -> List[Dict[str, str]]:
    """
    Generate summaries for articles using the specified model.
    
    Args:
        articles_df: DataFrame with articles (id, title, text)
        model_wrapper: Model wrapper instance
        model_name: Name of the model for identification
        max_articles: Maximum number of articles to process (None for all)
        batch_size: Number of articles to process in each batch
        
    Returns:
        List of dictionaries with generated summaries
    """
    results = []
    
    # Limit articles if specified
    if max_articles:
        articles_df = articles_df.head(max_articles)
    
    print(f"Generating summaries for {len(articles_df)} articles using {model_name}")
    
    # Process articles in batches
    for i in tqdm(range(0, len(articles_df), batch_size), desc="Processing articles"):
        batch_articles = articles_df.iloc[i:i+batch_size]
        
        for _, article in batch_articles.iterrows():
            try:
                # Create the user prompt
                user_prompt = USER_PROMPT_TEMPLATE.format(passage=article['text'])
                
                # Format the chat conversation
                if isinstance(model_wrapper, (AnthropicWrapper, GeminiWrapper)):
                    # For API-based models
                    conversation = model_wrapper.format_chat(
                        system_prompt=SYSTEM_PROMPT_TEMPLATE,
                        user_message=user_prompt
                    )
                    
                    # Generate response
                    response = model_wrapper.generate([conversation], max_new_tokens=500, temperature=0.7)
                    summary = response[0] if response else ""
                    
                else:
                    # For HuggingFace models
                    conversation = model_wrapper.format_chat(
                        system_prompt=SYSTEM_PROMPT_TEMPLATE,
                        user_message=user_prompt
                    )
                    
                    # Generate response
                    response = model_wrapper.generate([conversation], max_new_tokens=500, temperature=0.7)
                    summary = response[0] if response else ""
                
                # Store result
                results.append({
                    'id': article['id'],
                    'title': article['title'],
                    'text': article['text'],
                    'model_summary': summary.strip()
                })
                
                print(f"✅ Generated summary for {article['id']}: {article['title'][:50]}...")
                
            except Exception as e:
                print(f"❌ Error processing article {article['id']}: {e}")
                # Add empty summary for failed articles
                results.append({
                    'id': article['id'],
                    'title': article['title'],
                    'text': article['text'],
                    'model_summary': ""
                })
                continue
    
    return results

def save_dataset(
    results: List[Dict[str, str]],
    model_name: str,
    output_dir: str = "results_and_data/data/WikiSum"
) -> str:
    """
    Save the generated dataset to CSV file.
    
    Args:
        results: List of generated summaries
        model_name: Name of the model
        output_dir: Base output directory
        
    Returns:
        Path to the saved CSV file
    """
    # Create model-specific directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(model_dir, "dataset.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\nDataset saved to: {output_file}")
    print(f"Generated {len(df)} summaries")
    print(f"Non-empty summaries: {len(df[df['model_summary'].str.strip() != ''])}")
    
    return output_file

def load_model_wrapper(model_name: str, model_type: str) -> Any:
    """
    Load the appropriate model wrapper based on model type.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ("anthropic", "google", "huggingface")
        
    Returns:
        Model wrapper instance
    """
    if model_type == "anthropic":
        return load_anthropic_model(model_name)
    elif model_type == "google":
        return load_gemini_model(model_name)
    elif model_type == "huggingface":
        return load_model(model_name, device='auto')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    """
    Main function for generating model datasets.
    """
    parser = argparse.ArgumentParser(description="Generate dataset for new models")
    parser.add_argument("--model_name", required=True, help="Name of the model")
    parser.add_argument("--model_type", required=True, choices=["anthropic", "google", "huggingface"], 
                       help="Type of model")
    parser.add_argument("--max_articles", type=int, default=None, 
                       help="Maximum number of articles to process (default: all)")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size for processing (default: 1)")
    parser.add_argument("--articles_file", default="results_and_data/data/WikiSum/wikisum_0_20.csv",
                       help="Path to articles CSV file")
    parser.add_argument("--output_dir", default="results_and_data/data/WikiSum",
                       help="Output directory for datasets")
    parser.add_argument("--testing", action="store_true", 
                       help="Testing mode - process only first 3 articles")
    
    args = parser.parse_args()
    
    print("=== Generate Model Dataset for Assist Tag Recognition ===")
    print(f"Model: {args.model_name}")
    print(f"Type: {args.model_type}")
    print(f"Articles file: {args.articles_file}")
    print(f"Output directory: {args.output_dir}")
    
    if args.testing:
        print("🧪 TESTING MODE: Processing only first 3 articles")
        args.max_articles = 3
    
    try:
        # Load articles
        print(f"\nLoading articles from {args.articles_file}...")
        articles_df = load_wikisum_articles(args.articles_file)
        
        # Load model
        print(f"\nLoading model {args.model_name} ({args.model_type})...")
        model_wrapper = load_model_wrapper(args.model_name, args.model_type)
        
        # Generate summaries
        print(f"\nGenerating summaries...")
        results = generate_summaries(
            articles_df=articles_df,
            model_wrapper=model_wrapper,
            model_name=args.model_name,
            max_articles=args.max_articles,
            batch_size=args.batch_size
        )
        
        # Save dataset
        print(f"\nSaving dataset...")
        output_file = save_dataset(
            results=results,
            model_name=args.model_name,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Successfully generated dataset for {args.model_name}")
        print(f"📁 Output file: {output_file}")
        
        # Show sample results
        if results:
            print(f"\n📋 Sample results:")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result['id']}: {result['title'][:50]}...")
                print(f"     Summary: {result['model_summary'][:100]}...")
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
