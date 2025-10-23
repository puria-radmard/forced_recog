"""
Unified Dataset Generation Script

This script consolidates the functionality of the three previous dataset generation scripts:
- run_dataset_generation_Jesse.py
- quick_test_dataset_Jesse.py  
- ide_dataset_generator_Jesse.py

It generates dataset.csv files for models using the assist tag recognition prompts.
Supports both CLI and IDE usage with configuration from generate_dataset_config.yaml.

Features:
- Model type inference from model name
- Support for "max" tokens (uses model's maximum)
- Configurable via YAML
- IDE-friendly with hardcoded fallback config
- Batch processing for multiple models
- Testing mode support

Usage:
    # CLI usage
    python generate_dataset.py --model anthropic_claude-3-5-sonnet
    python generate_dataset.py --model google_gemini-2.5-flash --testing
    python generate_dataset.py --all
    
    # IDE usage - modify CONFIG at top of file and run directly
"""

import pandas as pd
import os
import argparse
import yaml
import sys
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
from dotenv import load_dotenv

# Import model wrappers
from model.anthropic import load_anthropic_model, AnthropicWrapper
from model.gemini import load_gemini_model, GeminiWrapper
from model.openai import load_openai_model, OpenAIWrapper
from model.load import load_model
from model.base import ChatTemplateWrapper

# Import prompts
from prompts.prompts_assist_tag import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

# Load environment variables
load_dotenv()

# ===== IDE CONFIGURATION =====
# IDE mode uses the same config file as CLI mode
# To change settings for IDE testing, modify configs/data/generate_dataset_config.yaml
# Or create a custom config file and modify the IDE_CONFIG_FILE path below

IDE_CONFIG_FILE = "configs/data/generate_dataset_config.yaml"
IDE_MODEL_KEY = "google_gemini-2.5-flash"  # Model key from config to use in IDE mode
IDE_TESTING_MODE = True  # Set to True for testing (limited articles)

# ===== END IDE CONFIGURATION =====

def infer_model_type(model_name: str) -> str:
    """Infer model type from model name."""
    model_name_lower = model_name.lower()
    
    if any(prefix in model_name_lower for prefix in ["claude", "anthropic"]):
        return "anthropic"
    elif any(prefix in model_name_lower for prefix in ["gemini", "google"]):
        return "google"
    elif any(prefix in model_name_lower for prefix in ["gpt", "openai"]):
        return "openai"
    elif "/" in model_name or any(prefix in model_name_lower for prefix in ["llama", "dialogpt", "microsoft", "meta"]):
        return "huggingface"
    else:
        # Default fallback - try to infer from common patterns
        if model_name.startswith("claude"):
            return "anthropic"
        elif model_name.startswith("gemini"):
            return "google"
        elif model_name.startswith("gpt"):
            return "openai"
        else:
            return "huggingface"

def get_max_tokens(model_wrapper: Any, max_tokens_config: Union[str, int]) -> int:
    """Get the actual max tokens value, handling 'max' option."""
    if max_tokens_config == "max":
        # Return a large number - models will use their own limits
        return 8192  # Conservative large number
    else:
        return int(max_tokens_config)

def setup_test_environment(testing: bool, output_dir: str) -> str:
    """Setup test environment if in testing mode."""
    if not testing:
        return output_dir
    
    # Create test directory in project root
    test_dir = "test_data"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    print(f"[TEST] Using test directory: {test_dir}")
    return test_dir

def load_config(config_file: str = "configs/data/generate_dataset_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        raise FileNotFoundError(f"Config file not found: {config_file}")

def load_articles(articles_file: str, max_articles: Optional[int] = None) -> pd.DataFrame:
    """Load articles from CSV file."""
    print(f"[INFO] Loading articles from {articles_file}...")
    
    if not os.path.exists(articles_file):
        raise FileNotFoundError(f"Articles file not found: {articles_file}")
    
    df = pd.read_csv(articles_file)
    print(f"[OK] Loaded {len(df)} articles")
    
    if max_articles:
        df = df.head(max_articles)
        print(f"[INFO] Processing {len(df)} articles (limited by max_articles)")
    
    return df

def load_model_wrapper(model_name: str, model_type: Optional[str] = None) -> Any:
    """Load the specified model wrapper."""
    if model_type is None:
        model_type = infer_model_type(model_name)
    
    print(f"[INFO] Loading {model_type} model: {model_name}")
    
    if model_type == "anthropic":
        return load_anthropic_model(model_name)
    elif model_type == "google":
        return load_gemini_model(model_name)
    elif model_type == "openai":
        return load_openai_model(model_name)
    elif model_type == "huggingface":
        return load_model(model_name, device='auto')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def generate_summaries(
    articles_df: pd.DataFrame,
    model_wrapper: Any,
    max_new_tokens: Union[str, int],
    temperature: float,
    show_detailed_output: bool = True
) -> List[Dict[str, str]]:
    """Generate summaries for articles."""
    print(f"\n[INFO] Generating summaries for {len(articles_df)} articles...")
    
    results = []
    actual_max_tokens = get_max_tokens(model_wrapper, max_new_tokens)
    
    for i, (_, article) in enumerate(tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing")):
        if show_detailed_output:
            print(f"\n[INFO] Processing article {i+1}/{len(articles_df)}: {article['title'][:50]}...")
        
        try:
            # Create user prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(passage=article['text'])
            
            # Format conversation and generate
            if isinstance(model_wrapper, (AnthropicWrapper, GeminiWrapper, OpenAIWrapper)):
                conversation = model_wrapper.format_chat(
                    system_prompt=SYSTEM_PROMPT_TEMPLATE,
                    user_message=user_prompt
                )
                response = model_wrapper.generate(
                    [conversation], 
                    max_new_tokens=actual_max_tokens,
                    temperature=temperature
                )
                summary = response[0] if response else ""
            else:
                # HuggingFace models
                conversation = model_wrapper.format_chat(
                    system_prompt=SYSTEM_PROMPT_TEMPLATE,
                    user_message=user_prompt
                )
                response = model_wrapper.generate(
                    [conversation], 
                    max_new_tokens=actual_max_tokens,
                    temperature=temperature
                )
                summary = response[0] if response else ""
            
            # Store result
            result = {
                'id': article['id'],
                'title': article['title'],
                'text': article['text'],
                'model_summary': summary.strip()
            }
            results.append(result)
            
            if show_detailed_output:
                print(f"[OK] Generated summary: {summary[:100]}...")
            
        except Exception as e:
            print(f"[ERROR] Error processing article {article['id']}: {e}")
            # Add empty summary for failed articles
            results.append({
                'id': article['id'],
                'title': article['title'],
                'text': article['text'],
                'model_summary': ""
            })
    
    return results

def save_dataset(results: List[Dict[str, str]], model_name: str, output_dir: str) -> str:
    """Save the generated dataset."""
    print(f"\n[INFO] Saving dataset...")
    
    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    output_file = os.path.join(model_dir, "dataset.csv")
    df.to_csv(output_file, index=False)
    
    print(f"[OK] Dataset saved to: {output_file}")
    print(f"[STATS] Generated {len(df)} summaries")
    print(f"[STATS] Non-empty summaries: {len(df[df['model_summary'].str.strip() != ''])}")
    
    return output_file

def show_results(results: List[Dict[str, str]], show_sample: bool = True, max_samples: int = 3):
    """Show sample results."""
    if not show_sample:
        return
    
    print(f"\n[RESULTS] Sample Results:")
    for i, result in enumerate(results[:max_samples]):
        print(f"  {i+1}. {result['id']}: {result['title'][:50]}...")
        print(f"     Summary: {result['model_summary'][:150]}...")
        print()

def run_single_model(
    model_key: str,
    config: Dict[str, Any],
    testing: bool = False
) -> bool:
    """Run dataset generation for a single model."""
    print(f"\n{'='*60}")
    print(f"Processing model: {model_key}")
    print(f"{'='*60}")
    
    try:
        # Get model configuration
        if model_key in config['models']:
            model_config = config['models'][model_key]
            model_actual_name = model_config['model_name']
        else:
            # IDE mode fallback
            model_actual_name = model_key
        
        # Get settings
        default_config = config['default']
        if testing:
            max_articles = config.get('testing', {}).get('max_articles', 3)
            show_detailed = config.get('testing', {}).get('show_detailed_output', True)
        else:
            max_articles = model_config.get('max_articles', default_config['max_articles'])
            show_detailed = default_config.get('show_detailed_output', True)
        
        max_new_tokens = model_config.get('max_new_tokens', default_config['max_new_tokens'])
        temperature = model_config.get('temperature', default_config['temperature'])
        articles_file = default_config['articles_file']
        output_dir = default_config['output_dir']
        
        # Setup test environment if needed
        output_dir = setup_test_environment(testing, output_dir)
        
        print(f"Model: {model_actual_name}")
        print(f"Type: {infer_model_type(model_actual_name)}")
        print(f"Max articles: {max_articles or 'All'}")
        print(f"Max tokens: {max_new_tokens}")
        print(f"Temperature: {temperature}")
        
        # Load articles
        articles_df = load_articles(articles_file, max_articles)
        
        # Load model
        model_wrapper = load_model_wrapper(model_actual_name)
        print("[OK] Model loaded successfully")
        
        # Generate summaries
        results = generate_summaries(
            articles_df, 
            model_wrapper, 
            max_new_tokens, 
            temperature,
            show_detailed
        )
        
        # Save dataset
        output_file = save_dataset(results, model_key, output_dir)
        
        # Show results
        show_results(results, default_config.get('show_sample_results', True))
        
        print(f"[SUCCESS] Successfully generated dataset for {model_key}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error generating dataset for {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dataset for models using assist tag recognition prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for specific model
    python generate_dataset.py --model anthropic_claude-3-5-sonnet
    
    # Generate for specific model in testing mode
    python generate_dataset.py --model google_gemini-2.5-flash --testing
    
    # Generate for all models
    python generate_dataset.py --all
    
    # Generate for all models in testing mode
    python generate_dataset.py --all --testing
    
    # Use custom config file
    python generate_dataset.py --model anthropic_claude-3-5-sonnet --config configs/custom_config.yaml
        """
    )
    
    parser.add_argument("--model", help="Specific model key to generate dataset for")
    parser.add_argument("--all", action="store_true", help="Generate datasets for all models")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode (limited articles)")
    parser.add_argument("--config", default="configs/data/generate_dataset_config.yaml", 
                       help="Configuration file path")
    
    return parser.parse_args()

def main():
    """Main function."""
    print("=== Unified Dataset Generation Script ===")
    print("Generating datasets for assist tag recognition experiments")
    
    args = parse_arguments()
    
    # Check if running in IDE mode (no arguments provided)
    if len(sys.argv) == 1:
        print(f"[IDE] IDE Mode: Using config file {IDE_CONFIG_FILE}")
        try:
            config = load_config(IDE_CONFIG_FILE)
            success = run_single_model(IDE_MODEL_KEY, config, testing=IDE_TESTING_MODE)
            if success:
                print(f"\n[SUCCESS] IDE dataset generation completed successfully!")
            else:
                print(f"\n[ERROR] IDE dataset generation failed!")
                return 1
            return 0
        except Exception as e:
            print(f"[ERROR] IDE mode failed: {e}")
            return 1
    
    # CLI mode
    if not args.model and not args.all:
        print("[ERROR] Please specify --model or --all")
        return 1
    
    # Load configuration
    config = load_config(args.config)
    
    if args.all:
        # Generate for all models
        print("[INFO] Generating datasets for all models...")
        success_count = 0
        total_count = len(config['models'])
        
        for model_key in config['models']:
            if run_single_model(model_key, config, args.testing):
                success_count += 1
        
        print(f"\n[STATS] Summary: {success_count}/{total_count} models processed successfully")
        
    else:
        # Generate for specific model
        print(f"[INFO] Generating dataset for model: {args.model}")
        success = run_single_model(args.model, config, args.testing)
        
        if success:
            print(f"[SUCCESS] Dataset generation completed successfully for {args.model}")
        else:
            print(f"[ERROR] Dataset generation failed for {args.model}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
