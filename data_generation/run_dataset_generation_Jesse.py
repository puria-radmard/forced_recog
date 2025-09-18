"""
Run Dataset Generation with Configuration

This script runs the dataset generation using configuration from generate_dataset_config.yaml.
It provides an easy way to generate datasets for multiple models or run in testing mode.

Usage:
    python run_dataset_generation_Jesse.py --model anthropic_claude-3-5-sonnet
    python run_dataset_generation_Jesse.py --model google_gemini-2.5-flash --testing
    python run_dataset_generation_Jesse.py --all  # Generate for all models
"""

import argparse
import yaml
import subprocess
import sys
from typing import Dict, Any

def load_config(config_file: str = "generate_dataset_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)

def run_model_generation(model_name: str, config: Dict[str, Any], testing: bool = False) -> bool:
    """Run dataset generation for a specific model."""
    if model_name not in config['models']:
        print(f"❌ Model {model_name} not found in configuration")
        return False
    
    model_config = config['models'][model_name]
    default_config = config['default']
    
    # Build command
    cmd = [
        "python", "generate_model_dataset_Jesse.py",
        "--model_name", model_config['model_name'],
        "--model_type", model_config['type'],
        "--articles_file", default_config['articles_file'],
        "--output_dir", default_config['output_dir']
    ]
    
    # Add optional parameters
    if testing:
        cmd.extend(["--testing"])
        max_articles = config['testing']['max_articles']
    else:
        max_articles = model_config.get('max_articles', default_config['max_articles'])
    
    if max_articles:
        cmd.extend(["--max_articles", str(max_articles)])
    
    batch_size = model_config.get('batch_size', default_config['batch_size'])
    if batch_size:
        cmd.extend(["--batch_size", str(batch_size)])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Successfully generated dataset for {model_name}")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating dataset for {model_name}: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run dataset generation with configuration")
    parser.add_argument("--model", help="Specific model to generate dataset for")
    parser.add_argument("--all", action="store_true", help="Generate datasets for all models")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode")
    parser.add_argument("--config", default="generate_dataset_config.yaml", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    if not args.model and not args.all:
        print("❌ Please specify --model or --all")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    if args.all:
        # Generate for all models
        print("🔄 Generating datasets for all models...")
        success_count = 0
        total_count = len(config['models'])
        
        for model_name in config['models']:
            print(f"\n{'='*50}")
            print(f"Processing model: {model_name}")
            print(f"{'='*50}")
            
            if run_model_generation(model_name, config, args.testing):
                success_count += 1
        
        print(f"\n📊 Summary: {success_count}/{total_count} models processed successfully")
        
    else:
        # Generate for specific model
        print(f"🔄 Generating dataset for model: {args.model}")
        success = run_model_generation(args.model, config, args.testing)
        
        if success:
            print(f"✅ Dataset generation completed successfully for {args.model}")
        else:
            print(f"❌ Dataset generation failed for {args.model}")
            sys.exit(1)

if __name__ == "__main__":
    main()


