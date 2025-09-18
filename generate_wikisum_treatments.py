"""
WikiSum Treatment File Generator

This script generates missing treatment CSV files for the WikiSum dataset.
It creates both capitalization and typo treatment files in the same format
as the existing ones.

USAGE:
- IDE mode: python generate_wikisum_treatments.py
- CLI mode: python generate_wikisum_treatments.py --base-dir path/to/WikiSum --models model1,model2

The script will:
1. Find all model directories in the base WikiSum directory
2. For each directory, check if treatment files are missing
3. Generate capitalization and typo treatment files using string_modifier.py
4. Save the files in the same format as existing treatment files
"""

import os
import pandas as pd
import argparse
import sys
from pathlib import Path
from data_generation.string_modifier import randomly_capitalize_string, introduce_typos_per_word

def find_model_directories(base_dir: str) -> list:
    """
    Find all model directories in the WikiSum base directory.
    
    Args:
        base_dir: Path to the WikiSum directory
        
    Returns:
        List of model directory paths
    """
    model_dirs = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return model_dirs
    
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it's a model directory (has dataset.csv)
            dataset_file = item / "dataset.csv"
            if dataset_file.exists():
                model_dirs.append(str(item))
    
    return sorted(model_dirs)

def check_missing_treatments(model_dir: str) -> dict:
    """
    Check which treatment files are missing for a model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Dictionary with missing treatment types
    """
    model_path = Path(model_dir)
    missing = {}
    
    # Check for capitalization treatment file
    cap_file = model_path / "dataset_capitalization_rates_injected.csv"
    missing['capitalization'] = not cap_file.exists()
    
    # Check for typo treatment file
    typo_file = model_path / "dataset_typo_rates_injected.csv"
    missing['typo'] = not typo_file.exists()
    
    return missing

def load_base_dataset(model_dir: str) -> pd.DataFrame:
    """
    Load the base dataset.csv file for a model.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        DataFrame with the base dataset
    """
    dataset_file = os.path.join(model_dir, "dataset.csv")
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Base dataset file not found: {dataset_file}")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(dataset_file, encoding=encoding)
                print(f"  ✅ Loaded {len(df)} rows from {os.path.basename(dataset_file)} (encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not decode file with any of the tried encodings: {encodings}")
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_file}: {e}")

def generate_capitalization_treatments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate capitalization treatment columns for the dataset.
    
    Args:
        df: Base dataset DataFrame
        
    Returns:
        DataFrame with added capitalization treatment columns
    """
    print("  🔤 Generating capitalization treatments...")
    
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Generate different capitalization treatments with S1-S4 strength levels
    # S1: 25% capitalization (light)
    result_df['S1'] = result_df['model_summary'].apply(
        lambda x: randomly_capitalize_string(str(x), 25) if pd.notna(x) else x
    )
    
    # S2: 50% capitalization (medium)
    result_df['S2'] = result_df['model_summary'].apply(
        lambda x: randomly_capitalize_string(str(x), 50) if pd.notna(x) else x
    )
    
    # S3: 75% capitalization (heavy)
    result_df['S3'] = result_df['model_summary'].apply(
        lambda x: randomly_capitalize_string(str(x), 75) if pd.notna(x) else x
    )
    
    # S4: 100% capitalization (all caps)
    result_df['S4'] = result_df['model_summary'].apply(
        lambda x: str(x).upper() if pd.notna(x) else x
    )
    
    print(f"  ✅ Generated capitalization treatments for {len(result_df)} rows")
    return result_df

def generate_typo_treatments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate typo treatment columns for the dataset.
    
    Args:
        df: Base dataset DataFrame
        
    Returns:
        DataFrame with added typo treatment columns
    """
    print("  🔤 Generating typo treatments...")
    
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Generate different typo treatments with S1-S4 strength levels
    # S1: Light typos (0.1 typos per word) - occasional typos
    result_df['S1'] = result_df['model_summary'].apply(
        lambda x: introduce_typos_per_word(str(x), 0.1) if pd.notna(x) else x
    )
    
    # S2: Medium typos (0.3 typos per word) - moderate typos
    result_df['S2'] = result_df['model_summary'].apply(
        lambda x: introduce_typos_per_word(str(x), 0.3) if pd.notna(x) else x
    )
    
    # S3: Heavy typos (0.6 typos per word) - frequent typos
    result_df['S3'] = result_df['model_summary'].apply(
        lambda x: introduce_typos_per_word(str(x), 0.6) if pd.notna(x) else x
    )
    
    # S4: Major typos (1.2 typos per word) - severe typos
    result_df['S4'] = result_df['model_summary'].apply(
        lambda x: introduce_typos_per_word(str(x), 1.2) if pd.notna(x) else x
    )
    
    print(f"  ✅ Generated typo treatments for {len(result_df)} rows")
    return result_df

def save_treatment_file(df: pd.DataFrame, output_path: str, treatment_type: str) -> None:
    """
    Save the treatment DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where to save the file
        treatment_type: Type of treatment (for logging)
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"  ✅ Saved {treatment_type} treatment file: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"  ❌ Failed to save {treatment_type} treatment file: {e}")

def process_model_directory(model_dir: str, force_regenerate: bool = False) -> None:
    """
    Process a single model directory to generate missing treatment files.
    
    Args:
        model_dir: Path to the model directory
        force_regenerate: Whether to regenerate existing files
    """
    model_name = os.path.basename(model_dir)
    print(f"\n📁 Processing model: {model_name}")
    
    # Check what treatments are missing
    missing = check_missing_treatments(model_dir)
    
    if not any(missing.values()):
        print(f"  ✅ All treatment files already exist for {model_name}")
        if not force_regenerate:
            return
        else:
            print(f"  🔄 Force regenerating all treatment files...")
    
    # Load the base dataset
    try:
        df = load_base_dataset(model_dir)
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        return
    
    # Generate capitalization treatments if missing or force regenerate
    if missing.get('capitalization', False) or force_regenerate:
        cap_df = generate_capitalization_treatments(df)
        cap_output_path = os.path.join(model_dir, "dataset_capitalization_rates_injected.csv")
        save_treatment_file(cap_df, cap_output_path, "capitalization")
    
    # Generate typo treatments if missing or force regenerate
    if missing.get('typo', False) or force_regenerate:
        typo_df = generate_typo_treatments(df)
        typo_output_path = os.path.join(model_dir, "dataset_typo_rates_injected.csv")
        save_treatment_file(typo_df, typo_output_path, "typo")

def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Generate missing WikiSum treatment files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (IDE mode)
  python generate_wikisum_treatments.py
  
  # Run with custom base directory
  python generate_wikisum_treatments.py --base-dir results_and_data/data/WikiSum
  
  # Process specific models only
  python generate_wikisum_treatments.py --models gpt-4.1-2025-04-14,gpt-4.1-mini-2025-04-14
  
  # Force regenerate all treatment files
  python generate_wikisum_treatments.py --force-regenerate
        """
    )
    
    parser.add_argument("--base-dir", 
                       default="results_and_data/data/WikiSum",
                       help="Path to the WikiSum base directory (default: results_and_data/data/WikiSum)")
    parser.add_argument("--models",
                       help="Comma-separated list of specific models to process (default: all models)")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regenerate all treatment files, even if they already exist")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually generating files")
    
    return parser.parse_args()

def main():
    """
    Main function - supports both CLI and IDE usage modes
    """
    print("=== WikiSum Treatment File Generator ===")
    print("This script generates missing capitalization and typo treatment files")
    
    # ===== DETERMINE CONFIGURATION =====
    # Check if running from CLI (has command line arguments) or IDE (no arguments)
    args = None
    if len(sys.argv) > 1:
        # CLI mode: parse arguments
        args = parse_arguments()
        base_dir = args.base_dir
        selected_models = args.models.split(',') if args.models else None
        force_regenerate = args.force_regenerate
        dry_run = args.dry_run
    else:
        # IDE mode: use hardcoded settings
        base_dir = "results_and_data/data/WikiSum"
        selected_models = None
        force_regenerate = False
        dry_run = False
    
    print(f"📁 Base directory: {base_dir}")
    print(f"🎯 Selected models: {selected_models if selected_models else 'All models'}")
    print(f"🔄 Force regenerate: {force_regenerate}")
    print(f"🔍 Dry run: {dry_run}")
    
    # ===== FIND MODEL DIRECTORIES =====
    model_dirs = find_model_directories(base_dir)
    
    if not model_dirs:
        print(f"❌ No model directories found in {base_dir}")
        return
    
    print(f"\n📊 Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"  - {os.path.basename(model_dir)}")
    
    # Filter by selected models if specified
    if selected_models:
        filtered_dirs = []
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            if model_name in selected_models:
                filtered_dirs.append(model_dir)
            else:
                print(f"  ⏭️  Skipping {model_name} (not in selected models)")
        
        model_dirs = filtered_dirs
        print(f"\n🎯 Processing {len(model_dirs)} selected model directories")
    
    if not model_dirs:
        print("❌ No model directories match the selected models")
        return
    
    # ===== PROCESS EACH MODEL DIRECTORY =====
    if dry_run:
        print(f"\n🔍 DRY RUN - Would process the following:")
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            missing = check_missing_treatments(model_dir)
            print(f"  📁 {model_name}:")
            if missing.get('capitalization', False):
                print(f"    - Would generate: dataset_capitalization_rates_injected.csv")
            if missing.get('typo', False):
                print(f"    - Would generate: dataset_typo_rates_injected.csv")
            if not any(missing.values()):
                print(f"    - All files already exist")
        return
    
    # Process each model directory
    success_count = 0
    error_count = 0
    
    for model_dir in model_dirs:
        try:
            process_model_directory(model_dir, force_regenerate)
            success_count += 1
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(model_dir)}: {e}")
            error_count += 1
    
    # ===== SUMMARY =====
    print(f"\n📊 PROCESSING COMPLETE!")
    print(f"  ✅ Successfully processed: {success_count} model directories")
    if error_count > 0:
        print(f"  ❌ Errors encountered: {error_count} model directories")
    
    print(f"\n💡 Treatment files generated:")
    print(f"  - dataset_capitalization_rates_injected.csv (with S1, S2, S3, S4 columns)")
    print(f"  - dataset_typo_rates_injected.csv (with S1, S2, S3, S4 columns)")
    print(f"    S1: Light treatment (25% caps / 0.1 typos per word)")
    print(f"    S2: Medium treatment (50% caps / 0.3 typos per word)")
    print(f"    S3: Heavy treatment (75% caps / 0.6 typos per word)")
    print(f"    S4: Major treatment (100% caps / 1.2 typos per word)")

if __name__ == "__main__":
    main()
