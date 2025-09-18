"""
WikiSum Data Formatter for Assist Tag Recognition

DUAL MODE OPERATION:
1. IDE MODE (default): Run without arguments for easy debugging
   - Uses hardcoded config path: configs/data/wikisum_data_formatter_config.yaml
   - Easy to modify and debug in IDE
   - All parameters configured through YAML file

2. CLI MODE: Run with arguments for production use
   - Specify custom config file with --config argument
   - Supports different configurations for different experiments
   - Operation modes: show_available, assist_tag_experiments, per_model, all_treatments

USAGE:
- IDE mode: python wikisum_data_formatter.py
- CLI mode: python wikisum_data_formatter.py --config path/to/config.yaml
- Show models: python wikisum_data_formatter.py --show-models --config path/to/config.yaml

CONFIGURATION:
- Data paths and model selection
- Treatment filtering and operation modes
- Output settings and display options

FEATURES:
- YAML-based configuration for easy parameter management
- Support for multiple operation modes
- Granular model selection by specific model names
- Flexible treatment filtering
- Detailed progress reporting and results summary
- Dual-mode operation for both IDE debugging and CLI production use

The WikiSum directory structure:
- results_and_data/data/WikiSum/
  - anthropic_claude-3-5-haiku-20241022/
    - dataset.csv (control dataset)
    - dataset_capitalization_rates_injected.csv (capitalization treatment)
    - dataset_typo_rates_injected.csv (typo treatment)
  - [other model directories with same structure]

Assist Tag Recognition Experiment Structure:
- results_and_data/experiments/WikiSum/
  - model1_vs_model2_control_comparison/
    - control.csv (model1 control responses)
    - treatment.csv (model2 control responses)
  - [other experiment directories]
"""

import pandas as pd
import os
import yaml
import argparse
import sys
from typing import List, Dict, Any, Optional, Set


def load_config(config_path: str = "configs/data/wikisum_data_formatter_config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Set default values for optional parameters
        defaults = {
            "selected_models": ["all"],
            "require_all_treatments": False,
            "treatment": None,
            "treatment_level": None,
            "operation_mode": "assist_tag_experiments",
            "show_available_treatments": True,
            "show_model_summary": True,
            "show_progress": True,
            "create_directories": True,
            "overwrite_existing": False,
            "verbose": True
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        return config
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")


def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="WikiSum Data Formatter for Assist Tag Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config file
  python wikisum_data_formatter.py
  
  # Run with custom config file
  python wikisum_data_formatter.py --config configs/custom_config.yaml
  
  # Show available models and treatments
  python wikisum_data_formatter.py --show-models --config configs/data/wikisum_data_formatter_config.yaml
        """
    )
    
    parser.add_argument("--config", 
                       default="configs/data/wikisum_data_formatter_config.yaml",
                       help="Path to the YAML configuration file (default: configs/data/wikisum_data_formatter_config.yaml)")
    parser.add_argument("--show-models", action="store_true",
                       help="Show available models and treatments and exit")
    
    return parser.parse_args()


def detect_available_treatments(base_path: str = "results_and_data/data/WikiSum") -> Dict[str, Set[str]]:
    """
    Detect which treatments are available for each model.
    
    Args:
        base_path: Base path to the WikiSum data directory
        
    Returns:
        Dictionary mapping model names to sets of available treatments
    """
    treatment_files = {
        'control': 'dataset.csv',
        'capitalization': 'dataset_capitalization_rates_injected.csv',
        'typo': 'dataset_typo_rates_injected.csv'
    }
    
    available_treatments = {}
    
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return available_treatments
    
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for model_dir in model_dirs:
        model_name = model_dir
        available = set()
        
        for treatment, filename in treatment_files.items():
            dataset_path = os.path.join(base_path, model_dir, filename)
            if os.path.exists(dataset_path):
                available.add(treatment)
        
        if available:  # Only include models that have at least one treatment
            available_treatments[model_name] = available
    
    return available_treatments

def filter_models_by_treatments(available_treatments: Dict[str, Set[str]], 
                               required_treatments: List[str]) -> List[str]:
    """
    Filter models to only include those that have all required treatments.
    
    Args:
        available_treatments: Dictionary mapping model names to available treatments
        required_treatments: List of treatments that must be available
        
    Returns:
        List of model names that have all required treatments
    """
    required_set = set(required_treatments)
    filtered_models = []
    
    for model_name, available in available_treatments.items():
        if required_set.issubset(available):
            filtered_models.append(model_name)
    
    return filtered_models

def load_wikisum_data(base_path: str = "results_and_data/data/WikiSum", 
                     treatment: str = "control",
                     treatment_level: str = None,
                     models: Optional[List[str]] = None,
                     require_all_treatments: bool = False) -> pd.DataFrame:
    """
    Load WikiSum data from multiple model directories and combine into a single DataFrame.
    
    Args:
        base_path: Base path to the WikiSum data directory
        treatment: Which treatment to load ('control', 'capitalization', 'typo')
        treatment_level: For capitalization: 'S2' or 'S4'
                        For typo: 'S2' or 'S4'
                        For control: None (ignored)
        models: Optional list of specific model names to include. If None, includes all available models.
        require_all_treatments: If True, only include models that have all treatments available.
        
    Returns:
        Combined DataFrame with columns: trial, model, treatment, passage, response
    """
    all_data = []
    
    # Map treatment names to filenames
    treatment_files = {
        'control': 'dataset.csv',
        'capitalization': 'dataset_capitalization_rates_injected.csv',
        'typo': 'dataset_typo_rates_injected.csv'
    }
    
    if treatment not in treatment_files:
        raise ValueError(f"Treatment must be one of: {list(treatment_files.keys())}")
    
    filename = treatment_files[treatment]
    
    # Validate treatment level for non-control treatments
    if treatment == 'capitalization':
        if treatment_level not in ['S2', 'S4']:
            raise ValueError(f"For capitalization treatment, treatment_level must be 'S2' or 'S4', got: {treatment_level}")
    elif treatment == 'typo':
        if treatment_level not in ['S2', 'S4']:
            raise ValueError(f"For typo treatment, treatment_level must be 'S2' or 'S4', got: {treatment_level}")
    elif treatment == 'control' and treatment_level is not None:
        print("Warning: treatment_level is ignored for control treatment")
    
    # Get all model directories
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return pd.DataFrame(columns=['trial', 'model', 'treatment', 'passage', 'response'])
    
    all_model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    if not all_model_dirs:
        print(f"No model directories found in {base_path}")
        return pd.DataFrame(columns=['trial', 'model', 'treatment', 'passage', 'response'])
    
    # Filter models based on requirements
    if models is not None:
        # Use only specified models
        model_dirs = [d for d in all_model_dirs if d in models]
        if not model_dirs:
            print(f"None of the specified models found in {base_path}")
            print(f"Available models: {all_model_dirs}")
            print(f"Specified models: {models}")
            return pd.DataFrame(columns=['trial', 'model', 'treatment', 'passage', 'response'])
    elif require_all_treatments:
        # Only include models that have all treatments
        available_treatments = detect_available_treatments(base_path)
        all_treatments = {'control', 'capitalization', 'typo'}
        model_dirs = filter_models_by_treatments(available_treatments, list(all_treatments))
        if not model_dirs:
            print(f"No models found with all treatments available")
            print(f"Available treatments per model:")
            for model, treatments in available_treatments.items():
                print(f"  {model}: {sorted(treatments)}")
            return pd.DataFrame(columns=['trial', 'model', 'treatment', 'passage', 'response'])
    else:
        # Use all available models
        model_dirs = all_model_dirs
    
    print(f"Loading {treatment} data from {len(model_dirs)} models...")
    if models is not None:
        print(f"Using specified models: {model_dirs}")
    elif require_all_treatments:
        print(f"Using models with all treatments: {model_dirs}")
    
    for model_dir in model_dirs:
        model_name = model_dir
        dataset_path = os.path.join(base_path, model_dir, filename)
        
        if os.path.exists(dataset_path):
            print(f"  Loading {model_name}...")
            try:
                # Try different encodings to handle various file formats
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(dataset_path, encoding=encoding)
                        print(f"    Successfully loaded with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError(f"Could not decode file with any of the tried encodings: {encodings_to_try}")
                
                # Transform to the required format
                df['model'] = model_name
                df['treatment'] = treatment
                df['passage'] = df['text']  # Use text as passage
                
                # Set response based on treatment type
                if treatment == 'control':
                    df['response'] = df['model_summary']  # Use model_summary as response
                else:
                    # For treatment datasets, use the specified treatment level column as response
                    if treatment_level not in df.columns:
                        raise ValueError(f"Treatment level column '{treatment_level}' not found in {model_name} data. Available columns: {df.columns.tolist()}")
                    df['response'] = df[treatment_level]  # Use treatment column as response
                
                df['trial'] = range(1, len(df) + 1)  # Add trial numbers
                
                # Select only the required columns (removed 'id' column)
                df = df[['trial', 'model', 'treatment', 'passage', 'response']]
                
                all_data.append(df)
                print(f"    Loaded {len(df)} rows from {model_name}")
                
            except Exception as e:
                print(f"    Error loading {model_name}: {e}")
                continue
        else:
            print(f"  File not found: {dataset_path}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal combined data: {len(combined_df)} rows")
        print(f"Models included: {combined_df['model'].unique().tolist()}")
        return combined_df
    else:
        print("No data found!")
        return pd.DataFrame(columns=['trial', 'model', 'treatment', 'passage', 'response'])

def save_wikisum_data(treatment: str = "control", 
                     treatment_level: str = None,
                     output_path: str = None,
                     models: Optional[List[str]] = None,
                     require_all_treatments: bool = False) -> str:
    """
    Load WikiSum data and save it in the combined format.
    
    Args:
        treatment: Which treatment to load ('control', 'capitalization', 'typo')
        treatment_level: For capitalization: 'S2' or 'S4'
                        For typo: 'S2' or 'S4'
                        For control: None (ignored)
        output_path: Path where to save the combined dataset (if None, auto-generates)
        models: Optional list of specific model names to include. If None, includes all available models.
        require_all_treatments: If True, only include models that have all treatments available.
        
    Returns:
        Path to the saved file
    """
    # Load the data
    df = load_wikisum_data(treatment=treatment, treatment_level=treatment_level, 
                          models=models, require_all_treatments=require_all_treatments)
    
    if df.empty:
        print("No data to save!")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        if treatment_level:
            output_path = f"results_and_data/data/WikiSum/combined/{treatment}_{treatment_level}_dataset.csv"
        else:
            output_path = f"results_and_data/data/WikiSum/combined/{treatment}_dataset.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the combined dataset
    df.to_csv(output_path, index=False)
    print(f"\nCombined dataset saved to: {output_path}")
    
    # Show summary statistics
    print(f"\nDataset Summary:")
    print(f"Treatment: {treatment}")
    print(f"Total rows: {len(df)}")
    print(f"Models: {df['model'].nunique()}")
    print(f"Unique trials: {df['trial'].nunique()}")
    print(f"Rows per model:")
    print(df['model'].value_counts())
    
    return output_path

def create_assist_tag_experiments(base_path: str = "results_and_data/data/WikiSum",
                                 output_base_path: str = "results_and_data/experiments/WikiSum",
                                 models: Optional[List[str]] = None) -> List[str]:
    """
    Create assist tag recognition experiments. Each experiment directory contains control.csv and treatment.csv
    for comparison by assist_tag_rec_Jesse.py.
    
    Experiment types:
    1. Cross-model control comparisons: model_vs_all_others_control_comparison
    2. Within-model typo comparisons: model_control vs model_typo
    3. Within-model capitalization comparisons: model_control vs model_capitalization
    
    Args:
        base_path: Base path to the WikiSum data directory
        output_base_path: Base path for output experiments
        models: Optional list of specific model names to include. If None, includes all available models.
        
    Returns:
        List of created experiment directory paths
    """
    # Get available treatments for each model
    available_treatments = detect_available_treatments(base_path)
    
    # Filter models if specified
    if models is not None:
        available_treatments = {k: v for k, v in available_treatments.items() if k in models}
    
    if not available_treatments:
        print("No models with treatments found!")
        return []
    
    # Create experiments directory
    os.makedirs(output_base_path, exist_ok=True)
    
    experiment_paths = []
    
    print("Creating assist tag recognition experiments...")
    print("Each experiment will contain control.csv and treatment.csv for comparison")
    
    # 1. Cross-model control comparisons (model vs all others)
    print(f"\n{'='*60}")
    print("Creating cross-model control comparisons (model vs all others)...")
    print(f"{'='*60}")
    
    model_list = list(available_treatments.keys())
    for model_name in model_list:
        if 'control' in available_treatments[model_name]:
            # Get all other models with control data
            other_models = [m for m in model_list if m != model_name and 'control' in available_treatments[m]]
            
            if other_models:
                experiment_name = f"{model_name}_vs_all_others_control_comparison"
                experiment_dir = os.path.join(output_base_path, experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)
                
            # Load target model's control data
            target_control = load_wikisum_data(base_path=base_path, treatment='control', models=[model_name])
            
            # Load all other models' control data
            all_other_control = load_wikisum_data(base_path=base_path, treatment='control', models=other_models)
                    
            if not target_control.empty and not all_other_control.empty:
                # Change treatment column for "other models" to 'other_model' so conversation generation works
                all_other_control_modified = all_other_control.copy()
                all_other_control_modified['treatment'] = 'other_model'
                    
                # Save target model as control, all others as treatment
                target_control.to_csv(os.path.join(experiment_dir, "control.csv"), index=False)
                all_other_control_modified.to_csv(os.path.join(experiment_dir, "treatment.csv"), index=False)
                    
                experiment_paths.append(experiment_dir)
                print(f"  ✅ Created {experiment_name}: {len(target_control)} vs {len(all_other_control)} responses")
                print(f"    Target model: {model_name}")
                print(f"    Other models: {other_models}")
            else:
                print(f"  ⚠️  No other models with control data found for comparison with {model_name}")
    
    
    # 2. Within-model typo comparisons
    print(f"\n{'='*60}")
    print("Creating within-model typo comparisons...")
    print(f"{'='*60}")
    
    for model_name, treatments in available_treatments.items():
        if 'control' in treatments and 'typo' in treatments:
            # Try both typo levels
            for typo_level in ['S2', 'S4']:
                try:
                    experiment_name = f"{model_name}_control_vs_typo_{typo_level}"
                    experiment_dir = os.path.join(output_base_path, experiment_name)
                    os.makedirs(experiment_dir, exist_ok=True)
                    
                    # Load control and typo data
                    control_df = load_wikisum_data(base_path=base_path, treatment='control', models=[model_name])
                    typo_df = load_wikisum_data(base_path=base_path, treatment='typo', 
                                              treatment_level=typo_level, models=[model_name])
                    
                    if not control_df.empty and not typo_df.empty:
                        # Save as control.csv and treatment.csv
                        control_df.to_csv(os.path.join(experiment_dir, "control.csv"), index=False)
                        typo_df.to_csv(os.path.join(experiment_dir, "treatment.csv"), index=False)
                        
                        experiment_paths.append(experiment_dir)
                        print(f"  ✅ Created {experiment_name}: {len(control_df)} vs {len(typo_df)} responses")
                        
                except Exception as e:
                    print(f"  ⚠️  Skipped {model_name} {typo_level}: {e}")
    
    # 3. Within-model capitalization comparisons
    print(f"\n{'='*60}")
    print("Creating within-model capitalization comparisons...")
    print(f"{'='*60}")
    
    for model_name, treatments in available_treatments.items():
        if 'control' in treatments and 'capitalization' in treatments:
            # Try both capitalization levels
            for cap_level in ['S2', 'S4']:
                try:
                    experiment_name = f"{model_name}_control_vs_capitalization_{cap_level}"
                    experiment_dir = os.path.join(output_base_path, experiment_name)
                    os.makedirs(experiment_dir, exist_ok=True)
                    
                    # Load control and capitalization data
                    control_df = load_wikisum_data(base_path=base_path, treatment='control', models=[model_name])
                    cap_df = load_wikisum_data(base_path=base_path, treatment='capitalization', 
                                             treatment_level=cap_level, models=[model_name])
                    
                    if not control_df.empty and not cap_df.empty:
                        # Save as control.csv and treatment.csv
                        control_df.to_csv(os.path.join(experiment_dir, "control.csv"), index=False)
                        cap_df.to_csv(os.path.join(experiment_dir, "treatment.csv"), index=False)
                        
                        experiment_paths.append(experiment_dir)
                        print(f"  ✅ Created {experiment_name}: {len(control_df)} vs {len(cap_df)} responses")
                        
                except Exception as e:
                    print(f"  ⚠️  Skipped {model_name} {cap_level}: {e}")
    
    print(f"\n{'='*60}")
    print("Summary of created experiments:")
    print(f"{'='*60}")
    for i, path in enumerate(experiment_paths, 1):
        experiment_name = os.path.basename(path)
        print(f"{i}. {experiment_name}: {path}")
    
    return experiment_paths


def create_per_model_treatment_datasets(base_path: str = "results_and_data/data/WikiSum",
                                       output_base_path: str = "results_and_data/experiments/WikiSum",
                                       models: Optional[List[str]] = None) -> List[str]:
    """
    Create per-model treatment datasets. Each dataset contains control + all treatment levels for one model.
    
    Args:
        base_path: Base path to the WikiSum data directory
        models: Optional list of specific model names to include. If None, includes all available models.
        
    Returns:
        Dictionary mapping model names to output file paths
    """
    # Get available treatments for each model
    available_treatments = detect_available_treatments(base_path)
    
    # Filter models if specified
    if models is not None:
        available_treatments = {k: v for k, v in available_treatments.items() if k in models}
    
    if not available_treatments:
        print("No models with treatments found!")
        return []
    
    output_paths = {}
    
    print("Creating per-model treatment datasets...")
    print("Each dataset will contain: control + all available treatment levels for that model")
    
    for model_name, treatments in available_treatments.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"Available treatments: {sorted(treatments)}")
        print(f"{'='*60}")
        
        all_model_data = []
        
        # Always include control data
        if 'control' in treatments:
            print(f"  Adding control data...")
            control_df = load_wikisum_data(base_path=base_path, treatment='control', models=[model_name])
            if not control_df.empty:
                all_model_data.append(control_df)
                print(f"    Added {len(control_df)} control rows")
        
        # Add capitalization treatments if available
        if 'capitalization' in treatments:
            print(f"  Adding capitalization treatments...")
            for level in ['S2', 'S4']:
                try:
                    cap_df = load_wikisum_data(base_path=base_path, treatment='capitalization', 
                                             treatment_level=level, models=[model_name])
                    if not cap_df.empty:
                        all_model_data.append(cap_df)
                        print(f"    Added {len(cap_df)} {level} rows")
                except Exception as e:
                    print(f"    Skipped {level}: {e}")
        
        # Add typo treatments if available
        if 'typo' in treatments:
            print(f"  Adding typo treatments...")
            for level in ['S2', 'S4']:
                try:
                    typo_df = load_wikisum_data(base_path=base_path, treatment='typo', 
                                              treatment_level=level, models=[model_name])
                    if not typo_df.empty:
                        all_model_data.append(typo_df)
                        print(f"    Added {len(typo_df)} {level} rows")
                except Exception as e:
                    print(f"    Skipped {level}: {e}")
        
        # Combine all data for this model
        if all_model_data:
            combined_df = pd.concat(all_model_data, ignore_index=True)
            
            # Create output path
            output_path = f"results_and_data/data/WikiSum/combined/{model_name}_all_treatments.csv"
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the combined dataset
            combined_df.to_csv(output_path, index=False)
            print(f"\n  ✅ Saved {len(combined_df)} total rows to: {output_path}")
            
            # Show summary
            print(f"  📊 Summary for {model_name}:")
            print(f"    Total rows: {len(combined_df)}")
            print(f"    Treatments: {sorted(combined_df['treatment'].unique())}")
            print(f"    Rows per treatment:")
            treatment_counts = combined_df['treatment'].value_counts()
            for treatment, count in treatment_counts.items():
                print(f"      {treatment}: {count}")
            
            output_paths[model_name] = output_path
        else:
            print(f"  ❌ No data found for {model_name}")
    
    print(f"\n{'='*60}")
    print("Summary of created per-model files:")
    print(f"{'='*60}")
    for model, path in output_paths.items():
        print(f"{model}: {path}")
    
    return output_paths


def create_all_treatment_datasets(base_path: str = "results_and_data/data/WikiSum",
                                 output_base_path: str = "results_and_data/experiments/WikiSum",
                                 models: Optional[List[str]] = None,
                                 require_all_treatments: bool = False) -> List[str]:
    """
    Create combined datasets for all treatments and treatment levels.
    
    Args:
        base_path: Base path to the WikiSum data directory
        models: Optional list of specific model names to include. If None, includes all available models.
        require_all_treatments: If True, only include models that have all treatments available.
        
    Returns:
        Dictionary mapping treatment names to output file paths
    """
    # Define treatment configurations
    treatment_configs = {
        'control': [None],
        'capitalization': ['S2', 'S4'],
        'typo': ['S2', 'S4']
    }
    
    output_paths = {}
    
    print("Creating combined datasets for all treatments and treatment levels...")
    
    for treatment, levels in treatment_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing {treatment} treatment")
        print(f"{'='*60}")
        
        for level in levels:
            if level:
                print(f"\n--- Processing {treatment} level: {level} ---")
                output_path = save_wikisum_data(treatment=treatment, treatment_level=level,
                                              models=models, require_all_treatments=require_all_treatments)
                if output_path:
                    key = f"{treatment}_{level}"
                    output_paths[key] = output_path
            else:
                print(f"\n--- Processing {treatment} (control) ---")
                output_path = save_wikisum_data(treatment=treatment, models=models, 
                                              require_all_treatments=require_all_treatments)
                if output_path:
                    output_paths[treatment] = output_path
    
    print(f"\n{'='*60}")
    print("Summary of created files:")
    print(f"{'='*60}")
    for treatment, path in output_paths.items():
        print(f"{treatment}: {path}")
    
    return output_paths

def show_available_treatments(base_path: str = "results_and_data/data/WikiSum") -> None:
    """
    Display available treatments for each model.
    
    Args:
        base_path: Base path to the WikiSum data directory
    """
    available_treatments = detect_available_treatments(base_path)
    
    print(f"\nAvailable treatments by model:")
    print(f"{'='*60}")
    
    if not available_treatments:
        print("No models with treatments found!")
        return
    
    for model_name, treatments in available_treatments.items():
        print(f"{model_name}: {sorted(treatments)}")
    
    # Show summary
    all_treatments = set()
    for treatments in available_treatments.values():
        all_treatments.update(treatments)
    
    print(f"\nSummary:")
    print(f"Total models: {len(available_treatments)}")
    print(f"Available treatments: {sorted(all_treatments)}")
    
    # Show models with all treatments
    all_treatment_set = {'control', 'capitalization', 'typo'}
    complete_models = [model for model, treatments in available_treatments.items() 
                      if all_treatment_set.issubset(treatments)]
    
    if complete_models:
        print(f"Models with all treatments: {complete_models}")
    else:
        print("No models have all treatments available")

def main():
    """
    Main function - supports both CLI and IDE usage modes
    """
    print("=== WikiSum Data Formatter for Assist Tag Recognition ===")
    print("This script formats WikiSum data for assist tag recognition experiments")
    
    # ===== DETERMINE CONFIG PATH =====
    # Check if running from CLI (has command line arguments) or IDE (no arguments)
    if len(sys.argv) > 1:
        # CLI mode: parse arguments
        args = parse_arguments()
        config_path = args.config
        
        # Handle show-models option
        if args.show_models:
            config = load_config(config_path)
            show_available_treatments(config.get("base_path", "results_and_data/data/WikiSum"))
            return
    else:
        # IDE mode: use hardcoded config path
        config_path = "configs/data/wikisum_data_formatter_config.yaml"
    
    # ===== LOAD CONFIGURATION =====
    config = load_config(config_path)
    
    # Extract configuration parameters
    base_path = config.get("base_path", "results_and_data/data/WikiSum")
    output_base_path = config.get("output_base_path", "results_and_data/experiments/WikiSum")
    selected_models = config.get("selected_models", ["all"])
    require_all_treatments = config.get("require_all_treatments", False)
    treatment = config.get("treatment", None)
    treatment_level = config.get("treatment_level", None)
    operation_mode = config.get("operation_mode", "assist_tag_experiments")
    show_available_treatments_flag = config.get("show_available_treatments", True)
    show_model_summary = config.get("show_model_summary", True)
    show_progress = config.get("show_progress", True)
    verbose = config.get("verbose", True)
    
    print(f"📋 Configuration loaded from {config_path}")
    print(f"🎯 Operation mode: {operation_mode}")
    print(f"🎯 Selected models: {selected_models}")
    print(f"📁 Base path: {base_path}")
    print(f"📁 Output path: {output_base_path}")
    print(f"🔬 Treatment filter: {treatment}")
    print(f"🔬 Treatment level: {treatment_level}")
    print(f"✅ Require all treatments: {require_all_treatments}")
    
    # ===== SHOW AVAILABLE TREATMENTS =====
    if show_available_treatments_flag:
        print(f"\n📊 Available treatments:")
        show_available_treatments(base_path)
    
    # ===== EXECUTE OPERATION =====
    if operation_mode == "show_available":
        # Already shown above, nothing more to do
        pass
    elif operation_mode == "assist_tag_experiments":
        print(f"\n🔬 Creating assist tag recognition experiments...")
        output_dirs = create_assist_tag_experiments(
            base_path=base_path,
            output_base_path=output_base_path,
            models=selected_models if selected_models != ["all"] else None
        )
        print(f"\n✅ Assist tag experiments created successfully!")
        print(f"📁 Experiment directories created: {len(output_dirs)}")
        if verbose and output_dirs:
            print("Created directories:")
            for dir_path in output_dirs:
                print(f"  - {dir_path}")
    elif operation_mode == "per_model":
        print(f"\n📊 Creating per-model treatment datasets...")
        output_files = create_per_model_treatment_datasets(
            base_path=base_path,
            output_base_path=output_base_path,
            models=selected_models if selected_models != ["all"] else None
        )
        print(f"\n✅ Per-model datasets created successfully!")
        print(f"📄 Files created: {len(output_files)}")
        if verbose and output_files:
            print("Created files:")
            for file_path in output_files:
                print(f"  - {file_path}")
    elif operation_mode == "all_treatments":
        print(f"\n📊 Creating all treatment datasets...")
        output_files = create_all_treatment_datasets(
            base_path=base_path,
            output_base_path=output_base_path,
            models=selected_models if selected_models != ["all"] else None,
            require_all_treatments=require_all_treatments
        )
        print(f"\n✅ All treatment datasets created successfully!")
        print(f"📄 Files created: {len(output_files)}")
        if verbose and output_files:
            print("Created files:")
            for file_path in output_files:
                print(f"  - {file_path}")
    else:
        raise ValueError(f"Unknown operation mode: {operation_mode}")
    
    print(f"\n🎉 Operation completed successfully!")


if __name__ == "__main__":
    main()