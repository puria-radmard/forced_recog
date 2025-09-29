import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pickle
from dotenv import load_dotenv
import wandb

from utils.util import YamlConfig


def load_modernbert_model(device: str = "auto") -> Tuple[AutoModel, AutoTokenizer]:
    """Load ModernBERT model and tokenizer."""
    model_name = "answerdotai/ModernBERT-base"
    
    print(f"Loading ModernBERT: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()
    
    print(f"ModernBERT loaded on {device}")
    return model, tokenizer


def embed_summaries(summaries: List[str], model: AutoModel, tokenizer: AutoTokenizer, 
                   batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    """Embed a list of summaries using ModernBERT."""
    
    embeddings = []
    device = next(model.parameters()).device
    
    print(f"Embedding {len(summaries)} summaries...")
    
    for i in tqdm(range(0, len(summaries), batch_size)):
        batch_summaries = summaries[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_summaries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def load_base_summaries(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all base model summaries from the specified directory."""
    
    print(f"Loading base summaries from: {base_dir}")
    
    base_summaries = {}
    
    # Look for files matching pattern T0.0_trial0_style*.csv
    for filename in os.listdir(base_dir):
        if filename.startswith("T0.0_trial0_style") and filename.endswith(".csv"):
            # Extract style name
            style_match = re.search(r'T0\.0_trial0_style([^.]+)\.csv', filename)
            if style_match:
                style = style_match.group(1)
                filepath = os.path.join(base_dir, filename)
                df = pd.read_csv(filepath)
                base_summaries[style] = df
                print(f"  Loaded {len(df)} summaries for style: {style}")
    
    if not base_summaries:
        raise ValueError(f"No base summary files found in {base_dir}")
    
    return base_summaries


def discover_steps(run_name: str, wandb_run_name: str, mode: str) -> List[Tuple[str, int]]:
    """Discover all available generation steps for a run."""
    
    if mode == "contrastive":
        base_dir = f"results_and_data/modal_results/results/main/{run_name}/test/contrastive_summaries/{wandb_run_name}"
        summary_file = "contrastive_summaries.csv"
    elif mode == "absolute":
        base_dir = f"results_and_data/modal_results/results/main/{run_name}/test/sfted_summaries/{wandb_run_name}"
        summary_file = "T0.0_trial0_stylenatural.csv"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'contrastive' or 'absolute'")
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Generation directory not found: {base_dir}")
    
    steps = []
    
    # Look for lora_adapters_step_* directories
    for dirname in os.listdir(base_dir):
        step_match = re.search(r'lora_adapters_step_(\d+)', dirname)
        if step_match:
            step_num = int(step_match.group(1))
            step_dir = os.path.join(base_dir, dirname)
            
            # Check if the summary file exists
            if os.path.exists(os.path.join(step_dir, summary_file)):
                steps.append((dirname, step_num))
    
    # Sort by step number
    steps.sort(key=lambda x: x[1])
    
    print(f"Found {len(steps)} {mode} steps: {[s[1] for s in steps]}")
    
    return steps


def load_summaries(summary_dir: str, mode: str) -> pd.DataFrame:
    """Load summaries from a specific step directory."""
    
    if mode == "contrastive":
        summary_file = os.path.join(summary_dir, "contrastive_summaries.csv")
    elif mode == "absolute":
        summary_file = os.path.join(summary_dir, "T0.0_trial0_stylenatural.csv")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    df = pd.read_csv(summary_file)
    print(f"  Loaded {len(df)} {mode} summaries")
    
    return df


def check_document_alignment(base_summaries: Dict[str, pd.DataFrame], 
                           generation_df: pd.DataFrame) -> List[int]:
    """
    Check document alignment and return common indices. 
    Warns but doesn't fail on misalignment.
    """
    
    # Get document indices from base summaries (find intersection of all styles)
    base_doc_indices = None
    for style, df in base_summaries.items():
        current_indices = set(df['document_idx'].values)
        if base_doc_indices is None:
            base_doc_indices = current_indices
        else:
            if current_indices != base_doc_indices:
                print(f"⚠️  Warning: Document indices mismatch in base summaries for style {style}")
                base_doc_indices = base_doc_indices.intersection(current_indices)
                print(f"   Using intersection: {len(base_doc_indices)} documents")
    
    # Check generation indices
    generation_doc_indices = set(generation_df['document_idx'].values)
    
    # Find common indices between base and generation
    common_indices = base_doc_indices.intersection(generation_doc_indices)
    
    if generation_doc_indices != base_doc_indices:
        missing_in_generation = base_doc_indices - generation_doc_indices
        missing_in_base = generation_doc_indices - base_doc_indices
        
        print(f"⚠️  Warning: Document index mismatch between base and generation summaries")
        if missing_in_generation:
            print(f"   Missing in generation: {len(missing_in_generation)} documents")
        if missing_in_base:
            print(f"   Missing in base: {len(missing_in_base)} documents")
        print(f"   Using common indices: {len(common_indices)} documents")
    else:
        print(f"✓ Document alignment verified: {len(common_indices)} documents")
    
    return sorted(list(common_indices))


def compute_style_projections(base_embeddings: Dict[str, np.ndarray], 
                            generation_embeddings: np.ndarray,
                            document_indices: List[int]) -> Dict[str, np.ndarray]:
    """
    Compute projections of generation embeddings onto style difference vectors.
    
    For each style pair, create a vector from base_style1 to base_style2,
    then project generation embeddings onto this vector and normalize 
    so base embeddings are at -1 and +1.
    """
    
    styles = list(base_embeddings.keys())
    projections = {}
    
    print("Computing style projections...")
    
    for i, style1 in enumerate(styles):
        for j, style2 in enumerate(styles):
            if i >= j:  # Skip self and duplicates
                continue
            
            pair_name = f"{style1}_vs_{style2}"
            print(f"  Computing projections for {pair_name}")
            
            # Get base embeddings for this style pair
            base1 = base_embeddings[style1]  # shape: (n_docs, embedding_dim)
            base2 = base_embeddings[style2]  # shape: (n_docs, embedding_dim)
            
            # Compute difference vectors for each document
            diff_vectors = base2 - base1  # shape: (n_docs, embedding_dim)
            
            # Normalize difference vectors
            diff_norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
            diff_vectors_normalized = diff_vectors / (diff_norms + 1e-8)
            
            # Project generation embeddings onto normalized difference vectors
            # For each document, project generation[doc] onto diff_vector[doc]
            projections_raw = np.sum(generation_embeddings * diff_vectors_normalized, axis=1)
            
            # Normalize so that base embeddings are at -1 and +1
            # base1 projection should be -0.5 * ||diff_vector||, base2 should be +0.5 * ||diff_vector||
            # So we divide by 0.5 * ||diff_vector|| to get -1 and +1
            scaling_factors = 0.5 * diff_norms.squeeze()
            projections_normalized = projections_raw / (scaling_factors + 1e-8)
            
            projections[pair_name] = projections_normalized
            
            # Verify: compute what base embeddings project to
            base1_proj = np.sum(base1 * diff_vectors_normalized, axis=1) / (scaling_factors + 1e-8)
            base2_proj = np.sum(base2 * diff_vectors_normalized, axis=1) / (scaling_factors + 1e-8)
            
            print(f"    Base {style1} projection mean: {base1_proj.mean():.3f} (should be ~-1)")
            print(f"    Base {style2} projection mean: {base2_proj.mean():.3f} (should be ~+1)")
            print(f"    Generation projection range: [{projections_normalized.min():.3f}, {projections_normalized.max():.3f}]")
    
    return projections


def create_plots(step_projections: Dict[int, Dict[str, np.ndarray]], 
                output_dir: str, run_name: str, wandb_run_name: str, mode: str) -> None:
    """Create histograms and line plots showing projection evolution."""
    
    if not step_projections:
        print("No step projections to plot")
        return
    
    style_pairs = list(next(iter(step_projections.values())).keys())
    steps = sorted(step_projections.keys())
    earliest_step = steps[0]
    
    print(f"Creating plots for {len(style_pairs)} style pairs across {len(steps)} steps")
    
    # Set up colors for different steps
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
    
    # Create figure with 2 rows
    n_pairs = len(style_pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(6*n_pairs, 10))
    
    # Handle single pair case
    if n_pairs == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Style Projection Evolution ({mode.title()} Mode)\nRun: {run_name}\nLoRA: {wandb_run_name}', 
                 fontsize=16, fontweight='bold')
    
    for pair_idx, pair_name in enumerate(style_pairs):
        
        # TOP ROW: Histograms
        ax_hist = axes[0, pair_idx]
        
        # Plot histogram for each step
        for step_idx, step in enumerate(steps):
            projections = step_projections[step][pair_name]
            
            ax_hist.hist(projections, bins=30, alpha=0.7, 
                        color=colors[step_idx], 
                        histtype='step', linewidth=2,
                        label=f'Step {step}', density=True)
        
        # Mark base model positions
        ax_hist.axvline(-1, color='red', linestyle='--', alpha=0.8, 
                       label=f'{pair_name.split("_vs_")[0]} (base)')
        ax_hist.axvline(+1, color='red', linestyle='--', alpha=0.8, 
                       label=f'{pair_name.split("_vs_")[1]} (base)')
        
        ax_hist.set_xlabel('Projection onto Style Difference Vector')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title(f'{pair_name.replace("_vs_", " vs ")} - Distributions')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # BOTTOM ROW: Line plots of relative changes
        ax_line = axes[1, pair_idx]
        
        # Get baseline projections (earliest step)
        baseline_projections = step_projections[earliest_step][pair_name]
        
        # Compute relative changes and statistics for each step
        step_means = []
        step_stds = []
        
        for step in steps:
            current_projections = step_projections[step][pair_name]
            relative_changes = current_projections - baseline_projections
            
            step_means.append(np.mean(relative_changes))
            step_stds.append(np.std(relative_changes))
        
        # Plot line with error bars
        ax_line.errorbar(steps, step_means, yerr=step_stds, 
                        marker='o', linewidth=2, capsize=5,
                        label='Mean ± Std')
        
        # Add horizontal line at 0 (no change)
        ax_line.axhline(0, color='black', linestyle='--', alpha=0.5, 
                       label='No change')
        
        ax_line.set_xlabel('Training Step')
        ax_line.set_ylabel(f'Change from Step {earliest_step}')
        ax_line.set_title(f'{pair_name.replace("_vs_", " vs ")} - Relative Change')
        ax_line.legend()
        ax_line.grid(True, alpha=0.3)
        
        print(f"  {pair_name}: final change = {step_means[-1]:.3f} ± {step_stds[-1]:.3f}")
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, f'style_projection_evolution_{mode}_{wandb_run_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved projection evolution plot: {plot_file}")


def main():
    """Main entry point with command line argument parsing."""
    
    if len(sys.argv) not in [4, 5]:
        print("Usage:")
        print("  All steps: python script.py config.yaml wandb_run_name mode")
        print("  Single step: python script.py config.yaml wandb_run_name artifact_name mode")
        print("  mode: 'contrastive' or 'absolute'")
        sys.exit(1)
    
    config_path = sys.argv[1]
    wandb_run_name = sys.argv[2]
    
    if len(sys.argv) == 4:
        # 3 args: config, wandb_run_name, mode
        artifact_name = "all"
        mode = sys.argv[3]
    else:
        # 4 args: config, wandb_run_name, artifact_name, mode
        artifact_name = sys.argv[3]
        mode = sys.argv[4]
    
    if mode not in ["contrastive", "absolute"]:
        print(f"Error: Invalid mode '{mode}'. Must be 'contrastive' or 'absolute'")
        sys.exit(1)
    
    print(f"Style Projection Analysis")
    print(f"Config: {config_path}")
    print(f"WandB Run: {wandb_run_name}")
    print(f"Artifact: {artifact_name}")
    print(f"Mode: {mode}")
    
    # Load configuration
    args = YamlConfig(config_path)
    run_name = args.args_name
    
    # Set up paths based on mode
    base_summaries_dir = f"results_and_data/modal_results/results/main/{run_name}/test/model_summaries"
    
    if mode == "contrastive":
        output_dir = f"results_and_data/modal_results/results/main/{run_name}/test/contrastive_summaries/{wandb_run_name}"
    else:  # absolute
        output_dir = f"results_and_data/modal_results/results/main/{run_name}/test/sfted_summaries/{wandb_run_name}"
    
    # Load embedding model
    model, tokenizer = load_modernbert_model()
    
    # Load base summaries
    base_summaries = load_base_summaries(base_summaries_dir)
    
    # Embed base summaries
    base_embeddings = {}
    for style, df in base_summaries.items():
        print(f"Embedding base summaries for style: {style}")
        # Sort by document_idx to ensure consistent ordering
        df_sorted = df.sort_values('document_idx')
        summaries = df_sorted['summary'].tolist()
        embeddings = embed_summaries(summaries, model, tokenizer)
        base_embeddings[style] = embeddings
    
    # Discover and process generation steps
    if artifact_name == "all":
        steps = discover_steps(run_name, wandb_run_name, mode)
    else:
        # Single step mode
        steps = [(artifact_name, int(re.search(r'step_(\d+)', artifact_name).group(1)))]
    
    step_projections = {}
    
    for step_name, step_num in steps:
        print(f"\nProcessing step {step_num} ({step_name})")
        
        # Load generation summaries for this step
        if mode == "contrastive":
            summary_dir = f"results_and_data/modal_results/results/main/{run_name}/test/contrastive_summaries/{wandb_run_name}/{step_name}"
        else:  # absolute
            summary_dir = f"results_and_data/modal_results/results/main/{run_name}/test/sfted_summaries/{wandb_run_name}/{step_name}"

        try:
            generation_df = load_summaries(summary_dir, mode)
        except FileNotFoundError:
            print(f'Skipping {step_name} - not found')
            continue
        
        # Check document alignment and get common indices
        common_indices = check_document_alignment(base_summaries, generation_df)
        
        if len(common_indices) == 0:
            print(f"❌ No common documents found for step {step_num}, skipping")
            continue
        
        # Filter base summaries to common indices
        filtered_base_embeddings = {}
        for style in base_embeddings.keys():
            # Get indices of common documents in the sorted base summaries
            style_df = base_summaries[style].sort_values('document_idx')
            mask = style_df['document_idx'].isin(common_indices)
            filtered_base_embeddings[style] = base_embeddings[style][mask.values]
        
        # Filter generation summaries to common indices
        generation_df_filtered = generation_df[generation_df['document_idx'].isin(common_indices)]
        generation_df_sorted = generation_df_filtered.sort_values('document_idx')
        generation_summaries = generation_df_sorted['summary'].tolist()
        
        # Embed generation summaries
        print(f"Embedding {mode} summaries for step {step_num}")
        generation_embeddings = embed_summaries(generation_summaries, model, tokenizer)
        
        # Compute projections
        projections = compute_style_projections(
            filtered_base_embeddings, generation_embeddings, common_indices
        )
        
        step_projections[step_num] = projections
        
        # Save projections for this step
        projections_file = os.path.join(summary_dir, "style_projections.pkl")
        with open(projections_file, 'wb') as f:
            pickle.dump(projections, f)
        print(f"Saved projections: {projections_file}")
    
    # Create visualization
    create_plots(step_projections, output_dir, run_name, wandb_run_name, mode)
    
    # Save aggregated results
    aggregated_file = os.path.join(output_dir, f"aggregated_style_projections_{mode}.pkl")
    with open(aggregated_file, 'wb') as f:
        pickle.dump({
            'step_projections': step_projections,
            'base_embeddings': base_embeddings,
            'config': config_path,
            'wandb_run_name': wandb_run_name,
            'mode': mode
        }, f)
    print(f"Saved aggregated results: {aggregated_file}")
    
    print(f"\n{'='*70}")
    print("Style Projection Analysis Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()