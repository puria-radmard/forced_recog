#!/usr/bin/env python3
"""
Parallel Batch Runner for Assist Tag Recognition Experiments

This script runs assist_tag_rec_Jesse.py on all experiment directories in parallel.
It uses subprocess to spawn multiple processes and tracks their progress.

USAGE:
    python run_all_experiments_parallel.py --max-workers N --config CONFIG --dry-run

EXAMPLES:
    # Run all experiments in parallel (default: 4 workers)
    python run_all_experiments_parallel.py

    # Run with custom number of parallel workers
    python run_all_experiments_parallel.py --max-workers 8

    # Dry run to see what would be executed
    python run_all_experiments_parallel.py --dry-run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


def find_experiment_directories(experiments_dir: str) -> List[str]:
    """
    Find all experiment directories in the specified folder.
    
    Args:
        experiments_dir: Path to the directory containing experiment folders
        
    Returns:
        List of experiment directory paths
    """
    if not os.path.exists(experiments_dir):
        print(f"❌ Experiments directory not found: {experiments_dir}")
        return []
    
    experiment_dirs = []
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains the required files
            control_file = os.path.join(item_path, "control.csv")
            treatment_file = os.path.join(item_path, "treatment.csv")
            if os.path.exists(control_file) and os.path.exists(treatment_file):
                experiment_dirs.append(item_path)
            else:
                print(f"⚠️  Skipping {item}: Missing control.csv or treatment.csv")
    
    return sorted(experiment_dirs)


def find_experiment_directories_from_config(config_file: str) -> List[str]:
    """
    Find experiment directories by reading the experiment_dir from a config file
    and discovering subdirectories within it.
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        List of experiment directory paths
    """
    import yaml
    
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        base_experiment_dir = config.get("experiment_dir", "")
        if not base_experiment_dir:
            print(f"❌ No experiment_dir found in config file: {config_file}")
            return []
        
        print(f"📋 Using experiment directory from config: {base_experiment_dir}")
        return find_experiment_directories(base_experiment_dir)
        
    except Exception as e:
        print(f"❌ Error reading config file {config_file}: {e}")
        return []


def shuffle_experiments_by_company(experiment_dirs: List[str]) -> List[str]:
    """
    Shuffle experiments to distribute different companies across workers.
    This helps prevent rate limiting by avoiding processing all models from
    the same company in sequence.
    
    Args:
        experiment_dirs: List of experiment directory paths
        
    Returns:
        Shuffled list of experiment directories
    """
    def get_company_from_path(exp_dir: str) -> str:
        """Extract company name from experiment directory path."""
        exp_name = os.path.basename(exp_dir)
        
        # Map model prefixes to companies
        if exp_name.startswith('anthropic_'):
            return 'anthropic'
        elif exp_name.startswith('google_'):
            return 'google'
        elif exp_name.startswith('gpt-'):
            return 'openai'
        else:
            return 'other'
    
    # Group experiments by company
    company_groups = {}
    for exp_dir in experiment_dirs:
        company = get_company_from_path(exp_dir)
        if company not in company_groups:
            company_groups[company] = []
        company_groups[company].append(exp_dir)
    
    # Shuffle within each company group
    for company in company_groups:
        random.shuffle(company_groups[company])
    
    # Interleave experiments from different companies
    shuffled_dirs = []
    max_length = max(len(group) for group in company_groups.values()) if company_groups else 0
    
    for i in range(max_length):
        for company in sorted(company_groups.keys()):  # Sort for consistent ordering
            if i < len(company_groups[company]):
                shuffled_dirs.append(company_groups[company][i])
    
    print(f"🔄 Shuffled experiments by company to distribute API load:")
    company_counts = {}
    for exp_dir in shuffled_dirs:
        company = get_company_from_path(exp_dir)
        company_counts[company] = company_counts.get(company, 0) + 1
    
    for company, count in sorted(company_counts.items()):
        print(f"   {company}: {count} experiments")
    
    return shuffled_dirs


def run_experiment_worker(experiment_dir: str, config_file: str, worker_id: int, 
                         progress_queue: queue.Queue, dry_run: bool = False) -> Tuple[str, bool, str, float]:
    """
    Worker function to run a single experiment.
    
    Args:
        experiment_dir: Path to the experiment directory
        config_file: Path to the configuration file
        worker_id: ID of the worker thread
        progress_queue: Queue for progress updates
        dry_run: If True, only print what would be executed
        
    Returns:
        Tuple of (experiment_name, success, output, duration)
    """
    experiment_name = os.path.basename(experiment_dir)
    
    # Build command
    cmd = [
        sys.executable, "assist_tag_rec_Jesse.py",
        "--config", config_file,
        "--experiment-dir", experiment_dir
    ]
    
    if dry_run:
        progress_queue.put(f"🔍 Worker {worker_id}: DRY RUN - Would execute: {' '.join(cmd)}")
        return experiment_name, True, "DRY RUN", 0.0
    
    progress_queue.put(f"🚀 Worker {worker_id}: Starting {experiment_name}")
    
    try:
        start_time = time.time()
        
        # Set environment to handle Unicode properly on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the command with real-time output capture
        # Set encoding to handle Unicode characters on Windows
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            env=env
        )
        
        # Capture output line by line
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Send progress updates for important lines
                if any(keyword in output.lower() for keyword in ['processing', 'completed', 'error', 'failed']):
                    progress_queue.put(f"📊 Worker {worker_id}: {output.strip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        end_time = time.time()
        duration = end_time - start_time
        
        output_text = '\n'.join(output_lines)
        
        if return_code == 0:
            progress_queue.put(f"✅ Worker {worker_id}: {experiment_name} completed successfully in {duration:.1f}s")
            return experiment_name, True, output_text, duration
        else:
            progress_queue.put(f"❌ Worker {worker_id}: {experiment_name} failed with exit code {return_code}")
            return experiment_name, False, output_text, duration
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time if 'start_time' in locals() else 0.0
        error_msg = f"Error running experiment '{experiment_name}': {e}"
        progress_queue.put(f"❌ Worker {worker_id}: {error_msg}")
        return experiment_name, False, error_msg, duration


def progress_monitor(progress_queue: queue.Queue, total_experiments: int, stop_event: threading.Event):
    """
    Monitor progress queue and display updates.
    
    Args:
        progress_queue: Queue containing progress updates
        total_experiments: Total number of experiments
        stop_event: Event to signal when to stop monitoring
    """
    completed = 0
    while not stop_event.is_set() or not progress_queue.empty():
        try:
            message = progress_queue.get(timeout=1.0)
            print(f"[{completed+1}/{total_experiments}] {message}")
            if "completed successfully" in message or "failed" in message:
                completed += 1
        except queue.Empty:
            continue


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel Batch Runner for Assist Tag Recognition Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--experiments-dir", 
                       default="results_and_data/experiments/to_run",
                       help="Directory containing experiment folders (default: results_and_data/experiments/to_run)")
    
    parser.add_argument("--config",
                       default="configs/assist_tag/assist_tag_config_batch.yaml", 
                       help="Path to the YAML configuration file (default: configs/assist_tag/assist_tag_config_batch.yaml)")
    
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers (default: 4)")
    
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without actually running")
    
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue running other experiments if some fail")
    
    return parser.parse_args()


def main():
    """Main function."""
    print("=== Parallel Batch Runner for Assist Tag Recognition Experiments ===")
    print("This script runs assist_tag_rec_Jesse.py on multiple experiment directories in parallel")
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1
    
    # Find experiment directories
    if args.experiments_dir != "results_and_data/experiments/to_run":
        # Use explicit directory if provided
        print(f"\n🔍 Scanning for experiments in: {args.experiments_dir}")
        experiment_dirs = find_experiment_directories(args.experiments_dir)
    else:
        # Use config file to determine experiment directory
        print(f"\n🔍 Reading experiment directory from config file: {args.config}")
        experiment_dirs = find_experiment_directories_from_config(args.config)
    
    if not experiment_dirs:
        print("❌ No valid experiment directories found!")
        return 1
    
    # Shuffle experiments to distribute different companies across workers
    experiment_dirs = shuffle_experiments_by_company(experiment_dirs)
    
    print(f"📊 Found {len(experiment_dirs)} experiment directories:")
    for i, exp_dir in enumerate(experiment_dirs, 1):
        exp_name = os.path.basename(exp_dir)
        print(f"  {i}. {exp_name}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No experiments will be executed")
        print(f"Configuration file: {args.config}")
        print(f"Max workers: {args.max_workers}")
    
    # Confirm execution
    if not args.dry_run:
        print(f"\n⚠️  About to run {len(experiment_dirs)} experiments in parallel with {args.max_workers} workers...")
        if not args.continue_on_error:
            print("⚠️  If any experiment fails, the batch will continue (use --continue-on-error for more control)")
        
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Cancelled by user")
            return 0
    
    # Set up progress monitoring
    progress_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Start progress monitor thread
    monitor_thread = threading.Thread(
        target=progress_monitor, 
        args=(progress_queue, len(experiment_dirs), stop_event)
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run experiments in parallel
    print(f"\n🚀 Starting parallel execution with {args.max_workers} workers...")
    start_time = time.time()
    
    results = []
    successful = 0
    failed = 0
    
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all experiments
            future_to_experiment = {
                executor.submit(
                    run_experiment_worker, 
                    exp_dir, 
                    args.config, 
                    i+1, 
                    progress_queue, 
                    args.dry_run
                ): exp_dir for i, exp_dir in enumerate(experiment_dirs)
            }
            
            # Process completed experiments
            for future in as_completed(future_to_experiment):
                experiment_dir = future_to_experiment[future]
                try:
                    experiment_name, success, output, duration = future.result()
                    results.append((experiment_name, success, output, duration))
                    
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        if not args.continue_on_error:
                            print(f"\n❌ Stopping batch due to failure in {experiment_name}")
                            # Cancel remaining futures
                            for f in future_to_experiment:
                                f.cancel()
                            break
                            
                except Exception as e:
                    experiment_name = os.path.basename(experiment_dir)
                    print(f"❌ Exception in {experiment_name}: {e}")
                    failed += 1
                    if not args.continue_on_error:
                        break
    
    finally:
        # Stop progress monitoring
        stop_event.set()
        monitor_thread.join(timeout=5)
    
    # Summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"📊 PARALLEL BATCH EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Workers used: {args.max_workers}")
    print(f"Total duration: {total_duration:.1f} seconds")
    
    if successful > 0:
        avg_duration = sum(duration for _, _, _, duration in results if duration > 0) / successful
        print(f"Average per experiment: {avg_duration:.1f} seconds")
        if total_duration > 0:
            print(f"Speedup vs sequential: {len(experiment_dirs) * avg_duration / total_duration:.1f}x")
    
    # Show detailed results
    print(f"\n📋 Detailed Results:")
    for experiment_name, success, output, duration in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {experiment_name} ({duration:.1f}s)")
    
    if failed > 0:
        print(f"\n❌ {failed} experiment(s) failed")
        return 1
    else:
        print(f"\n✅ All experiments completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
