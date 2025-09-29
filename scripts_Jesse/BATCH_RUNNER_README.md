# Batch Runner for Assist Tag Recognition Experiments

This directory contains a batch runner script that can execute `run_experiment.py` on multiple experiment directories without creating separate YAML configuration files for each one.

## Files

- `run_all_experiments.py` - Sequential batch runner script
- `run_all_experiments_parallel.py` - **Parallel batch runner script (recommended)**
- `run_all_experiments.sh` - Bash script for parallel execution (Linux/Mac)
- `run_all_experiments.bat` - Windows batch file wrapper
- `run_experiment.py` - Updated to accept `--experiment-dir` argument
- `configs/assist_tag_config_batch.yaml` - Configuration file that points to the `to_run` directory

## Usage

### Basic Usage

#### Parallel Execution (Recommended)
```bash
# Run all experiments in parallel (default: 4 workers)
python run_all_experiments_parallel.py

# Run with custom number of parallel workers
python run_all_experiments_parallel.py --max-workers 8

# Windows users can use the batch file
run_all_experiments.bat --max-workers 6
```

#### Sequential Execution
```bash
# Run all experiments sequentially (slower but simpler)
python run_all_experiments.py

# Run all experiments in a specific directory
python run_all_experiments.py --experiments-dir results_and_data/experiments/other_experiments
```

### Advanced Options

#### Parallel Execution
```bash
# Dry run to see what would be executed
python run_all_experiments_parallel.py --dry-run

# Use a custom configuration file
python run_all_experiments_parallel.py --config configs/assist_tag_config.yaml

# Continue running even if some experiments fail
python run_all_experiments_parallel.py --continue-on-error

# Show help
python run_all_experiments_parallel.py --help
```

#### Sequential Execution
```bash
# Dry run to see what would be executed
python run_all_experiments.py --dry-run

# Use a custom configuration file
python run_all_experiments.py --config configs/assist_tag_config.yaml

# Continue running even if some experiments fail
python run_all_experiments.py --continue-on-error

# Show help
python run_all_experiments.py --help
```

### Individual Experiment

```bash
# Run a single experiment
python run_experiment.py --config configs/assist_tag_config_batch.yaml --experiment-dir results_and_data/experiments/to_run/experiment_name

# Show models in a specific experiment
python run_experiment.py --config configs/assist_tag_config_batch.yaml --experiment-dir results_and_data/experiments/to_run/experiment_name --show-models
```

## How It Works

1. **Config-Based Discovery**: The script reads the `experiment_dir` from the batch config file (`configs/assist_tag_config_batch.yaml`) which points to `results_and_data/experiments/to_run`
2. **Subdirectory Scanning**: It then scans the `to_run` directory for subdirectories containing `control.csv` and `treatment.csv` files
3. **Configuration Override**: For each experiment, it uses the `--experiment-dir` argument to override the experiment directory setting in the YAML config
4. **Batch Execution**: Runs `run_experiment.py` sequentially on each valid experiment directory
5. **Progress Tracking**: Shows progress, timing, and success/failure status for each experiment

## Directory Structure

```
results_and_data/experiments/to_run/
├── experiment_1/
│   ├── control.csv
│   └── treatment.csv
├── experiment_2/
│   ├── control.csv
│   └── treatment.csv
└── ...
```

## Features

### Parallel Execution (Recommended)
- ✅ **Parallel Processing**: Runs multiple experiments simultaneously
- ✅ **Configurable Workers**: Set number of parallel workers (default: 4)
- ✅ **Real-time Progress**: Live progress updates from all workers
- ✅ **Performance**: Significant speedup over sequential execution
- ✅ **Cross-platform**: Works on Windows, Linux, and Mac

### General Features
- ✅ **Single Config**: Uses one YAML configuration file for all experiments
- ✅ **Directory Override**: Automatically overrides experiment directory per run
- ✅ **Progress Tracking**: Shows real-time progress and timing
- ✅ **Error Handling**: Can continue or stop on errors
- ✅ **Dry Run**: Preview what would be executed
- ✅ **Validation**: Checks for required files before running
- ✅ **Summary**: Detailed execution summary at the end

## Performance Benefits

**Parallel execution provides significant performance improvements:**

- **4x faster** with 4 workers (default)
- **8x faster** with 8 workers
- **Linear scaling** up to your system's capabilities
- **Better resource utilization** of multi-core systems

**Example timing:**
- Sequential: 4 experiments × 45 seconds each = 3 minutes
- Parallel (4 workers): 4 experiments ÷ 4 workers × 45 seconds = 45 seconds

## Example Output

```
=== Batch Runner for Assist Tag Recognition Experiments ===
This script runs run_experiment.py on multiple experiment directories

🔍 Scanning for experiments in: results_and_data/experiments/to_run
📊 Found 4 experiment directories:
  1. anthropic_claude-3-5-haiku-20241022_vs_all_others_control_comparison
  2. anthropic_claude-sonnet-4-20250514_vs_all_others_control_comparison
  3. gpt-4.1-2025-04-14_vs_all_others_control_comparison
  4. gpt-4.1-mini-2025-04-14_vs_all_others_control_comparison

🚀 Starting batch execution...

================================================================================
🚀 Running experiment: anthropic_claude-3-5-haiku-20241022_vs_all_others_control_comparison
📁 Directory: results_and_data/experiments/to_run/anthropic_claude-3-5-haiku-20241022_vs_all_others_control_comparison
⚙️  Config: configs/assist_tag_config.yaml
================================================================================

✅ Experiment 'anthropic_claude-3-5-haiku-20241022_vs_all_others_control_comparison' completed successfully in 45.2 seconds

...

================================================================================
📊 BATCH EXECUTION SUMMARY
================================================================================
Total experiments: 4
Successful: 4
Failed: 0
Total duration: 180.5 seconds
Average per experiment: 45.1 seconds

✅ All experiments completed successfully!
```
