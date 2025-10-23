# Mock Batch Configuration Files

This directory contains mock batch configuration files for running fast integration tests across all mock experiments.

## Available Mock Batch Configs

### AT_2T (Assistant Tag 2-Turn)
- `AT_2T/rec_config_batch_mock.yaml` - Recognition paradigm
- `AT_2T/pref_config_batch_mock.yaml` - Preference paradigm

### AT_IR (Assistant Tag Injected Response)
- `AT_IR/rec_config_batch_mock.yaml` - Recognition paradigm

### UT_2T (User Tag 2-Turn)
- `UT_2T/rec_config_batch_mock.yaml` - Recognition paradigm
- `UT_2T/pref_config_batch_mock.yaml` - Preference paradigm

## Mock Test Data

All mock batch configs are configured to run experiments in:
```
results_and_data/experiments/mock_test_data/
```

This directory contains:
- `mock_control_vs_capitalization_S2/` - Capitalization treatment experiments
- `mock_control_vs_typo_S2/` - Typo treatment experiments
- `mock_vs_all_others_control_comparison/` - Model comparison experiments

## Usage

### Using run_all_experiments_parallel.py

The recommended way to run mock batch experiments is using the parallel batch runner:

```bash
# Dry run to see what would be executed
python scripts_Jesse/run_all_experiments_parallel.py --config configs/operationalizations/AT_2T/rec_config_batch_mock.yaml --dry-run

# Run with default settings (4 workers)
python scripts_Jesse/run_all_experiments_parallel.py --config configs/operationalizations/AT_2T/rec_config_batch_mock.yaml

# Run with custom number of workers
python scripts_Jesse/run_all_experiments_parallel.py --config configs/operationalizations/AT_2T/rec_config_batch_mock.yaml --max-workers 2

# Continue on error (don't stop if one experiment fails)
python scripts_Jesse/run_all_experiments_parallel.py --config configs/operationalizations/AT_2T/rec_config_batch_mock.yaml --continue-on-error
```

### Using run_experiment.py directly

You can also run a specific mock experiment directly:

```bash
python scripts_Jesse/run_experiment.py --config configs/operationalizations/AT_2T/rec_config_batch_mock.yaml --experiment-dir results_and_data/experiments/mock_test_data/mock_control_vs_capitalization_S2
```

## Configuration Details

All mock batch configs share these settings:
- **max_conversations**: 5 (reduced for faster testing)
- **truncate_words**: 20 (short for faster processing)
- **logging**: Enabled with INFO level
- **log output**: `results_and_data/logs/mock_test_data/`
- **model selection**: Automatically inferred from data (all models in data)

## How It Works

1. **Config specifies parent directory**: The `experiment_dir` in each batch config points to `results_and_data/experiments/mock_test_data`
2. **Auto-discovery**: `run_all_experiments_parallel.py` scans for subdirectories containing `control.csv` and `treatment.csv`
3. **Parallel execution**: Each subdirectory is run as a separate experiment in parallel
4. **Results**: Results are saved to `results_and_data/results/mock_test_data/`
5. **Logs**: Logs are saved to `results_and_data/logs/mock_test_data/`

## Benefits

- **Fast**: Mock models provide instant responses
- **Free**: No API calls or costs
- **Reliable**: Deterministic results for consistent testing
- **Complete**: Tests the entire pipeline including logging and results
- **Parallel**: Multiple experiments run simultaneously for speed

## Example Workflow

```bash
# 1. Run AT_2T recognition experiments (all 3 mock experiments)
python scripts_Jesse/run_all_experiments_parallel.py --config configs/operationalizations/AT_2T/rec_config_batch_mock.yaml

# 2. Run UT_2T recognition experiments
python scripts_Jesse/run_all_experiments_parallel.py --config configs/operationalizations/UT_2T/rec_config_batch_mock.yaml

# 3. Check results
ls results_and_data/results/mock_test_data/

# 4. Check logs
ls results_and_data/logs/mock_test_data/

# 5. Clean up after testing
Remove-Item -Recurse -Force results_and_data/results/mock_test_data
Remove-Item -Recurse -Force results_and_data/logs/mock_test_data
```

## Comparison with Single Mock Configs

| Feature | Single Mock Config | Batch Mock Config |
|---------|-------------------|-------------------|
| Target | Single experiment directory | All experiments in parent directory |
| Config files | `rec_config_mock.yaml`, `pref_config_mock.yaml` | `rec_config_batch_mock.yaml`, `pref_config_batch_mock.yaml` |
| Usage | `run_experiment.py` | `run_all_experiments_parallel.py` |
| Conversations | 10 per experiment | 5 per experiment (faster batch testing) |
| Use case | Test specific experiment | Test entire mock dataset |

