# Automatic Log Path Generation

## Overview

The logging system now supports automatic path generation that mirrors the results directory structure. This eliminates the need to manually specify log paths for each experiment.

---

## How It Works

### Automatic Path Mirroring

When you set `output_dir: "auto"` in your logging configuration, the system automatically generates a log directory path that mirrors your experiment directory structure:

```
Experiment Directory:
results_and_data/experiments/mock_test_data-UT_2T_rec/mock_control_vs_capitalization_S2

Auto-Generated Log Directory:
results_and_data/logs/mock_test_data-UT_2T_rec/mock_control_vs_capitalization_S2
```

The transformation is simple: `experiments/` → `logs/`

---

## Configuration

### Using Auto-Generated Paths (Default & Recommended)

Simply omit the `output_dir` field and the path will be auto-generated:

```yaml
logging:
  enabled: true
  level: "INFO"
  # output_dir not specified - will auto-generate path
```

### Using Custom Paths (Advanced)

You can still specify custom paths if needed:

```yaml
logging:
  enabled: true
  level: "INFO"
  output_dir: "results_and_data/logs/my_custom_directory"  # Custom path
```

### Legacy Format (Still Supported)

```yaml
logging:
  enabled: true
  level: "INFO"
  output_dir: "experiment_dir/conversation_logs"  # Saves in experiment directory
```

---

## Examples

### Example 1: Mock Test

**Config:**
```yaml
experiment_dir: "results_and_data/experiments/mock_test_data-UT_2T_rec/mock_control_vs_capitalization_S2"
logging:
  enabled: true
  level: "DEBUG"
  # output_dir omitted - will auto-generate
```

**Result:**
- Results: `results_and_data/results/mock_test_data-UT_2T_rec/mock_control_vs_capitalization_S2_choice_results.csv`
- Logs: `results_and_data/logs/mock_test_data-UT_2T_rec/mock_control_vs_capitalization_S2/UT_2T_mock_control_vs_capitalization_S2_TIMESTAMP.json`

### Example 2: WikiSum Experiment

**Config:**
```yaml
experiment_dir: "results_and_data/experiments/WikiSum-AT_2T_pref/gpt-4.1-2025-04-14_control_vs_typo_S2"
logging:
  enabled: true
  level: "INFO"
  # output_dir omitted - will auto-generate
```

**Result:**
- Results: `results_and_data/results/WikiSum-AT_2T_pref/gpt-4.1-2025-04-14_control_vs_typo_S2_choice_results.csv`
- Logs: `results_and_data/logs/WikiSum-AT_2T_pref/gpt-4.1-2025-04-14_control_vs_typo_S2/AT_2T_gpt-4.1-2025-04-14_control_vs_typo_S2_TIMESTAMP.json`

### Example 3: Custom Path

**Config:**
```yaml
experiment_dir: "results_and_data/experiments/WikiSum/my_experiment"
logging:
  output_dir: "special_logs/important_run"
```

**Result:**
- Results: `results_and_data/results/WikiSum/my_experiment_choice_results.csv`
- Logs: `special_logs/important_run/my_experiment_TIMESTAMP.json`

---

## Benefits

1. **Consistency**: Logs and results follow the same directory structure
2. **Simplicity**: No need to manually specify paths for each config
3. **Organization**: Easy to find logs corresponding to specific results
4. **Maintenance**: Update experiment directories without updating log paths

---

## Implementation Details

### Path Resolution Function

The `get_logs_directory_path()` function in `experiment_utils.py` handles the automatic path generation:

```python
def get_logs_directory_path(data_file: str) -> str:
    """
    Get the expected logs directory path based on the data file path.
    Mirrors the results directory structure: experiments/ -> logs/
    """
    # Transforms:
    # results_and_data/experiments/DATASET/SUBDIRECTORY
    # to:
    # results_and_data/logs/DATASET/SUBDIRECTORY
```

### Fallback Behavior

- If `output_dir` is not specified or empty: uses automatic path generation (DEFAULT)
- If `output_dir` is a custom string: uses that path directly
- If `output_dir` starts with `"experiment_dir/"`: resolves relative to experiment directory

---

## Migration Guide

### Updating Existing Configs

To migrate from manual paths to automatic paths:

**Before:**
```yaml
experiment_dir: "results_and_data/experiments/WikiSum/experiment_1"
logging:
  output_dir: "results_and_data/logs/WikiSum_custom"
```

**After:**
```yaml
experiment_dir: "results_and_data/experiments/WikiSum/experiment_1"
logging:
  enabled: true
  level: "INFO"
  # output_dir omitted - will auto-generate results_and_data/logs/WikiSum/experiment_1
```

### No Breaking Changes

All existing configs with explicit paths will continue to work. Auto-generation only happens when `output_dir` is not specified.

---

## Directory Structure Overview

```
results_and_data/
├── experiments/
│   ├── mock_test_data-AT_2T_rec/
│   │   └── mock_control_vs_typo_S2/
│   │       ├── control.csv
│   │       └── treatment.csv
│   └── WikiSum-AT_IR/
│       └── gpt-4_control_vs_typo_S2/
│           ├── control.csv
│           └── treatment.csv
├── results/
│   ├── mock_test_data-AT_2T_rec/
│   │   └── mock_control_vs_typo_S2_choice_results.csv
│   └── WikiSum-AT_IR/
│       └── gpt-4_control_vs_typo_S2_choice_results.csv
└── logs/
    ├── mock_test_data-AT_2T_rec/
    │   └── mock_control_vs_typo_S2/
    │       └── AT_2T_mock_control_vs_typo_S2_TIMESTAMP.json
    └── WikiSum-AT_IR/
        └── gpt-4_control_vs_typo_S2/
            └── AT_IR_gpt-4_control_vs_typo_S2_TIMESTAMP.json
```

---

*Last updated: October 11, 2025*

