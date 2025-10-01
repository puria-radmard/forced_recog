# Testing Pipeline - Mock Model Integration Tests

This directory contains the testing pipeline for all operationalizations in the forced_recog project.

## Overview

The testing pipeline provides **fast integration testing** using **mock models** to validate the entire experiment pipeline without making LLM calls. Tests complete in seconds and cost nothing.

**Key Benefits:**
- ⚡ **Fast**: All tests complete in ~10 seconds
- 💰 **Free**: No API calls or GPU required
- ✅ **Comprehensive**: Tests entire pipeline end-to-end
- 🔄 **Reproducible**: Deterministic mock responses

**What Gets Tested:**
- Conversation generation logic
- Model loading and initialization
- Chat formatting for all experiment types and paradigms:
  - AT_2T Recognition (self-recognition) - typo dataset
  - AT_2T Preference (model preference) - capitalization dataset
  - AT_IR Recognition (injected response) - capitalization dataset
  - UT_2T Recognition (user tag recognition) - typo dataset
  - UT_2T Preference (user tag preference) - all_others dataset (large)
- Forward pass and logits processing
- Result calculation and storage
- Logging system functionality
- File I/O operations
- Prompt loading from consolidated YAML files

**Dataset Coverage:**
- `mock_control_vs_typo_S2`: Tested by AT_2T_REC, UT_2T_REC
- `mock_control_vs_capitalization_S2`: Tested by AT_2T_PREF, AT_IR_REC
- `mock_vs_all_others_control_comparison`: Tested by UT_2T_PREF (once, as it's large)

## Directory Structure

```
results_and_data/
├── experiments/
│   └── mock_test_data/        # Shared mock test data
│       ├── control.csv        # 5 control responses
│       └── treatment.csv      # 5 treatment responses
├── results/
│   └── testing/               # Test results directory
│       └── (mock test results)
└── logs/
    └── testing/               # Test logs directory
        └── (conversation logs for each test)

configs/operationalizations/
├── AT_2T/
│   ├── rec_config.yaml        # Production config
│   ├── rec_config_test.yaml   # Legacy test config
│   └── rec_config_mock.yaml   # Mock testing config
├── AT_IR/
│   ├── rec_config.yaml        # Production config
│   └── rec_config_mock.yaml   # Mock testing config
└── UT_2T/
    ├── rec_config.yaml        # Production config
    ├── rec_config_test.yaml   # Legacy test config
    └── rec_config_mock.yaml   # Mock testing config
```

## Usage

### List Available Tests
```bash
python scripts_Jesse/run_testing_pipeline.py --list
```

### Run All Tests (~10 seconds)
```bash
python scripts_Jesse/run_testing_pipeline.py
```

**Expected Output:**
```
Running all tests...
==================================================
Running test: AT_2T_REC
...
AT_2T_REC: PASS (2.1s)
--------------------------------------------------

Running test: AT_2T_PREF
...
AT_2T_PREF: PASS (2.2s)
--------------------------------------------------

Running test: AT_IR
...
AT_IR: PASS (2.3s)
--------------------------------------------------

Running test: UT_2T_REC
...
UT_2T_REC: PASS (2.4s)
--------------------------------------------------

Running test: UT_2T_PREF
...
UT_2T_PREF: PASS (2.5s)
--------------------------------------------------

Testing Pipeline Summary:
==================================================
Total Tests: 5
Passed: 5
Failed: 0
Total Duration: 11.5s

All tests passed successfully!
```

### Run Specific Test
```bash
python scripts_Jesse/run_testing_pipeline.py --config AT_2T_REC    # Recognition paradigm
python scripts_Jesse/run_testing_pipeline.py --config AT_2T_PREF   # Preference paradigm
python scripts_Jesse/run_testing_pipeline.py --config AT_IR        # Injected response
python scripts_Jesse/run_testing_pipeline.py --config UT_2T_REC    # Recognition paradigm
python scripts_Jesse/run_testing_pipeline.py --config UT_2T_PREF   # Preference paradigm
```

### Clean Testing Directories
```bash
python scripts_Jesse/run_testing_pipeline.py --clean
```

### Check Test Results
```bash
python scripts_Jesse/run_testing_pipeline.py --check
```

## Mock Test Configurations

Each mock test configuration uses the mock model for instant testing:

- **selected_models**: `["mock"]` (deterministic mock model)
- **max_conversations**: 10 (generates 10 test conversations)
- **truncate_words**: 20 (short text for faster processing)
- **logging.level**: "DEBUG" (detailed output for validation)
- **experiment_dir**: Points to shared `mock_test_data` directory
- **experiment_type**: AT_2T, AT_IR, or UT_2T

The mock model always chooses "1" with ~73% confidence, resulting in 50% accuracy (correct when control is response 1).

## Test Output

The testing pipeline generates:
1. **Test Results**: Saved to `results_and_data/results/testing/`
2. **Conversation Logs**: Saved to experiment-specific `conversation_logs/` subdirectories
3. **Test Report**: `results_and_data/results/testing/test_report.txt`

## Integration with CI/CD

The testing pipeline is perfect for CI/CD workflows due to its speed and zero dependencies:

```bash
# Run tests and exit with appropriate code
python scripts_Jesse/run_testing_pipeline.py
echo $?  # 0 for success, 1 for failure
```

**CI/CD Benefits:**
- ✅ No GPU required
- ✅ No API keys needed
- ✅ Completes in seconds
- ✅ Deterministic results
- ✅ Tests all experiment types

**Example GitHub Actions:**
```yaml
- name: Run Integration Tests
  run: |
    conda activate forced_recog
    python scripts_Jesse/run_testing_pipeline.py
```

## Troubleshooting

### Common Issues

1. **Missing Test Configs**: Run `--list` to verify all configs exist
2. **Permission Errors**: Ensure write access to testing directories
3. **Timeout Issues**: Tests have a 5-minute timeout per configuration

### Validation

The pipeline automatically validates:
- All test config files exist
- Required scripts are present
- Directory structure is correct

### Cleanup

Use `--clean` to remove all test data and start fresh:
```bash
python scripts_Jesse/run_testing_pipeline.py --clean
```

## Development

When adding new operationalizations:
1. Create a new subdirectory in `configs/operationalizations/`
2. Add a `*_config_test.yaml` file
3. Update the `available_tests` dictionary in `run_testing_pipeline.py`
4. Test the new configuration

## Mock Testing vs Production

| Aspect | Mock Testing | Production |
|--------|--------------|------------|
| Model | Mock (instant) | Real LLM |
| Conversations | 10 per test | "max" or high number |
| Duration | ~3 seconds | Minutes to hours |
| Cost | $0 | API costs |
| Hardware | None | GPU or API keys |
| Truncation | 20 words | 50+ words |
| Logging | DEBUG | INFO |
| Purpose | Pipeline validation | Scientific results |

## When to Use Each

### Use Mock Testing Pipeline When:
- ✅ Making code changes to the experiment pipeline
- ✅ Validating new features
- ✅ Running pre-commit checks
- ✅ Setting up CI/CD
- ✅ Onboarding new contributors
- ✅ Debugging pipeline issues

### Use Production Configs When:
- ✅ Conducting actual experiments
- ✅ Generating scientific results
- ✅ Comparing model performance
- ✅ Creating publication data
