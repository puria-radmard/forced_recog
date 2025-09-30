# Testing Pipeline

This directory contains the testing pipeline for all operationalizations in the forced_recog project.

## Overview

The testing pipeline provides a systematic way to test all experiment configurations using dedicated test configs that:
- Use shorter text truncation for faster testing
- Save results to dedicated testing directories
- Enable DEBUG logging for detailed output
- Limit conversations to 1 per model for quick validation

## Directory Structure

```
results_and_data/
├── results/
│   └── testing/           # Test results directory
│       ├── AT_2T_test_experiment/
│       ├── AT_IR_test_experiment/
│       └── UT_2T_test_experiment/
└── logs/
    └── testing/           # Test logs directory
        └── (conversation logs for each test)

configs/operationalizations/
├── AT_2T/
│   ├── rec_config.yaml        # Production config
│   └── rec_config_test.yaml   # Test config
├── AT_IR/
│   ├── rec_config.yaml        # Production config
│   └── rec_config_test.yaml   # Test config
└── UT_2T/
    ├── rec_config.yaml        # Production config
    └── rec_config_test.yaml   # Test config
```

## Usage

### List Available Tests
```bash
python scripts_Jesse/run_testing_pipeline.py --list
```

### Run All Tests
```bash
python scripts_Jesse/run_testing_pipeline.py
```

### Run Specific Test
```bash
python scripts_Jesse/run_testing_pipeline.py --config AT_2T
python scripts_Jesse/run_testing_pipeline.py --config AT_IR
python scripts_Jesse/run_testing_pipeline.py --config UT_2T
```

### Clean Testing Directories
```bash
python scripts_Jesse/run_testing_pipeline.py --clean
```

### Check Test Results
```bash
python scripts_Jesse/run_testing_pipeline.py --check
```

## Test Configurations

Each test configuration is optimized for testing:

- **max_conversations**: 1 (instead of "max" for production)
- **truncate_words**: 20 (instead of 50+ for production)
- **logging.level**: "DEBUG" (instead of "INFO" for production)
- **experiment_dir**: Points to testing directories

## Test Output

The testing pipeline generates:
1. **Test Results**: Saved to `results_and_data/results/testing/`
2. **Conversation Logs**: Saved to experiment-specific `conversation_logs/` subdirectories
3. **Test Report**: `results_and_data/results/testing/test_report.txt`

## Integration with CI/CD

The testing pipeline can be integrated into CI/CD workflows:

```bash
# Run tests and exit with appropriate code
python scripts_Jesse/run_testing_pipeline.py
echo $?  # 0 for success, 1 for failure
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

## Production vs Testing

| Aspect | Production | Testing |
|--------|------------|---------|
| Conversations | "max" or high number | 1 |
| Truncation | 50+ words | 20 words |
| Logging | INFO | DEBUG |
| Results Dir | Production experiments | Testing directory |
| Purpose | Full analysis | Quick validation |
