# Mock Model Testing System

## Overview

The Mock Model Testing System provides a way to test the complete experiment pipeline without making actual LLM API calls or loading GPU models. This enables:

- **Instant feedback** on pipeline functionality
- **Fast iteration** during development
- **Complete flow testing** (conversation generation, logging, results processing)
- **No costs** (no API calls)
- **No hardware requirements** (no GPU needed)

## Architecture

### MockChatWrapper

The `MockChatWrapper` class inherits from `BaseChatWrapper` and implements all required methods:
- `format_chat()`: Returns mock-formatted chat strings
- `forward()`: Returns mock logits based on configured mode
- `generate()`: Returns mock text responses

### Integration

The mock model integrates seamlessly with the existing codebase:
1. `model/load.py` recognizes model names starting with "mock"
2. `run_experiment.py` treats mock models like any other model
3. All logging, result saving, and analysis work normally

## Usage

### 1. Basic Usage

Simply specify a mock model in your config file:

```yaml
selected_models: ["mock"]
```

### 2. Testing Modes

Three testing modes are available:

#### Deterministic Mode (Default)
```yaml
selected_models: ["mock"]  # or "mock-deterministic"
```
- Always chooses "1" with 80% confidence
- Predictable results for regression testing
- Best for validating correct pipeline behavior

#### Random Mode
```yaml
selected_models: ["mock-random"]
```
- Random choices with reproducible seed
- Tests result aggregation logic
- Useful for checking statistics calculations

#### Alternating Mode
```yaml
selected_models: ["mock-alternating"]
```
- Alternates between "1" and "2"
- Tests balanced accuracy scenarios
- Validates choice distribution handling

### 3. Example Config

See `configs/operationalizations/AT_2T/rec_config_mock.yaml` for a complete example:

```yaml
experiment_dir: "results_and_data/experiments/mock_test_data"
max_conversations: 10
experiment_type: "AT_2T"
selected_models: ["mock"]
truncate_words: 20
logging:
  enabled: true
  level: "DEBUG"
```

### 4. Running Mock Tests

#### Command Line
```bash
python scripts_Jesse/run_experiment.py --config configs/operationalizations/AT_2T/rec_config_mock.yaml
```

#### Batch File (Windows)
```bash
test_mock_model.bat
```

## Test Data

Mock test data is provided in `results_and_data/experiments/mock_test_data/`:
- `control.csv`: 5 control responses
- `treatment.csv`: 5 treatment responses (typos, capitalization, other_model)

This generates 10 conversations (5 trials × 2 orderings) for quick testing.

## Expected Behavior

### Deterministic Mode
- Always selects choice "1"
- Logits: ~2.0 for "1", ~0.5 for "2"
- Probability: ~73% for "1", ~27% for "2"
- Accuracy: 50% (correct only when control is response 1)

### Random Mode
- Random selections (seeded)
- Variable logits
- Accuracy: ~50% on average

### Alternating Mode
- Alternates: "1", "2", "1", "2", ...
- Probability: ~73% for selected choice
- Accuracy: ~50% (balanced)

## Output

Mock tests produce the same outputs as real experiments:

### 1. Results CSV
```
results_and_data/results/mock_test_data/mock_test_data_choice_results.csv
```

Contains all standard fields:
- conversation_id, trial
- prob_choice_1, prob_choice_2
- selected_choice, correct_choice
- is_correct

### 2. Conversation Logs (if enabled)
```
results_and_data/experiments/mock_test_data/conversation_logs/AT_2T_mock_test_data_YYYYMMDD_HHMMSS.json
```

Contains:
- Formatted chat inputs
- Mock logits/probabilities
- Results and metadata

### 3. Console Output
```
[TESTING] MockChatWrapper initialized in 'deterministic' mode
[TESTING]   This wrapper generates instant mock responses for testing
[TESTING]   No actual LLM calls will be made

=== Processing base model: mock ===
Choice tokens: [[3], [4]]
Processing mock: 100%|████████| 10/10 [00:00<00:00]

📊 RESULTS SUMMARY:
  Overall accuracy: 0.500 (5/10)
```

## Integration Testing

### Test Different Pipeline Components

1. **Conversation Generation**
   ```yaml
   max_conversations: 5
   ```
   Verify conversation pairing logic

2. **Logging System**
   ```yaml
   logging:
     enabled: true
     level: "DEBUG"
   ```
   Validate logging output

3. **Result Processing**
   ```yaml
   experiment_type: "AT_2T"  # or "AT_IR", "UT_2T"
   ```
   Test different experiment types

4. **Text Truncation**
   ```yaml
   truncate_words: 20  # or "max"
   ```
   Verify truncation logic

### Multiple Mock Models

Test multi-model scenarios:
```yaml
selected_models: ["mock-deterministic", "mock-random", "mock-alternating"]
```

This creates separate result sets for each mode, testing:
- Model-specific result tracking
- Multi-model aggregation
- Result merging logic

## Development Workflow

### 1. Make Code Changes
Edit experiment pipeline, logging, or analysis code

### 2. Quick Validation
```bash
python scripts_Jesse/run_experiment.py --config configs/operationalizations/AT_2T/rec_config_mock.yaml
```
Takes ~1 second for 10 conversations

### 3. Check Output
- Verify results CSV structure
- Check log file format
- Validate console output

### 4. Iterate
Repeat until satisfied, then test with real models

## Comparison: Mock vs Real Testing

| Aspect | Mock Testing | Real Testing |
|--------|-------------|--------------|
| Speed | ~1 second | Minutes to hours |
| Cost | $0 | API costs |
| Hardware | None | GPU or API keys |
| Reproducibility | Perfect (seeded) | Variable |
| Use Case | Pipeline testing | Scientific results |

## Troubleshooting

### Mock Model Not Recognized
**Error**: `Unknown model type for 'mock'`

**Solution**: Ensure `model/load.py` includes:
```python
if model_name_lower.startswith('mock'):
    return load_mock_model(model_name)
```

### Token Encoding Issues
**Error**: Choice tokens not found

**Solution**: MockTokenizer returns fixed token IDs:
- "1" → token ID 3
- "2" → token ID 4

These must match `get_choice_tokens()` expectations.

### Unexpected Results
**Issue**: Results don't match expected mode behavior

**Solution**: Check mode spelling:
- "mock" or "mock-deterministic" ✓
- "Mock-Deterministic" ✗ (case sensitive)

## Future Enhancements

Potential additions to the mock system:

1. **Configurable Probabilities**
   ```yaml
   selected_models: ["mock-custom:0.6,0.4"]
   ```

2. **Latency Simulation**
   ```python
   MockChatWrapper(mode="deterministic", delay=0.5)
   ```

3. **Error Simulation**
   ```python
   MockChatWrapper(mode="error-prone", error_rate=0.1)
   ```

4. **Response Templates**
   ```python
   MockChatWrapper(mode="template", template="I choose {choice}")
   ```

## Best Practices

1. **Always test with mock first** before running expensive real experiments
2. **Use deterministic mode** for regression tests
3. **Enable DEBUG logging** to see detailed mock behavior
4. **Keep test data small** (5-10 trials) for fast iteration
5. **Validate full pipeline** (data load → processing → results → analysis)
6. **Clean up test output** before committing

## See Also

- `model/base.py`: BaseChatWrapper interface
- `model/mock.py`: MockChatWrapper implementation
- `model/load.py`: Model loading logic
- `scripts_Jesse/run_experiment.py`: Main experiment script
- `configs/operationalizations/AT_2T/rec_config_mock.yaml`: Example config

