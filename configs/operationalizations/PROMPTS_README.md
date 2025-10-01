# Consolidated Prompt System

## Overview

To eliminate duplication and improve maintainability, prompts are now consolidated into separate `prompts.yaml` files in each operationalization directory. Config files reference these prompts instead of containing them inline.

## Structure

```
configs/operationalizations/
├── AT_2T/
│   ├── prompts.yaml          # Consolidated prompts for AT_2T experiments
│   ├── rec_config.yaml        # Recognition configs reference prompts.yaml
│   ├── pref_config.yaml       # Preference configs reference prompts.yaml
│   └── ...
├── AT_IR/
│   ├── prompts.yaml          # Consolidated prompts for AT_IR experiments
│   ├── rec_config.yaml        # Recognition configs reference prompts.yaml
│   └── ...
└── UT_2T/
    ├── prompts.yaml          # Consolidated prompts for UT_2T experiments
    ├── rec_config.yaml        # Recognition configs reference prompts.yaml
    ├── pref_config.yaml       # Preference configs reference prompts.yaml
    └── ...
```

## Prompts.yaml Format

Each `prompts.yaml` file contains:

```yaml
# Common prompts (used across all paradigms)
system: "System prompt text..."
user: "User prompt template with {passage} placeholder..."

# Recognition paradigm detection prompt
rec_detection: "Detection prompt for recognition..."

# Preference paradigm detection prompt
pref_detection: "Detection prompt for preference..."
```

### Prompt Types

- **`system`**: System prompt that sets up the model's role
- **`user`**: User prompt template (usually contains `{passage}` placeholder)
- **`rec_detection`**: Detection prompt for recognition paradigm
- **`pref_detection`**: Detection prompt for preference paradigm

## Config File Format

Config files now specify prompts using `prompt_set` and `prompt_paradigm`:

```yaml
# Experiment Type Configuration
experiment_type: "AT_2T"  # Must match directory name

# Prompt Configuration
prompt_set: true  # Load prompts from prompts.yaml file
prompt_paradigm: "rec"  # Options: "rec" (recognition) or "pref" (preference)
```

### How It Works

1. **`prompt_set: true`**: Tells the system to load prompts from the corresponding `prompts.yaml` file
2. **`prompt_paradigm`**: Specifies which detection prompt to use (`rec_detection` or `pref_detection`)
3. **Automatic loading**: The system uses `experiment_type` to find the correct `prompts.yaml` file

### Example

For a config with:
```yaml
experiment_type: "AT_2T"
prompt_set: true
prompt_paradigm: "rec"
```

The system will:
1. Look for `configs/operationalizations/AT_2T/prompts.yaml`
2. Load `system`, `user`, and `rec_detection` prompts
3. Use these prompts in the experiment

## Auto-Loading (Fallback)

If neither `prompt_set` nor inline `prompts` are specified, the system will automatically:
1. Infer the paradigm from the config filename (`pref` or `rec`)
2. Load prompts from the appropriate `prompts.yaml` file
3. Print a message: `[INFO] Auto-loaded prompts for {experiment_type}/{prompt_paradigm}`

## Legacy Support

Configs can still use inline prompts (though this is deprecated):

```yaml
# Prompt Templates (legacy format)
prompts:
  system: "System prompt..."
  user: "User prompt..."
  detection: "Detection prompt..."
```

This format is maintained for backward compatibility but should be migrated to use `prompt_set`.

## Benefits

### Before (Inline Prompts)
- ❌ Prompts duplicated across 20+ config files
- ❌ Hard to update prompts consistently
- ❌ Risk of prompts diverging between configs
- ❌ Verbose config files

### After (Consolidated Prompts)
- ✅ Prompts defined once per experiment type
- ✅ Easy to update prompts in one location
- ✅ Guaranteed consistency across configs
- ✅ Clean, concise config files
- ✅ Clear separation between config and prompts

## Experiment Types

### AT_2T (Assistant Tag 2-Turn)
Tests if models can recognize which response was originally produced by themselves in a 2-turn conversation.

**Prompts**: `configs/operationalizations/AT_2T/prompts.yaml`

### AT_IR (Assistant Tag Injected Response)
Tests if models can detect injected text in their own responses.

**Prompts**: `configs/operationalizations/AT_IR/prompts.yaml`

### UT_2T (User Tag 2-Turn)
Tests if models can recognize their own responses from a user's perspective in a 2-turn conversation.

**Prompts**: `configs/operationalizations/UT_2T/prompts.yaml`

## Migration

All existing configs have been automatically migrated using `scripts_Jesse/update_configs_to_prompt_sets.py`.

To migrate new configs manually:
1. Remove the `prompts:` section from the config
2. Add `prompt_set: true`
3. Add `prompt_paradigm: "rec"` or `"pref"`
4. Ensure `experiment_type` matches the directory name (AT_2T, AT_IR, or UT_2T)

## Editing Prompts

To update prompts:
1. Edit the appropriate `prompts.yaml` file
2. Changes apply to all configs using that prompt set
3. No need to update individual config files

Example: To update the AT_2T recognition prompt:
```bash
# Edit this file:
configs/operationalizations/AT_2T/prompts.yaml

# Changes automatically apply to:
# - AT_2T/rec_config.yaml
# - AT_2T/rec_config_batch.yaml
# - AT_2T/rec_config_mock.yaml
# - etc.
```

## Testing

Test prompt loading with any config:
```bash
python scripts_Jesse/run_experiment.py --config configs/operationalizations/AT_2T/rec_config.yaml --show-models
```

The script will automatically load and validate prompts from the appropriate `prompts.yaml` file.

