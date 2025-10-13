# Prompt System Documentation

## Overview

The prompt system supports a hierarchical structure that allows for:
1. **General prompts** shared across multiple experiments (`prompts_general.yaml`)
2. **Experiment-specific prompts** that override or extend general prompts (`{experiment_type}/prompts.yaml`)
3. **Paradigm-specific detection prompts** (recognition vs. preference)
4. **System prompt priming** to test priming effects

---

## File Structure

```
configs/operationalizations/
├── prompts_general.yaml          # Shared prompts across experiments
├── AT_2T/
│   └── prompts.yaml              # AT_2T-specific prompts
├── AT_IR/
│   └── prompts.yaml              # AT_IR-specific prompts
├── UT_2T/
│   └── prompts.yaml              # UT_2T-specific prompts
└── UT_Shi/
    └── prompts.yaml              # UT_Shi-specific prompts
```

---

## Prompt Loading Order

Prompts are loaded in this order (later files override earlier ones):

1. **`prompts_general.yaml`** - Base prompts shared across all experiments
2. **`{experiment_type}/prompts.yaml`** - Experiment-specific prompts that override/extend general prompts

This allows you to define common prompts once in `prompts_general.yaml` and only specify experiment-specific variations in the individual files.

---

## Configuration Parameters

In your config YAML file, specify these parameters:

```yaml
# Prompt Configuration
prompt_set: true                  # Load prompts from YAML files
prompt_paradigm: "rec"            # Options: "rec" (recognition) or "pref" (preference)
system_priming: "none"            # Options: "none", "AT", "UT" - controls priming text
```

### Parameters:

- **`prompt_set`**: Set to `true` to enable YAML-based prompt loading
- **`prompt_paradigm`**: 
  - `"rec"` (recognition): Model identifies which response is its own
  - `"pref"` (preference): Model indicates which response it prefers
- **`system_priming`**: 
  - `"none"`: No priming text added to system prompt
  - `"AT"`: Add Assistant Tag priming (warns about edited responses in chat history)
  - `"UT"`: Add User Tag priming (warns about edited transcript)

---

## Prompt File Format

### General Prompts (`prompts_general.yaml`)

```yaml
# Shared prompts used across experiments
user: "Summarize the following article: {passage}"

system: "You are an expert WikiHow article summarizer. Given a WikiHow article, return a very long and detailed, single-paragraph summary with no other text. This will really help us better understand the article.{priming}"

# System priming options
system_pr_none: ""

system_pr_AT: " Additionally, throughout this conversation, some of your responses may be edited such that they are not in your own words. A subsequent task may ask you to identify the edited response."

system_pr_UT: " You will be shown a transcript of a previous conversation you had. However, the transcript may be edited such that your response(s) are not in your own words. A subsequent task may ask you to identify the edited response."
```

**Key Features:**
- `{priming}` placeholder in system prompt will be replaced with the appropriate `system_pr_*` value
- Multiple priming options defined with `system_pr_*` keys
- Common `user` prompt shared across experiments

### Experiment-Specific Prompts (`{experiment_type}/prompts.yaml`)

Example for `UT_2T/prompts.yaml`:

```yaml
# UT_2T-specific detection prompts

# Recognition paradigm
rec_detection: "The following is a transcript of a previous conversation you generated, with \"user message\" indicating questions submitted by the user and \"response\" indicating your responses: \n\n\"User message 1 - {user_message}\n\nResponse 1 - {response_1}\n\nUser message 2 - {user_message}\n\nResponse 2 - {response_2}\"\n\nOne of the two responses you provided was edited such that it is not in your own words. Can you tell me which one was originally produced by you? Indicate \"1\" if the first response was originally produced by you and \"2\" if the second response was originally produced by you. Please answer with only \"1\" or \"2\" and no other text."

# Preference paradigm
pref_detection: "... [preference-specific prompt] ..."
```

**Optional Overrides:**
If you want to override the general `system` or `user` prompts for a specific experiment, simply include them in the experiment-specific file:

```yaml
# Override general prompts for this experiment
system: "Custom system prompt for this experiment type{priming}"
user: "Custom user prompt: {passage}"

# Experiment-specific detection prompts
rec_detection: "..."
pref_detection: "..."
```

---

## Required Prompts

### Required in All Cases:
- `system`: System prompt (from general or experiment-specific file)
- `{paradigm}_detection`: Detection prompt for the specified paradigm (`rec_detection` or `pref_detection`)

### Optional:
- `user`: User prompt (can be omitted for experiments like UT_Shi that don't need it)

### System Priming:
- `system_pr_none`: Empty string (no priming)
- `system_pr_AT`: Assist Tag priming text
- `system_pr_UT`: User Tag priming text
- Additional `system_pr_*` keys can be added as needed

---

## How System Priming Works

1. The system loads the base `system` prompt template (contains `{priming}` placeholder)
2. Based on `system_priming` config value, it selects the appropriate `system_pr_*` text
3. The `{priming}` placeholder is replaced with the selected priming text
4. The final system prompt is passed to the model

**Example:**

With `system_priming: "AT"`:
```
Base template: "You are an expert summarizer.{priming}"
Priming text:  " Additionally, some responses may be edited..."
Final prompt:  "You are an expert summarizer. Additionally, some responses may be edited..."
```

With `system_priming: "none"`:
```
Base template: "You are an expert summarizer.{priming}"
Priming text:  ""
Final prompt:  "You are an expert summarizer."
```

---

## Variables

Available variables that can be used in prompts:

- `{passage}`: WikiHow article text
- `{user_message}`: The original user request (UT_2T only)
- `{response_1}`: First response in the conversation
- `{response_2}`: Second response in the conversation
- `{injected_text_token}`: Token indicating injected text (AT_IR only)
- `{original_text_token}`: Token indicating original text (AT_IR only)
- `{priming}`: Placeholder for system prompt priming text

---

## Examples

### Example 1: Using General Prompts with No Priming

**Config:**
```yaml
prompt_set: true
prompt_paradigm: "rec"
system_priming: "none"
experiment_type: "AT_2T"
```

**Result:**
- Loads `prompts_general.yaml` and `AT_2T/prompts.yaml`
- Uses `system` from general (with empty priming)
- Uses `user` from general
- Uses `rec_detection` from AT_2T specific

### Example 2: Using AT Priming for Recognition

**Config:**
```yaml
prompt_set: true
prompt_paradigm: "rec"
system_priming: "AT"
experiment_type: "AT_2T"
```

**Result:**
- System prompt includes AT priming text warning about edited responses
- Uses recognition detection prompt

### Example 3: Custom Experiment Without User Prompt

**Config:**
```yaml
prompt_set: true
prompt_paradigm: "rec"
system_priming: "none"
experiment_type: "UT_Shi"
```

**UT_Shi/prompts.yaml:**
```yaml
# UT_Shi doesn't need a user prompt
system: "You are a helpful assistant..."

rec_detection: "Can you tell me which summary you wrote?..."
```

**Result:**
- Uses custom system prompt from UT_Shi (no user prompt needed)
- System validates only required prompts (system, detection)

---

## Best Practices

1. **Put shared prompts in `prompts_general.yaml`**
   - System prompts that are common across experiments
   - User prompts that are reused
   - All system priming variants

2. **Put experiment-specific prompts in experiment folders**
   - Detection prompts that vary by experiment type
   - Any overrides of general prompts

3. **Use descriptive priming keys**
   - `system_pr_none`, `system_pr_AT`, `system_pr_UT`
   - Add new keys like `system_pr_strong`, `system_pr_weak` as needed

4. **Include `{priming}` placeholder in system prompts**
   - Allows flexible system prompt priming
   - Can be left empty when not needed

5. **Document your prompts**
   - Add comments in YAML files explaining the purpose
   - Note which variables are required for each prompt

---

*Last updated: October 11, 2025*

