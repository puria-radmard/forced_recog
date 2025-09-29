# Forced Recognition Project

A comprehensive system for testing AI model self-recognition capabilities through assist tag recognition experiments. This project evaluates whether AI models can identify which responses they originally generated when presented with modified versions.

## 🎯 Project Overview

This project tests AI models' ability to recognize their own generated content when it has been modified (e.g., through capitalization changes, typos, or other treatments). The system supports multiple model providers (Anthropic, Google, OpenAI, HuggingFace) and various experimental protocols.

## 📁 Core File Structure

### 🔬 Main Experiment Scripts

#### `scripts_Jesse/run_experiment.py` - **CORE EXPERIMENT SCRIPT**
The primary script for running assist tag recognition experiments. This is the main entry point for all experiments.

**Key Features:**
- **Dual Mode Operation**: IDE mode (default) for debugging, CLI mode for production
- **Multi-Provider Support**: Anthropic, Google, OpenAI, HuggingFace models
- **Experiment Types**: `assist_tag` (self-recognition) and `user_tag` (preference)
- **YAML Configuration**: All parameters configurable through YAML files
- **Model Selection**: Run all models or specific subsets

**Usage:**
```bash
# IDE mode (uses hardcoded config)
python scripts_Jesse/run_experiment.py

# CLI mode (specify config)
python scripts_Jesse/run_experiment.py --config configs/operationalizations/AT_2T/rec_config.yaml

# Show available models
python scripts_Jesse/run_experiment.py --show-models --config configs/operationalizations/AT_2T/rec_config.yaml
```

#### `scripts_Jesse/run_all_experiments_parallel.py` - **BATCH PROCESSOR**
Runs `run_experiment.py` across multiple experiment directories in parallel.

**Key Features:**
- **Parallel Execution**: Configurable number of workers (default: 4)
- **Progress Tracking**: Real-time status updates and completion monitoring
- **Error Handling**: Robust error reporting and recovery
- **Dry Run Mode**: Preview what would be executed

**Usage:**
```bash
# Run all experiments in parallel
python scripts_Jesse/run_all_experiments_parallel.py

# Custom worker count
python scripts_Jesse/run_all_experiments_parallel.py --max-workers 8

# Preview execution
python scripts_Jesse/run_all_experiments_parallel.py --dry-run
```

### 📊 Data Generation

#### `data_generation/generate_wikisum_treatments.py`
Generates treatment files for WikiSum dataset experiments.

**Purpose:**
- Creates capitalization and typo treatment files
- Maintains consistent data format across experiments
- Supports multiple model directories

#### `data_generation/string_modifier.py`
Core utility for text modifications (capitalization, typos, etc.).

### 📈 Analysis Pipeline

#### `analysis_Jesse/analyze_results_1.py` - **PRIMARY ANALYSIS**
Analyzes experiment results and generates detailed reports.

**Outputs:**
- Accuracy by experiment and model
- Detailed pivot tables and heatmaps
- Treatment-specific breakdowns
- Statistical summaries

#### `analysis_Jesse/analyze_results_2.py` - **AGGREGATION**
Aggregates results from multiple experiments across different conditions.

**Features:**
- Combines data from multiple experiment directories
- Splits by tag type (UT/AT) and treatment type (caps/typo vs model_comps)
- Generates comprehensive comparison tables

#### Visualization Scripts
- `create_scatter_plots.py` - Preference vs recognition scatter plots
- `create_at_ut_comparison_plots.py` - AT vs UT accuracy comparisons
- `create_pref_rec_comparison_plots.py` - Preference vs recognition comparisons
- `create_recognition_accuracy_plots.py` - Recognition accuracy focused plots
- `create_at_ut_recognition_plots.py` - AT vs UT recognition analysis
- `create_at_ut_preference_plots.py` - AT vs UT preference analysis
- `create_at_ut_combined_plots.py` - Combined preference and recognition analysis

### ⚙️ Configuration

#### `configs/operationalizations/` - **EXPERIMENT CONFIGURATIONS**
YAML configuration files for different experiment types:

**AT_2T (Assistant Tags, 2-Turn):**
- `rec_config.yaml` - Recognition experiments
- `pref_config.yaml` - Preference experiments
- `*_batch.yaml` - Batch processing configurations

**UT_2T (User Tags, 2-Turn):**
- `rec_config.yaml` - Recognition experiments
- `pref_config.yaml` - Preference experiments
- `*_batch.yaml` - Batch processing configurations

**Data Configuration:**
- `configs/data/` - Dataset generation configurations

**Key Configuration Parameters:**
- `experiment_dir`: Path to experiment data
- `max_conversations`: Number of conversations to process
- `experiment_type`: "assist_tag" or "user_tag"
- `selected_models`: Model selection criteria
- `truncate_words`: Text truncation settings
- `prompts`: System and user prompt templates

### 🗂️ Data Structure

#### `results_and_data/`
- `experiments/` - Raw experiment data and treatment files
- `results/` - Processed experiment results
- `analysis/` - Analysis outputs and visualizations
- `data/` - Source datasets (WikiSum, etc.)

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (API keys)
# Create .env file with your API keys
```

### 2. Run a Single Experiment
```bash
# Edit config file
nano configs/operationalizations/AT_2T/rec_config.yaml

# Run experiment
python scripts_Jesse/run_experiment.py
```

### 3. Run Batch Experiments
```bash
# Run all experiments in parallel
python scripts_Jesse/run_all_experiments_parallel.py --max-workers 4
```

### 4. Analyze Results
```bash
# Primary analysis
python analysis_Jesse/analyze_results_1.py --results-dir results_and_data/results/EXPERIMENT_NAME --output-dir results_and_data/analysis/EXPERIMENT_NAME

# Aggregation analysis
python analysis_Jesse/analyze_results_2.py

# Generate visualizations
python analysis_Jesse/create_scatter_plots.py
```

## 🔧 Key Components

### Model Support
- **Anthropic**: Claude models (haiku, sonnet, opus)
- **Google**: Gemini models (flash, pro)
- **OpenAI**: GPT models (4o-mini, 4.1-mini, 4.1)
- **HuggingFace**: Various open-source models

### Experiment Types
- **Assist Tag Recognition**: Can the model identify its own responses?
- **User Tag Preference**: Which response does the model prefer?

### Treatment Types
- **Capitalization**: Various capitalization patterns
- **Typos**: Different typo intensities
- **Model Comparisons**: Cross-model response comparisons

## 📋 Workflow

1. **Data Preparation**: Generate treatment files using `generate_wikisum_treatments.py`
2. **Configuration**: Set up experiment parameters in YAML configs
3. **Execution**: Run experiments using `run_experiment.py` or batch processing
4. **Analysis**: Process results with analysis pipeline
5. **Visualization**: Generate plots and reports

## 🛠️ Development Notes

### For New Developers

**Start Here:**
1. Read `scripts_Jesse/run_experiment.py` - This is the core experiment logic
2. Examine `configs/operationalizations/AT_2T/rec_config.yaml` - Understand configuration structure
3. Run a simple experiment to understand the workflow
4. Explore `analysis_Jesse/` scripts to understand result processing

**Key Files to Understand:**
- `scripts_Jesse/run_experiment.py` - Main experiment logic
- `scripts_Jesse/run_all_experiments_parallel.py` - Batch processing
- `analysis_Jesse/analyze_results_1.py` - Primary analysis
- `configs/operationalizations/AT_2T/rec_config.yaml` - Configuration template

**Important Directories:**
- `scripts_Jesse/` - Core experiment scripts
- `analysis_Jesse/` - Analysis and visualization
- `configs/operationalizations/` - Experiment configurations
- `data_generation/` - Data generation utilities
- `results_and_data/` - All data and results

## 📝 Configuration Examples

### Basic Recognition Experiment
```yaml
experiment_dir: "results_and_data/experiments/WikiSum/model_comparison"
max_conversations: 100
experiment_type: "assist_tag"
selected_models: ["all"]
truncate_words: "max"
```

### Batch Processing
```yaml
experiment_dir: "results_and_data/experiments/to_run"
max_conversations: "max"
experiment_type: "assist_tag"
selected_models: ["all"]
```

## 🔍 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure `.env` file contains valid API keys
2. **Model Not Found**: Check model names in configuration
3. **Data Path Errors**: Verify experiment directory paths
4. **Memory Issues**: Reduce `max_conversations` or use fewer parallel workers

### Debug Mode
Use IDE mode for debugging:
```bash
python scripts_Jesse/run_experiment.py
```

This uses hardcoded configuration and provides detailed output for troubleshooting.

## 📊 Output Structure

### Experiment Results
- Raw response data
- Accuracy metrics
- Treatment-specific results
- Model performance comparisons

### Analysis Outputs
- Pivot tables and heatmaps
- Statistical summaries
- Visualization plots
- Aggregated comparisons

## 🤝 Contributing

When working on this codebase:
1. **Start with `run_experiment.py`** - Understand the core experiment logic
2. **Use configuration files** - Don't hardcode parameters
3. **Test with small datasets** - Use `max_conversations: 2` for testing
4. **Follow the analysis pipeline** - Use the established analysis scripts
5. **Document changes** - Update this README when adding new features

## 📚 Additional Resources

- **Model Documentation**: Check individual model provider documentation
- **Configuration Examples**: See `configs/operationalizations/` for various setups
- **Analysis Examples**: Examine `analysis_Jesse/` for result processing patterns
- **Data Generation**: Check `data_generation/` for dataset creation utilities
- **Data Format**: Check `results_and_data/` for expected data structures
