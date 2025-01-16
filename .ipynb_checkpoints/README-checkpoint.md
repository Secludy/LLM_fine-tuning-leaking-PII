# Email Generation Pipeline

Welcome to the **Email Generation Pipeline**! This project provides a streamlined approach for fine-tuning large language models (LLMs) and generating corporate emails across various categories. It also includes scripts for validating environments, checking GPU availability, training models, generating emails, and logging each step of the process.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation & Setup](#installation--setup)
  - [1. Conda Environments](#1-conda-environments)
  - [2. Setup Scripts](#2-setup-scripts)
- [Usage](#usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Generating Emails](#2-generating-emails)
  - [3. Log Files](#3-log-files)
- [Key Scripts](#key-scripts)
  - [train.sh](#trainsh)
  - [pii_scripts/train.py](#pii_scriptstrainpy)
  - [pii_scripts/vllm_generate_costcopy](#pii_scriptsvllm_generate_costcopy)
- [Data & Outputs](#data--outputs)
- [Customizing the Pipeline](#customizing-the-pipeline)
  - [Modifying Training Hyperparameters](#modifying-training-hyperparameters)
  - [Changing Generation Settings](#changing-generation-settings)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)

---

## Overview

This pipeline is designed to demonstrate how to:

1. **Fine-tune** a base language model using LoRA (Low-Rank Approximation) techniques.  
2. **Generate** realistic corporate emails for categories such as Client Communications, Department Updates, and more.  
3. **Log** all phases (environment validation, training, generation) for better traceability and debugging.

---

## Features

- **Environment Validation**: Checks conda installations, required environments, and GPU availability.  
- **Fine-tuning with LoRA**: Leverages [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft) for efficient model training.  
- **Adapter Merging**: Offers both a primary merge approach and a fallback “manual merge” to combine LoRA adapter weights with the base model.  
- **Email Generation**: Uses [vLLM](https://github.com/vllm-project/vllm) to generate emails in various corporate categories.  
- **Logging**: Detailed logs of training and generation are saved in the `logs/` directory.  

---

## Directory Structure

```
.
├── README.md
├── generated_email_examples_no_dp_4_PII.json
├── logs
│   ├── generation.log
│   └── training.log
├── pii_scripts
│   ├── data
│   │   ├── final_instruction_formatted_no_dp_1_PII.jsonl
│   │   ├── final_instruction_formatted_no_dp_2_PII.jsonl
│   │   └── final_instruction_formatted_no_dp_4_PII.jsonl
│   ├── outputs
│   │   └── leakage_results
│   │       ├── leakage_results_no_dp_1_PII.json
│   │       ├── leakage_results_no_dp_2_PII.json
│   │       ├── leakage_results_no_dp_4_PII.json
│   │       ├── missing_pii_entries_no_dp_1_PII.json
│   │       ├── missing_pii_entries_no_dp_2_PII.json
│   │       ├── missing_pii_entries_no_dp_4_PII.json
│   │       ├── per_category_stats_no_dp_1_PII.json
│   │       ├── per_category_stats_no_dp_2_PII.json
│   │       └── per_category_stats_no_dp_4_PII.json
│   ├── train.py
│   └── vllm_generate_costco.py
├── setup_train.sh
├── setup_vllm.sh
└── train.sh

6 directories, 22 files
```

Key folders and files:

- **`pii_scripts/`**: Core Python scripts for training (`train.py`) and email generation (`vllm_generate_costco.py`).  
- **`logs/`**: Contains `training.log` and `generation.log` for detailed tracking of pipeline activities.  
- **`train.sh`**: Main shell script orchestrating the entire fine-tuning and email generation workflow.  
- **`generated_email_examples_no_dp_4_PII.json`**: Example output file containing generated emails.

---

## Installation & Setup

### 1. Conda Environments

This project assumes you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.  
You should have two conda environments ready:

- **Training Environment**: Named `train_env` in the default configuration of `train.sh`.
- **vLLM Environment**: Named `vllm_env` in the default configuration of `train.sh`.

### 2. Setup Scripts

There are two optional helper scripts:

- `setup_train.sh`: A sample script for creating and configuring your `train_env`.  
- `setup_vllm.sh`: A sample script for creating and configuring your `vllm_env`.  

Modify or run these scripts to create environments if they do not yet exist.

---

## Usage

Below is a high-level guide to using the pipeline:

### 1. Training the Model

1. **Navigate** to the project root.  
2. **Run** `train.sh` with the required arguments:

   ```bash
   ./train.sh <model_name> <output_suffix>
   ```

   **Example**:

   ```bash
   ./train.sh mistralai/Mistral-7B-Instruct-v0.3 no_dp_4_PII
   ```

   - `<model_name>`: The identifier for your base model (e.g., `mistralai/Mistral-7B-Instruct-v0.3`).  
   - `<output_suffix>`: Any suffix used to name your output directories/files (e.g., `no_dp_4_PII`).

   **Actions Performed**:
   - Validates environment (checks for conda, correct envs, GPU availability).
   - Runs `train.py` from `pii_scripts/` to fine-tune the base model.
   - Merges LoRA adapters into the base model.
   - Saves merged model to `trained_model_<output_suffix>` directory.

### 2. Generating Emails

After training completes, `train.sh` automatically proceeds to the generation phase:

- **Generates** corporate emails by calling `vllm_generate_costco.py` in the `vLLM_ENV`.
- **Saves** the output to a file named `generated_email_examples_<output_suffix>.json` in the project root.

**Manual Generation**  
If you only want to generate (and skip training), you can manually call:

```bash
conda activate vllm_env
python pii_scripts/vllm_generate_costco.py \
  --model-path trained_model_no_dp_4_PII \
  --output-file generated_email_examples_no_dp_4_PII.json
conda deactivate
```

### 3. Log Files

- **`logs/training.log`**: Logs all training steps, environment checks, and training outputs.  
- **`logs/generation.log`**: Logs all generation steps, environment checks, and generation details.

Both are automatically managed (cleared and appended) by `train.sh`.

---

## Key Scripts

### **train.sh**

- **Path**: `./train.sh`  
- **Purpose**:  
  1. Checks conda environments and GPU availability.  
  2. Activates `train_env`, runs `train.py`, and merges the trained adapter with the base model.  
  3. Activates `vllm_env`, runs `vllm_generate_costco.py` for email generation.  

**Usage**:

```bash
./train.sh <model_name> <output_suffix>
```
  
### **pii_scripts/train.py**

- **Path**: `pii_scripts/train.py`
- **Purpose**:  
  1. Loads the base model (with 4-bit quantization configuration).  
  2. Uses [LoRA](https://arxiv.org/abs/2106.09685) for parameter-efficient fine-tuning.  
  3. Splits dataset into train/validation, trains the model, and saves an adapter.  
  4. Merges adapter weights back into the base model for a fully merged `.bin` or safetensors model.  

**Key Points**:
- Adjusts quantization settings with `BitsAndBytesConfig`.  
- Uses `SFTTrainer` from `trl` for fine-tuning.  
- Expects an input JSONL named `final_instruction_formatted_<output_suffix>.jsonl`.

### **pii_scripts/vllm_generate_costco.py**

- **Path**: `pii_scripts/vllm_generate_costco.py`
- **Purpose**:  
  1. Loads a merged model via [vLLM](https://github.com/vllm-project/vllm).  
  2. Generates corporate emails for multiple categories in a single or batch manner.  
  3. Saves outputs to a specified `.json` file.  

**Key Points**:
- Relies on `SamplingParams` to control temperature, top-p, top-k, and token limits.  
- Contains a list of corporate email categories.  
- Implements retry logic to ensure enough valid emails are generated.

---

## Data & Outputs

- **Training Data**:  
  In `pii_scripts/data/`, you’ll find JSONL files like:
  - `final_instruction_formatted_no_dp_1_PII.jsonl`  
  - `final_instruction_formatted_no_dp_2_PII.jsonl`  
  - `final_instruction_formatted_no_dp_4_PII.jsonl`  

  These are the instruction datasets used for fine-tuning.

- **Generation Output**:  
  - `generated_email_examples_no_dp_4_PII.json`: A sample output file containing the generated emails.

- **Leakage Results (Optional)**:  
  - `pii_scripts/outputs/leakage_results/` holds JSON files like `leakage_results_no_dp_4_PII.json`.  
  - These are not directly used in training/generation but may relate to separate PII leakage checks.

---

## Customizing the Pipeline

### Modifying Training Hyperparameters

Open `pii_scripts/train.py` and locate the `training_args` definition:

```python
training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    ...
)
```

You can adjust parameters such as `num_train_epochs`, `learning_rate`, and `max_seq_length` to suit your needs.

### Changing Generation Settings

Inside `pii_scripts/vllm_generate_costco.py`, look for:

```python
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=500
)
```

- **`temperature`**: Controls randomness.  
- **`top_p`** and **`top_k`**: Adjust sampling diversity.  
- **`max_tokens`**: Limits the generation length.

Feel free to tweak these values based on your desired email style and length.

---

## Acknowledgements

- **[Hugging Face Transformers](https://github.com/huggingface/transformers)**: For base model architectures and tokenizers.  
- **[TRL (Transformers Reinforcement Learning)](https://github.com/lvwerra/trl)** and **[PEFT](https://github.com/huggingface/peft)**: For the SFT (Supervised Fine-Tuning) trainer and LoRA configurations.  
- **[vLLM Project](https://github.com/vllm-project/vllm)**: For efficient text generation.

---

## Contact

For code maintenance or additional inquiries, please reach out to:

**David Zagardo**  
- GitHub: [@dzagardo](#)  
- Email: [dzagardo@alumni.cmu.edu](mailto:dzagardo@alumni.cmu.edu)