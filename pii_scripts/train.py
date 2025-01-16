# LLM_fine-tuning-leaking/pii_scripts/train.py

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
import torch
import json
import os
from dotenv import load_dotenv
import shutil
import dataclasses
import numpy as np
import gc
import sys
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set memory split to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Load environment variables
load_dotenv()
hf_token = os.getenv('HUGGING_FACE_API')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

def manual_merge_model(base_model_path, adapter_path, output_path):
    """Manual merge method with added checks and minimal key transformations.
    Changes:
    - Ensure loading a non-quantized base model.
    - Avoid removing quantization keys here. Just copy over original weights and adapter weights.
    """
    print("Loading base model...")
    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # Load the base model without quantization
    original_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True
    )
    original_state_dict = original_model.state_dict()

    print("\n[manual_merge_model] Original base model keys:")
    for k in original_state_dict.keys():
        print(" ", k)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging weights while preserving architecture...")
    merged_state_dict = {}
    for key in original_state_dict:
        merged_state_dict[key] = original_state_dict[key].clone()
    
    # Merge adapter weights into the base model weights
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            clean_name = name
            if clean_name.startswith("base_model."):
                clean_name = clean_name[len("base_model."):]
            if clean_name in merged_state_dict:
                merged_state_dict[clean_name].copy_(param)

    if 'model.layers.0.mlp.down_proj.weight' not in merged_state_dict:
        print("[manual_merge_model] WARNING: 'model.layers.0.mlp.down_proj.weight' missing after merge! "
              "This suggests no non-quantized parameter was found. Ensure your base model is non-quantized.")

    original_model.load_state_dict(merged_state_dict)

    print("\n[manual_merge_model] Final merged state dict keys:")
    for k in merged_state_dict.keys():
        print(" ", k)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        
    print("Saving merged model...")
    original_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB"
    )
    
    config.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path
    )
    tokenizer.save_pretrained(output_path)
    
    print("Manual merge complete.")
    return True

def save_merged_model(base_model_path, adapter_path, output_path):
    """Improved function to save merged model that handles quantization and LoRA properly"""
    try:
        print("Loading base model for merging...")
        # Load base model without quantization for merging
        config = AutoConfig.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # Important: Load in float16 without quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        print("Loading adapter...")
        # Load the PEFT adapter
        model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            device_map="auto"
        )

        print("Merging weights...")
        # Merge and unload adapter weights
        model = model.merge_and_unload()
        
        # Convert to half precision
        model = model.half()
        
        print("Cleaning state dict...")
        # Get the state dict and clean up layer names
        state_dict = model.state_dict()
        cleaned_dict = {}
        
        for key, value in state_dict.items():
            # Remove problematic prefixes
            new_key = key.replace("base_model.model.", "")
            new_key = new_key.replace("model.model.", "model.")
            
            # Skip any remaining LoRA parameters
            if any(x in new_key.lower() for x in ["lora", "adapter"]):
                continue
                
            cleaned_dict[new_key] = value

        if os.path.exists(output_path):
            print(f"Removing existing directory {output_path}")
            shutil.rmtree(output_path)

        print(f"Saving merged model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        
        # Update model's state dict with cleaned version
        model.state_dict = lambda: cleaned_dict
        
        # Save with appropriate settings
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # Save config and tokenizer
        config.torch_dtype = torch.float16
        config.save_pretrained(output_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path
        )
        tokenizer.save_pretrained(output_path)

        print("Validating saved model...")
        # Try loading the saved model to verify
        test_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Check for any problematic keys
        test_dict = test_model.state_dict()
        problems = [k for k in test_dict.keys() if 'base_model' in k or 'lora' in k.lower()]
        if problems:
            print("Warning: Found problematic keys in saved model:")
            for k in problems:
                print(f" - {k}")
        else:
            print("Model saved and validated successfully!")

        return True

    except Exception as e:
        print(f"Error during model saving: {str(e)}")
        traceback.print_exc()
        return False

###################################
# LOSS LOGGING + TRAJECTORY STORING
###################################
class LogAndStoreLossCallback(TrainerCallback):
    """
    - Logs losses every `logging_steps`.
    - Stores *every* step's loss into `self.losses` so we can do
      fancy post-run trajectory metrics.
    """
    def __init__(self):
        self.losses = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.log_history:
            # The last record in log_history has the newest loss
            last_record = state.log_history[-1]
            loss_val = last_record.get('loss', None)
            if loss_val is not None:
                print(f"Step {state.global_step}: Loss = {loss_val:.4f}")
                self.losses.append(loss_val)

    def get_losses(self):
        return self.losses.copy()

# Model Configuration
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
model_name = "meta-llama/Llama-3.1-8B"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\nReloading base model with quantization...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
    trust_remote_code=True
)


peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

EOS_TOKEN = tokenizer.eos_token
tokenizer.padding_side = 'right'

training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=5e-4,
    bf16=True,
    warmup_ratio=0.03,
    logging_steps=1,
    logging_strategy="steps",
    evaluation_strategy="epoch",
    eval_steps=1,
    report_to=["tensorboard"],
    save_strategy="steps",
    save_steps=100,
    max_seq_length=512,
    dataset_text_field="text",
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,
)  

def main(output_suffix=None):  # e.g. "no_dp_4_PII"
    """
    Main training function that loads data from JSONL and handles model output paths.
    Args:
        output_suffix (str): Suffix for output files (e.g. "no_dp_4_PII")
    """
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct input/output paths
    input_jsonl = os.path.join(current_dir, "data", "final_instruction_formatted_no_dp_4_PII.jsonl")
    
    # Construct output paths based on suffix
    if output_suffix:
        adapter_name = f"trained_model_adapter_{output_suffix}"
        merged_output_dir = f"trained_model_{output_suffix}"
    else:
        adapter_name = "trained_model_adapter"
        merged_output_dir = "trained_model"
    
    # Load and process the JSONL dataset
    print("\nLoading dataset from:", input_jsonl)
    from datasets import load_dataset
    
    # Load the JSONL file using the Hugging Face datasets library
    raw_dataset = load_dataset('json', data_files=input_jsonl)['train']
    print(f"Loaded {len(raw_dataset)} examples")
    
    # Split into train/validation
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })
    
    print(f"Training set size: {len(dataset_dict['train'])}")
    print(f"Validation set size: {len(dataset_dict['validation'])}")
    
    # Print a sample to verify format
    print("\nSample from training data:")
    print(dataset_dict["train"][0])
    
    print("\n***** Starting Training *****")
    # Build trainer
    trainer = SFTTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        args=training_args,
        peft_config=peft_config
    )
    
    # Attach our custom callback that records step losses
    loss_callback = LogAndStoreLossCallback()
    trainer.add_callback(loss_callback)

    print("Starting training...")
    train_output = trainer.train()
    print("Training complete.")

    final_loss = train_output.metrics.get("train_loss", None)
    losses = loss_callback.get_losses()
    
    print(f"\nFinal train_loss (from HF metrics): {final_loss}")
    print(f"Tracked {len(losses)} step losses")

    # Save model with adapter
    print("\nSaving model adapter...")
    trained_model = trainer.model
    for p in trained_model.parameters():
        p.requires_grad = False
    
    if os.path.exists(adapter_name):
        shutil.rmtree(adapter_name)
    trained_model.save_pretrained(adapter_name)
    tokenizer.save_pretrained(adapter_name)
    print(f"Adapter saved to {adapter_name}")

    # Attempt merge
    try:
        print("\nAttempting primary merge method...")
        if save_merged_model(model_name, adapter_name, merged_output_dir):
            print("Successfully saved merged model!")
        else:
            print("\nPrimary merge failed, attempting manual merge...")
            if manual_merge_model(model_name, adapter_name, merged_output_dir):
                print("Successfully saved merged model using manual merge!")
            else:
                raise Exception("Manual merge also failed")
    except Exception as e:
        print(f"\nERROR: Merge attempts failed: {e}")
        print(f"Adapter weights remain in {adapter_name}")
        traceback.print_exc()

    print("\nTraining pipeline complete!")
    return {
        "train_loss": final_loss,
        "loss_trajectory": losses,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train model with configurable output names')
    parser.add_argument('--output-suffix', type=str, help='Suffix for output files (e.g., "no_dp_4_PII")')
    args = parser.parse_args()
    
    main(output_suffix=args.output_suffix)