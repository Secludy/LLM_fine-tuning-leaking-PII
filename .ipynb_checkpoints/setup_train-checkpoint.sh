#!/bin/bash
set -e

# Create a new Conda environment
echo "Creating a new Conda environment 'train_env' with Python 3.10..."
conda create -n train_env python=3.10 -y

# Activate the Conda environment
echo "Activating the Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate train_env

# Uninstall existing packages
echo "Uninstalling existing torch, torchvision, torchaudio, and xformers..."
pip uninstall torch torchvision torchaudio xformers -y

# Install PyTorch 2.1.0 with CUDA 12.1 support
echo "Installing torch, torchvision, and torchaudio with CUDA 12.1 support..."
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install xformers version compatible with PyTorch 2.1.0
echo "Installing xformers==0.0.23..."
pip install xformers==0.0.23

# Install unsloth with the appropriate extras
echo "Installing unsloth..."
pip install "unsloth[cu121-torch210] @ git+https://github.com/unslothai/unsloth.git"

# Install packages with exact command
pip install packaging unsloth-zoo datasets "accelerate==1.3.0" "peft==0.14.0" "bitsandbytes==0.45.0" "transformers==4.48.0" "trl==0.13.0" "torch==2.1.0" scipy python-dotenv tensorboard

echo "Environment setup completed successfully!"