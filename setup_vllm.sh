#!/bin/bash
set -e

# Ensure conda is initialized
eval "$(conda shell.bash hook)"

# Create a new Conda environment
conda create -n vllm_env python=3.10 -y

# Activate the environment
conda activate vllm_env

# Install vllm
pip install vllm==0.5.4 matplotlib seaborn plotly reportlab pyahocorasick

pip install -U kaleido