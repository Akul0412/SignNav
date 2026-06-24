#!/bin/bash
#SBATCH --job-name=signnav_decouple
#SBATCH --partition=msigpu            # <-- EDIT if needed (check `sinfo`)
#SBATCH --gres=gpu:a100:1
#SBATCH --account=gini
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL --mail-user=munda057@umn.edu
#SBATCH --time=00:40:00
#SBATCH --output=signnav_decouple_%j.log   # stdout+stderr -> this file

# --- environment (weights pre-cached on the login node; no network from compute) ---
export HF_HOME=/projects/standard/gini/shared/munda057/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/.bashrc
conda activate qwen_test                 # env with torch + transformers 4.51.3

# pre-filled to the warning-sign trip (the one in your last run). Change if needed.
TRIP="$HOME/SignNav/signnav_scripts/datasets/extracted/rosbag2_keller_c12_warning_sign"
GOAL="Restrooms"

# run from the experiments dir so `signnav_reasoner` imports and convert/ resolves
cd ~/SignNav/signnav_scripts/experiments

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== start: $(date '+%H:%M:%S') ==="

# --- single run, decouple on, force off, masking off (model loads once) ---
# run_decouple_test.py must live in this experiments dir alongside signnav_reasoner/
python run_decouple_test.py "$TRIP" "$GOAL"

echo "=== end: $(date '+%H:%M:%S') ==="
