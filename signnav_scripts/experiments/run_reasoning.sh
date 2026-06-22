#!/bin/bash
#SBATCH --job-name=signnav_reason
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --account=gini
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL --mail-user=munda057@umn.edu
#SBATCH --time=01:00:00
#SBATCH --output=signnav_reason_%j.log

# --- environment ---
export HF_HOME=/projects/standard/gini/shared/munda057/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/.bashrc
conda activate qwen_test

cd ~/SignNav

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== start: $(date '+%H:%M:%S') ==="

# --- run sequential live reasoning (Qwen2.5-VL-72B, 4-bit, on the H100) ---
# EDIT --trip and --goal for the trip you want. Start with the left-turn trip.
python signnav_scripts/experiments/live_reason_trip.py \
    --trip signnav_scripts/datasets/extracted/rosbag2_keller_29 \
    --goal "room 2-130" \
    --every 8 \
    --model 32b \
    --precision bf16 \
    --out reasoning_keller40_72b.txt

python signnav_scripts/experiments/neuro_symbolic_sign.py \
    --trip signnav_scripts/datasets/extracted/rosbag2_keller_29 \
    --goal "room 2-130" \
    --every 8 \
    --out neurosym_keller29.txt

echo "=== end: $(date '+%H:%M:%S') ==="