#!/bin/bash
#SBATCH --job-name=signnav_test1
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=gini
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL --mail-user=munda057@umn.edu
#SBATCH --time=00:30:00
#SBATCH --output=signnav_test1_%j.log

# --- environment ---
# HF cache in project space (weights pre-downloaded here on the login node)
export HF_HOME=/projects/standard/gini/shared/munda057/hf_cache
# don't hit the network from the compute node — weights must already be cached
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/.bashrc
conda activate qwen_test                 # <-- EDIT: env with torch + latest transformers

cd ~/SignNav

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- run Test 1 ---
# EDIT the frame path and box (x y w h) to bracket a real sign in a clear frame.
# Run with NO --box first to let Qwen find the sign in the whole frame, then tighten.
echo "=== start: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
START_TS=$(date +%s)

python signnav_scripts/experiments/stress_test_teacher.py \
    --frame signnav_scripts/datasets/extracted/rosbag2_keller_24/frames/image.png

END_TS=$(date +%s)
echo "=== end: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "=== elapsed: $((END_TS - START_TS)) seconds ==="
echo "=== done ==="