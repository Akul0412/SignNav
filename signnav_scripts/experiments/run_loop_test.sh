#!/bin/bash
#SBATCH --job-name=signnav_loop_test
#SBATCH --partition=msigpu                 
#SBATCH --gres=gpu:a100:1
#SBATCH --account=gini
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=signnav_loop_test_%j.log

export HF_HOME=/projects/standard/gini/shared/munda057/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/.bashrc
conda activate qwen_test

cd ~/SignNav

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== start: $(date '+%H:%M:%S') ==="

# Test the full loop on a real extracted trip, every 5th frame.
# Sign detection: OpenCV dark-panel HEURISTIC (use_yolo=False) — no torchvision needed,
#   and YOLO can't see indoor signs anyway. Hazards: GroundingDINO. Read+reason: Qwen-7B fp16.

cd signnav_scripts/experiments

python -m signnav_reasoner.loop \
    --journey ../datasets/extracted/rosbag2_keller_c11_long_horizon \
    --goal "Elevators"

echo "=== end: $(date '+%H:%M:%S') ==="
