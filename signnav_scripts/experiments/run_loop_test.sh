#!/bin/bash
#SBATCH --job-name=signnav_loop_test
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --account=gini
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL --mail-user=munda057@umn.edu
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
# Uses REAL models: YOLO (signs) + GroundingDINO (hazards) + Qwen-7B (read+reason).
# 7B to mirror the Jetson's placeholder model.
cd signnav_scripts/experiments
python -m signnav_reasoner.loop \
    --frames ../datasets/extracted/rosbag2_keller_c1/frames \
    --goal "room 4-205" \
    --every 5 \
    2>&1

echo "=== end: $(date '+%H:%M:%S') ==="