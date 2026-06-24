#!/bin/bash
#SBATCH --job-name=signnav_notice
#SBATCH --partition=msigpu          # <-- EDIT if needed (check `sinfo`)
#SBATCH --gres=gpu:a100:1
#SBATCH --account=gini
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL --mail-user=munda057@umn.edu
#SBATCH --time=00:40:00
#SBATCH --output=signnav_notice_%j.log   # stdout+stderr -> this file

# --- environment (weights pre-cached on the login node; no network from compute) ---
export HF_HOME=/projects/standard/gini/shared/munda057/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/.bashrc
conda activate qwen_test                 # env with torch + transformers 4.51.3

# >>> EDIT THIS: the Restroom-Closed trip dir (must contain frames/, odom.csv, frame_index.csv)
TRIP="$HOME/SignNav/signnav_scripts/datasets/extracted/rosbag2_keller_c12_warning_sign"
GOAL="Restrooms"

# run from the experiments dir so `signnav_reasoner` imports and convert/ resolves
cd ~/SignNav/signnav_scripts/experiments

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== start: $(date '+%H:%M:%S') ==="


# sanity: make sure the trip dir was edited and exists
if [ ! -d "$TRIP/frames" ]; then
    echo "ERROR: '$TRIP/frames' not found — edit the TRIP variable in this script."
    exit 1
fi

# --- Test A + Test B (model loads once inside the driver) ---
# run_notice_test.py must live in this experiments dir alongside signnav_reasoner/
python run_notice_test.py "$TRIP" "$GOAL"

echo "=== end: $(date '+%H:%M:%S') ==="