#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --mem=256000

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"

# ============================================================
# Job array
#
# 10 data configurations × 3 seeds = 30 experiments
# Maximum 10 experiments running simultaneously
# ============================================================

#SBATCH --array=0-29%10

#SBATCH --job-name=aug-ablation

# Separate output/error file for every array task
#SBATCH --output=aug-ablation_%A_%a.out
#SBATCH --error=aug-ablation_%A_%a.err


# ============================================================
# Shell settings
# ============================================================

set -x
set -euo pipefail

# ============================================================
# Seeds
# ============================================================

SEEDS=(0 42 12345)

# ============================================================
# Data configurations
#
# Add/remove configs here.
#
# IMPORTANT:
# The number of array tasks must be:
#
#     number of data configs × number of seeds
#
# ============================================================

DATA_CONFIGS=(
    "../python-chebai-graph/configs/data/augmented/aug-ablation/FGN.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/FGN+E.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/FGN+E+WGN.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/FGN+WGN.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/gn_wall_fgwa_nfge.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/gn_wall_fgwa_wfge.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/gnwa_fgwa_nfge.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/gnwa_fgwa_wfge.yml"
    "../python-chebai-graph/configs/data/augmented/aug-ablation/WGN.yml"
    "../python-chebai-graph/configs/data/chebi50_baseline.yml"
)

# ============================================================
# Determine data config and seed from array task ID
# ============================================================

# Array task ordering:
# All data configurations are run for seed 0 first,
# followed by all data configurations for seed 42,
# and finally all data configurations for seed 12345.
# This is intentionally because the data is generated in first run for each data configuration.
# Then the following runs for the same data configuration can re-use the same generated data.

NUM_SEEDS=${#SEEDS[@]}
NUM_DATA_CONFIGS=${#DATA_CONFIGS[@]}

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_DATA_CONFIGS))
DATA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_DATA_CONFIGS))

DATA_CONFIG=${DATA_CONFIGS[$DATA_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# ============================================================
# Create experiment name
# ============================================================

DATA_NAME=$(basename "$DATA_CONFIG" .yml)
RUN_NAME="${DATA_NAME}_s${SEED}"

# ============================================================
# Print job information
# ============================================================

echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Job ID:    $SLURM_ARRAY_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Data index:      $DATA_INDEX"
echo "Seed index:      $SEED_INDEX"
echo "Data config:     $DATA_CONFIG"
echo "Data name:       $DATA_NAME"
echo "Seed:            $SEED"
echo "Run name:        $RUN_NAME"
echo "Node:            $(hostname)"
echo "Date:            $(date)"
echo "============================================================"

echo "GPU information:"
nvidia-smi

echo "============================================================"

# ============================================================
# Temporary directory
# ============================================================

export TMPDIR=/home/staff/a/akhedekar/atmp_dir/

# ============================================================
# Activate Python environment
# ============================================================

source /home/staff/a/akhedekar/python-chebai-graph/.venv/bin/activate

# ============================================================
# Set working directory
# ============================================================

CHEBAI_DIR="/home/staff/a/akhedekar/python-chebai"
cd "$CHEBAI_DIR"
export SSL_CERT_FILE=$(python -m certifi)

# ============================================================
# Check selected configuration
# ============================================================

if [[ ! -f "$DATA_CONFIG" ]]; then
    echo "ERROR: Data configuration does not exist:"
    echo "$DATA_CONFIG"
    exit 1
fi

# ============================================================
# Run training
# ============================================================

python -m chebai fit \
    --trainer=configs/training/default_trainer.yml \
    --trainer.logger=configs/training/wandb_logger.yml \
    --model=../python-chebai-graph/configs/model/baselines/gat.yml \
    --model.train_metrics=configs/metrics/micro-macro-f1.yml \
    --model.test_metrics=configs/metrics/micro-macro-f1.yml \
    --model.val_metrics=configs/metrics/micro-macro-f1.yml \
    --data="$DATA_CONFIG" \
    --data.init_args.batch_size=64 \
    --trainer.accumulate_grad_batches=1 \
    --data.init_args.num_workers=10 \
    --model.pass_loss_kwargs=false \
    --data.init_args.chebi_version=252 \
    --trainer.min_epochs=200 \
    --trainer.max_epochs=200 \
    --model.criterion=configs/loss/bce_unweighted.yml \
    --trainer.logger.init_args.name="$RUN_NAME" \
    --model.init_args.optimizer_kwargs.lr=0.002 \
    --data.init_args.splits_file_path=data/chebi_v252/ChEBI50/processed/splits.csv \
    --trainer.logger.init_args.tags='["augmented_paper","aug-ablation"]' \
    --seed_everything="$SEED"


# ============================================================
# Finished
# ============================================================

echo "============================================================"
echo "Training finished successfully"
echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Data config:     $DATA_CONFIG"
echo "Seed:            $SEED"
echo "Run name:        $RUN_NAME"
echo "Node:            $(hostname)"
echo "Date:            $(date)"
echo "============================================================"
