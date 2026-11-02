#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --mem=256000

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"

# 3 experiments, maximum 3 running at the same time
#SBATCH --array=0-2%3

#SBATCH --job-name=chebi_gatv2_final

# Separate log files for every array task
#SBATCH --output=chebi_gatv2_final_%A_%a.out
#SBATCH --error=chebi_gatv2_final_%A_%a.err


# ============================================================
# Shell settings
# ============================================================
set -x
set -euo pipefail
# ============================================================
# Define seeds
# ============================================================

SEEDS=(0 42 12345)

if [[ -z "${SEEDS[$SLURM_ARRAY_TASK_ID]}" ]]; then
    echo "ERROR: Invalid array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="chebi_gatv2_final_${SEED}"

# ============================================================
# Print job information
# ============================================================

echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Job ID:    $SLURM_ARRAY_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
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
# Run training
# ============================================================

python -m chebai fit \
    --trainer=configs/training/default_trainer.yml \
    --trainer.logger=configs/training/wandb_logger.yml \
    --model=../python-chebai-graph/configs/model/augmented/pooling/amg_pool/gat.yml \
    --model.train_metrics=configs/metrics/micro-macro-f1.yml \
    --model.test_metrics=configs/metrics/micro-macro-f1.yml \
    --model.val_metrics=configs/metrics/micro-macro-f1.yml \
    --data=../python-chebai-graph/configs/data/augmented/final/chebi50_final_augmented.yml \
    --data.init_args.batch_size=64 \
    --data.init_args.num_workers=10 \
    --data.init_args.chebi_version=252 \
    --model.pass_loss_kwargs=false \
    --trainer.min_epochs=200 \
    --trainer.max_epochs=200 \
    --model.criterion=configs/loss/bce_weighted.yml \
    --trainer.logger.init_args.name="$RUN_NAME" \
    --data.init_args.splits_file_path=data/chebi_v252/ChEBI50/processed/splits.csv \
    --seed_everything="$SEED" \
    --trainer.logger.init_args.tags='["augmented_paper", "chebi", "final:gnwa_fgwa_wfge"]' \

echo "============================================================"
echo "Training finished"
echo "Seed: $SEED"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "============================================================"
