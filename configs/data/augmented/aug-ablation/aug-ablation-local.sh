#!/bin/bash

# ============================================================
# Local single-GPU adaptation of ../aug-ablation-script.sh
#
# The original is a Slurm job-array script (10 data configs x 3
# seeds = 30 experiments, 10 concurrent). This version runs the
# SAME experiments serially on a single local GPU, without Slurm.
#
# Changes vs. the Slurm version:
#   - Removed all #SBATCH scheduling directives.
#   - Fixed hardcoded /home/staff/a/akhedekar paths -> local user.
#   - Replaced SLURM_ARRAY_TASK_ID array fan-out with a serial
#     loop over [seeds] x [data configs].
#
# Requires (verified on this machine):
#   - local venv:  /home/aditya/python-chebai-graph/.venv
#   - chebai CLI:  /home/aditya/python-chebai  (python -m chebai)
#   - chebai_graph src: /home/aditya/python-chebai-graph.
#   - wandb credentials in ~/.netrc (or run with a local logger).
# ============================================================

set -x
set -euo pipefail

# ============================================================
# Local paths
# ============================================================
CHEBAI_DIR="/home/aditya/python-chebai"
CHEBAI_GRAPH_DIR="/home/aditya/python-chebai-graph"


# ============================================================
# Seeds
# ============================================================
SEEDS=(0 42 12345)

# ============================================================
# Data configurations (all resolved to absolute local paths)
# ============================================================
DATA_CONFIGS=(
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/FGN.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/FGN+E.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/FGN+E+WGN.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/FGN+WGN.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/gn_wall_fgwa_nfge.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/gn_wall_fgwa_wfge.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/gnwa_fgwa_nfge.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/gnwa_fgwa_wfge.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/augmented/aug-ablation/WGN.yml"
    "$CHEBAI_GRAPH_DIR/configs/data/chebi50_baseline.yml"
)

# ============================================================
# Activate Python environment
# ============================================================
source "$CHEBAI_GRAPH_DIR/.venv/bin/activate"

# ============================================================
# Set working directory (chebai CLI looks up configs/... from cwd)
# ============================================================
cd "$CHEBAI_DIR"
export SSL_CERT_FILE=$(python -m certifi)

echo "============================================================"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "TMPDIR: $TMPDIR"
echo "GPU:"
nvidia-smi
echo "============================================================"

# ============================================================
# Serial loop over all seeds and data configurations
# ============================================================
for SEED in "${SEEDS[@]}"; do
    for DATA_CONFIG in "${DATA_CONFIGS[@]}"; do

        if [[ ! -f "$DATA_CONFIG" ]]; then
            echo "ERROR: Data configuration does not exist: $DATA_CONFIG"
            exit 1
        fi

        DATA_NAME=$(basename "$DATA_CONFIG" .yml)
        RUN_NAME="${DATA_NAME}_s${SEED}"

        echo "============================================================"
        echo "Starting run: $RUN_NAME"
        echo "Seed:   $SEED"
        echo "Config: $DATA_CONFIG"
        echo "============================================================"

        python -m chebai fit \
            --trainer=configs/training/default_trainer.yml \
            --trainer.logger=configs/training/wandb_logger.yml \
            --model="$CHEBAI_GRAPH_DIR/configs/model/baselines/gat.yml" \
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

        echo "============================================================"
        echo "Finished run: $RUN_NAME"
        echo "============================================================"
    done
done

echo "============================================================"
echo "All experiments finished successfully"
echo "============================================================"