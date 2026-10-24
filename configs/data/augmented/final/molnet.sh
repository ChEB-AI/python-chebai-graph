#!/bin/bash

# ============================================================
# Local single-GPU adaptation of ../aug-ablation-script.sh
# Run this script using:  `bash configs/data/augmented/final/molnet.sh`
#
# The original is a Slurm job-array script. This version runs the
# six MolNet data configurations x three seeds serially on one GPU.
#
# Changes vs. the Slurm version:
#   - Removed all #SBATCH scheduling directives.
#   - Fixed hardcoded /home/staff/a/akhedekar paths -> local user.
#   - Replaced SLURM_ARRAY_TASK_ID array fan-out with a serial loop
#     over [seeds] x [data configs].
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
echo "GPU:"
nvidia-smi
echo "============================================================"

# ============================================================
# Serial loop over all seeds and data configurations
# ============================================================
DATASETS=(BBBP BACE SIDER ClinTox Tox21 ToxCast)

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        case "$DATASET" in
            BBBP)
                TRAINER="configs/training/binary_trainer.yml"
                METRICS="configs/metrics/binary-f1-roc-auc.yml"
                DROPOUT="0.5"
                LR_GNN="0.0005"
                LR_LINEAR="0.001"
                ;;
            BACE)
                TRAINER="configs/training/binary_trainer.yml"
                METRICS="configs/metrics/binary-f1-roc-auc.yml"
                DROPOUT="0.5"
                LR_GNN="0.001"
                LR_LINEAR="0.001"
                ;;
            SIDER)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                DROPOUT="0.5"
                LR_GNN="0.0005"
                LR_LINEAR="0.001"
                ;;
            ClinTox)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                DROPOUT="0.5"
                LR_GNN="0.001"
                LR_LINEAR="0.001"
                ;;
            Tox21)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                DROPOUT="0.7"
                LR_GNN="0.0005"
                LR_LINEAR="0.001"
                ;;
            ToxCast)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                DROPOUT="0.7"
                LR_GNN="0.001"
                LR_LINEAR="0.001"
                ;;
            *)
                echo "Unsupported dataset: $DATASET" >&2
                exit 1
                ;;
        esac

        DATASET_LOWER=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')
        RUN_NAME="${DATASET_LOWER}_gat_amg_augmented_s${SEED}"

        echo "============================================================"
        echo "Starting run: $RUN_NAME"
        echo "Dataset: $DATASET"
        echo "Seed:    $SEED"
        echo "============================================================"

        python -m chebai fit \
            --trainer="$TRAINER" \
            --trainer.logger=configs/training/wandb_logger.yml \
            "--trainer.logger.init_args.tags=[\"augmented_paper\",\"aug-ablation\",\"$DATASET_LOWER\"]" \
            --trainer.callbacks=configs/training/roc-auc_callbacks.yml \
            --model="$CHEBAI_GRAPH_DIR/configs/model/augmented/pooling/amg_pool/gat.yml" \
            --model.train_metrics="$METRICS" \
            --model.test_metrics="$METRICS" \
            --model.val_metrics="$METRICS" \
            --data="$CHEBAI_GRAPH_DIR/configs/data/augmented/final/${DATASET}_final_augmented.yml" \
            --data.init_args.batch_size=32 \
            --data.init_args.num_workers=10 \
            --model.pass_loss_kwargs=false \
            --model.criterion=configs/loss/bce_unweighted.yml \
            --trainer.logger.init_args.name="$RUN_NAME" \
            --data.init_args.splits_file_path="data/${DATASET}:MNClassification/${DATASET}/processed/splits.csv" \
            --model.config.dropout="$DROPOUT" \
            --model.optimizer_kwargs.lr_gnn="$LR_GNN" \
            --model.optimizer_kwargs.lr_linear="$LR_LINEAR" \
            --seed_everything="$SEED" \
            --model.pretrained_checkpoint=/home/aditya/python-chebai-graph/msp_best_epoch=92_train_loss=0.0009.ckpt \
            --data.init_args.tokens_directory=_shared \


        echo "============================================================"
        echo "Finished run: $RUN_NAME"
        echo "============================================================"
    done
done

echo "============================================================"
echo "All experiments finished successfully"
echo "============================================================"
