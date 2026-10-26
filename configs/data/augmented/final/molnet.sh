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

set -Eeuo pipefail
trap 'status=$?; echo "ERROR: command failed with status ${status} at line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2; exit "$status"' ERR

# ============================================================
# Local paths
# ============================================================
CHEBAI_DIR="/home/aditya/python-chebai"
CHEBAI_GRAPH_DIR="/home/aditya/python-chebai-graph"
LOG_DIR="$CHEBAI_DIR/logs"
LOG_FILE="$LOG_DIR/molnet_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$LOG_DIR"

# Keep output visible in the terminal while recording stdout, stderr, and
# Bash trace output in a timestamped text file.
exec > >(tee -a "$LOG_FILE") 2>&1
set -x
echo "Writing MolNet run log to: $LOG_FILE"


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
RESULTS_FILE="$CHEBAI_DIR/logs/molnet_test_results.tsv"
mkdir -p "$(dirname "$RESULTS_FILE")"
printf 'dataset\tseed\tmetric\tvalue\tcheckpoint\n' > "$RESULTS_FILE"

for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        case "$DATASET" in
            BBBP)
                TRAINER="configs/training/binary_trainer.yml"
                METRICS="configs/metrics/binary-f1-roc-auc.yml"
                CALLBACKS="configs/training/binary_callbacks.yml"
                CHECKPOINT_PREFIX="best_roc-auc_"
                METRICS_TO_REPORT=(test_roc-auc test_f1)
                DROPOUT="0.5"
                LR_GNN="0.0005"
                LR_LINEAR="0.001"
                ;;
            BACE)
                TRAINER="configs/training/binary_trainer.yml"
                METRICS="configs/metrics/binary-f1-roc-auc.yml"
                CALLBACKS="configs/training/binary_callbacks.yml"
                CHECKPOINT_PREFIX="best_roc-auc_"
                METRICS_TO_REPORT=(test_roc-auc test_f1)
                DROPOUT="0.5"
                LR_GNN="0.001"
                LR_LINEAR="0.001"
                ;;
            SIDER)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                CALLBACKS="configs/training/roc-auc_callbacks.yml"
                CHECKPOINT_PREFIX="best_himol-marco-roc-auc_"
                METRICS_TO_REPORT=(test_himol-marco-roc-auc test_micro-f1 test_macro-f1)
                DROPOUT="0.5"
                LR_GNN="0.0005"
                LR_LINEAR="0.001"
                ;;
            ClinTox)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                CALLBACKS="configs/training/roc-auc_callbacks.yml"
                CHECKPOINT_PREFIX="best_himol-marco-roc-auc_"
                METRICS_TO_REPORT=(test_himol-marco-roc-auc test_micro-f1 test_macro-f1)
                DROPOUT="0.5"
                LR_GNN="0.001"
                LR_LINEAR="0.001"
                ;;
            Tox21)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                CALLBACKS="configs/training/roc-auc_callbacks.yml"
                CHECKPOINT_PREFIX="best_himol-marco-roc-auc_"
                METRICS_TO_REPORT=(test_himol-marco-roc-auc test_micro-f1 test_macro-f1)
                DROPOUT="0.7"
                LR_GNN="0.0005"
                LR_LINEAR="0.001"
                ;;
            ToxCast)
                TRAINER="configs/training/default_trainer.yml"
                METRICS="configs/metrics/micro-macro-f1-roc-auc.yml"
                CALLBACKS="configs/training/roc-auc_callbacks.yml"
                CHECKPOINT_PREFIX="best_himol-marco-roc-auc_"
                METRICS_TO_REPORT=(test_himol-marco-roc-auc test_micro-f1 test_macro-f1)
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

        RUN_MARKER=$(mktemp)
        CALLBACK_ARGS=(--trainer.callbacks="$CALLBACKS")

        python -m chebai fit \
            --trainer="$TRAINER" \
            --trainer.logger=configs/training/wandb_logger.yml \
            "--trainer.logger.init_args.tags=[\"augmented_paper\", \"classif-head\", \"aug-ablation\",\"$DATASET_LOWER\"]" \
            "${CALLBACK_ARGS[@]}" \
            --model="$CHEBAI_GRAPH_DIR/configs/model/augmented/pooling/no_pooling/gat.yml" \
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
            --model.init_args.n_linear_layers=2 \

        CHECKPOINT_PATH=$(
            find "$CHEBAI_DIR/logs" -type f \
                -name "${CHECKPOINT_PREFIX}*.ckpt" \
                -newer "$RUN_MARKER" -print | sort | tail -n 1
        )
        rm -f "$RUN_MARKER"
        if [[ -z "$CHECKPOINT_PATH" ]]; then
            echo "Could not find best checkpoint for $RUN_NAME" >&2
            exit 1
        fi

        TEST_NAME="${RUN_NAME}_test"
        TEST_MARKER=$(mktemp)
        python -m chebai test \
            --trainer="$TRAINER" \
            --trainer.logger=configs/training/csv_logger.yml \
            --trainer.logger.init_args.name="$TEST_NAME" \
            --trainer.devices=1 \
            --trainer.num_nodes=1 \
            --ckpt_path="$CHECKPOINT_PATH" \
            --model="$CHEBAI_GRAPH_DIR/configs/model/augmented/pooling/no_pooling/gat.yml" \
            --model.test_metrics="$METRICS" \
            --data="$CHEBAI_GRAPH_DIR/configs/data/augmented/final/${DATASET}_final_augmented.yml" \
            --data.init_args.batch_size=32 \
            --data.init_args.num_workers=10 \
            --model.pass_loss_kwargs=false \
            --model.criterion=configs/loss/bce_unweighted.yml \
            --data.init_args.splits_file_path="data/${DATASET}:MNClassification/${DATASET}/processed/splits.csv" \
            --data.init_args.tokens_directory=_shared \
            --model.init_args.n_linear_layers=2 \

        TEST_METRICS_FILE=$(
            find "$CHEBAI_DIR/logs/$TEST_NAME" -type f \
                -name metrics.csv -newer "$TEST_MARKER" -print | sort | tail -n 1
        )
        rm -f "$TEST_MARKER"
        if [[ ! -f "$TEST_METRICS_FILE" ]]; then
            echo "Could not find fresh test metrics for $TEST_NAME" >&2
            exit 1
        fi
        METRIC_VALUES=$(python - "$TEST_METRICS_FILE" "${METRICS_TO_REPORT[@]}" <<'PY'
import csv
import sys

metrics_path, *metric_names = sys.argv[1:]
values = {metric_name: None for metric_name in metric_names}
with open(metrics_path, newline="") as metrics_file:
    for row in csv.DictReader(metrics_file):
        for metric_name in metric_names:
            if row.get(metric_name):
                values[metric_name] = row[metric_name]
for metric_name in metric_names:
    value = values[metric_name]
    if value is None:
        raise SystemExit(f"Metric {metric_name!r} not found in {metrics_path}")
    print(f"{metric_name}\t{value}")
PY
)
        while IFS=$'\t' read -r TEST_METRIC TEST_VALUE; do
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$DATASET" "$SEED" "$TEST_METRIC" "$TEST_VALUE" "$CHECKPOINT_PATH" \
                >> "$RESULTS_FILE"
            echo "Test $TEST_METRIC: $TEST_VALUE"
        done <<< "$METRIC_VALUES"
        echo "============================================================"
        echo "Finished run: $RUN_NAME"
        echo "============================================================"
    done
done

echo "============================================================"
echo "All experiments finished successfully"
echo "============================================================"

python - "$RESULTS_FILE" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

results = defaultdict(list)
with open(sys.argv[1], newline="") as results_file:
    for row in csv.DictReader(results_file, delimiter="\t"):
        results[(row["dataset"], row["metric"])].append(float(row["value"]))

print("\nDataset test performance (best validation checkpoint per seed)")
print("dataset\tmetric\tmean\tsample_sd\tn")
for (dataset, metric), values in sorted(results.items()):
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values) if len(values) > 1 else math.nan
    print(f"{dataset}\t{metric}\t{mean:.6f}\t{sample_sd:.6f}\t{len(values)}")
PY
