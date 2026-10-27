#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --threads-per-core=1
#SBATCH --mem=256000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --job-name=zinc
#SBATCH --output=zinc.out
#SBATCH --error=zinc.err


# ============================================================
# Shell settings
# ============================================================
set -x
# ============================================================
# Define seeds
# ============================================================

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
    --trainer=configs/training/zinc_pretrainer.yml \
    --trainer.limit_val_batches=0 \
    --trainer.logger=configs/training/wandb_logger.yml \
    --trainer.logger.init_args.tags='["augmented_paper","zinc:pretraining","final:gnwa_fgwa_wfge"]' \
    --trainer.logger.init_args.name=zinc15pretrain_gat_amg_augmented \
    --model=../python-chebai-graph/configs/model/pretraining/gat_msp.yml \
    --data=../python-chebai-graph/configs/data/augmented/pretraining/zinc15.yml \
    --data.init_args.batch_size=32 \
    --model.pass_loss_kwargs=true \
    --data.init_args.num_workers=10 \
    --trainer.logger.init_args.name=zinc15pretrain_gat_amg_augmented \
    --data.init_args.splits_file_path=data/Zinc15:Pretrain/Zinc15/processed/splits.csv \
