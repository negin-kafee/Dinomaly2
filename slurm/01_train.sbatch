#!/usr/bin/env bash
#SBATCH --job-name=dino_train
#SBATCH --partition=gpuq
#SBATCH --time=30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/01_train_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/01_train_%j.err

# Dinomaly2 brain-MRI training (DDP via torchrun). Resumes automatically from
# last.pth if the job is interrupted and re-queued. NGPU and --gres are set at
# submit time by submit_pipeline.sh.
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"

NGPU="${NGPU:-${SLURM_GPUS_ON_NODE:-1}}"

echo "=== Node: $(hostname)  Job: ${SLURM_JOB_ID}  NGPU=${NGPU}  DATASET=${DATASET} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

source "${ENV_ACTIVATE}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

RDZV_PORT=$((20000 + SLURM_JOB_ID % 20000))

torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
    --master_port="${RDZV_PORT}" \
    "${REPO}/dinomaly_brain.py" \
    --data_path "${DATA_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --save_name "${SAVE_NAME}" \
    --backbone "${BACKBONE}" \
    --image_size "${IMAGE_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --total_iters "${TOTAL_ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --ckpt_interval "${CKPT_INTERVAL}" \
    --slice_cache "${SLICE_CACHE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}"

echo "=== Training done. Checkpoints in ${SAVE_DIR}/${SAVE_NAME} ==="
ls -la "${SAVE_DIR}/${SAVE_NAME}" || true
