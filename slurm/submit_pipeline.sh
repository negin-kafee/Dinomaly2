#!/usr/bin/env bash
# Submit the full Dinomaly2 brain-MRI pipeline (train -> infer -> eval) with SLURM
# dependencies. Does NOT cancel or modify any other job.
#
# Usage:
#   DATASET=MOOD_IXI_all \
#   DATA_PATH=/project/detectanomaly/training/datasets/T1T2_combined/MOOD_IXI_all \
#   INFER_MODALITIES="t1 t2" NGPU=4 BATCH_SIZE=16 \
#   bash submit_pipeline.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"
mkdir -p "${HERE}/logs"

NGPU="${NGPU:-4}"

echo "Experiment : ${DATASET}"
echo "Data path  : ${DATA_PATH}"
echo "Modalities : ${INFER_MODALITIES}"
echo "Train GPUs : ${NGPU} (per-GPU batch ${BATCH_SIZE})"

# Informational free-GPU report (no action taken on other jobs).
echo "--- gpuq free GPUs ---"
for n in $(sinfo -h -p gpuq -N -o "%N" | sort -u); do
  used=$(scontrol show node "$n" 2>/dev/null | grep -oP 'AllocTRES=.*gres/gpu=\K[0-9]+' || true)
  used=${used:-0}; echo "  $n: free=$((4-used))/4"
done

export DATASET DATA_PATH INFER_MODALITIES BATCH_SIZE NUM_WORKERS TOTAL_ITERS \
       BACKBONE IMAGE_SIZE CROP_SIZE SAVE_NAME NGPU

# 1) training (request NGPU on a single node)
TRAIN_ID=$(sbatch --parsable \
    --gres=gpu:${NGPU} \
    --job-name="dtr_${DATASET}" \
    --export=ALL \
    "${HERE}/01_train.sbatch")
echo "Submitted training      : ${TRAIN_ID}"

# 2) inference (runs right after training succeeds)
INFER_ID=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_ID} \
    --job-name="din_${DATASET}" \
    --export=ALL \
    "${HERE}/02_infer.sbatch")
echo "Submitted inference     : ${INFER_ID}  (afterok:${TRAIN_ID})"

# 3) evaluation (runs right after inference succeeds)
EVAL_ID=$(sbatch --parsable \
    --dependency=afterok:${INFER_ID} \
    --job-name="dev_${DATASET}" \
    --export=ALL \
    "${HERE}/03_eval.sbatch")
echo "Submitted evaluation    : ${EVAL_ID}  (afterok:${INFER_ID})"

echo "Done. Track with: squeue -u \$USER | grep -E '${DATASET}'"
