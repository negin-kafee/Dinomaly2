#!/usr/bin/env bash
#SBATCH --job-name=dino_infer_raw
#SBATCH --partition=gpuq
#SBATCH --time=30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02b_infer_raw_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02b_infer_raw_%j.err

# One-off: BraTS inference producing RAW (un-normalized) per-subject predictions
# plus a dumped healthy calibration-value array, so normalization quantiles can
# be swept offline (tools/sweep_norm.py) and applied (tools/apply_norm.py).
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"

echo "=== Node: $(hostname)  Job: ${SLURM_JOB_ID}  DATASET=${DATASET}  MODS=${INFER_MODALITIES} (RAW) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

source "${ENV_ACTIVATE}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

CKPT_USE="${CKPT}"
[ -f "${CKPT_USE}" ] || CKPT_USE="${CKPT_LAST}"
echo "Using checkpoint: ${CKPT_USE}"

CALIB_VALUES="${SAVE_DIR}/calib_values.npy"

for MOD in ${INFER_MODALITIES}; do
    OUT_DIR="${SAVE_DIR}/infer_${MOD}"
    echo "--- RAW inference modality=${MOD} -> ${OUT_DIR} ---"
    python "${REPO}/inference_brain.py" \
        --brats_root "${BRATS_ROOT}" \
        --modality "${MOD}" \
        --ckpt "${CKPT_USE}" \
        --out_dir "${OUT_DIR}" \
        --backbone "${BACKBONE}" \
        --crop_size "${CROP_SIZE}" \
        --save_size 256 \
        --batch_size "${INFER_BATCH_SIZE}" \
        --num_workers 4 \
        --calib_data_path "${DATA_PATH}" \
        --calib_slices "${CALIB_SLICES}" \
        --dump_calib_values "${CALIB_VALUES}" \
        --no_normalize
done

echo "=== RAW inference done; calib values at ${CALIB_VALUES} ==="
