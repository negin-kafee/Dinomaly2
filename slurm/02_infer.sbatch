#!/usr/bin/env bash
#SBATCH --job-name=dino_infer
#SBATCH --partition=gpuq
#SBATCH --time=30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02_infer_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02_infer_%j.err

# Dinomaly2 BraTS inference -> per-subject raw predictions (.npz + optional .nii.gz).
# Runs for each modality in INFER_MODALITIES. Single GPU is plenty for inference.
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"

echo "=== Node: $(hostname)  Job: ${SLURM_JOB_ID}  DATASET=${DATASET}  MODS=${INFER_MODALITIES} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

source "${ENV_ACTIVATE}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

CKPT_USE="${CKPT}"
[ -f "${CKPT_USE}" ] || CKPT_USE="${CKPT_LAST}"
echo "Using checkpoint: ${CKPT_USE}"

for MOD in ${INFER_MODALITIES}; do
    OUT_DIR="${SAVE_DIR}/infer_${MOD}"
    echo "--- inference modality=${MOD} -> ${OUT_DIR} ---"
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
        --save_nifti
done

echo "=== Inference done ==="
