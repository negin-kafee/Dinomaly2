#!/usr/bin/env bash
#SBATCH --job-name=dino_apply
#SBATCH --partition=cpuq
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02d_apply_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02d_apply_%j.err

# Apply the selected map-normalization quantiles (config NORM_START/NORM_END) to
# RAW per-subject predictions in place, for each modality in INFER_MODALITIES.
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"
source "${ENV_ACTIVATE}"
cd "${REPO}"

CALIB_VALUES="${SAVE_DIR}/calib_values.npy"
for MOD in ${INFER_MODALITIES}; do
    echo "--- apply norm modality=${MOD} (start_q=${NORM_START} end_q=${NORM_END}) ---"
    python tools/apply_norm.py \
        --raw_dir "${SAVE_DIR}/infer_${MOD}/raw_predictions" \
        --calib_values "${CALIB_VALUES}" \
        --start_q "${NORM_START}" \
        --end_q "${NORM_END}" \
        --out_json "${SAVE_DIR}/infer_${MOD}/norm_applied.json"
done
echo "=== apply done ==="
