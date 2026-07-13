#!/usr/bin/env bash
#SBATCH --job-name=dino_sweep
#SBATCH --partition=cpuq
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02c_sweep_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/02c_sweep_%j.err

# Offline sweep of map-normalization quantiles on RAW predictions.
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"
source "${ENV_ACTIVATE}"
cd "${REPO}"

CALIB_VALUES="${SAVE_DIR}/calib_values.npy"
for MOD in ${INFER_MODALITIES}; do
    echo "===== ${MOD} sweep ====="
    python tools/sweep_norm.py \
        --raw_dir "${SAVE_DIR}/infer_${MOD}/raw_predictions" \
        --calib_values "${CALIB_VALUES}" \
        --max_subjects 60 --stride 2 \
        --start_qs 0.5 0.9 0.95 \
        --end_qs 0.95 0.99 0.995 0.999 0.9999 1.0
    echo
done
echo "=== sweep done ==="
