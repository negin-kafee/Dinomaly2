#!/usr/bin/env bash
#SBATCH --job-name=dino_eval
#SBATCH --partition=cpuq
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/03_eval_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/03_eval_%j.err

# Post-hoc evaluation: runs the user's compute_metrics_soumick.py on the raw
# predictions of each inferred modality. One metrics_summary.csv per modality.
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"

echo "=== Node: $(hostname)  Job: ${SLURM_JOB_ID}  DATASET=${DATASET}  MODS=${INFER_MODALITIES} ==="
source "${ENV_ACTIVATE}"
cd "${REPO}"

for MOD in ${INFER_MODALITIES}; do
    RAW_DIR="${SAVE_DIR}/infer_${MOD}/raw_predictions"
    OUT_DIR="${SAVE_DIR}/eval_${MOD}"
    mkdir -p "${OUT_DIR}"
    echo "--- eval modality=${MOD} : ${RAW_DIR} ---"
    python "${EVAL_SCRIPT}" \
        --raw_dir "${RAW_DIR}" \
        --output_dir "${OUT_DIR}"
    # tag the summary row with experiment + modality for the mega table
    SUMMARY="${OUT_DIR}/metrics_summary.csv"
    if [ -f "${SUMMARY}" ]; then
        cp "${SUMMARY}" "${OUT_BASE}/summary_${DATASET}_${MOD}.csv"
        echo "Summary -> ${OUT_BASE}/summary_${DATASET}_${MOD}.csv"
    fi
done

echo "=== Evaluation done ==="
