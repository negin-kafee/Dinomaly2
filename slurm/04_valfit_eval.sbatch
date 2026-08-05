#!/usr/bin/env bash
#SBATCH --job-name=dino_valfit
#SBATCH --partition=cpuq
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/04_valfit_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/04_valfit_%j.err

# Leak-free protocol: fit normalization quantile + threshold on the 24 val
# subjects, apply to the 312 test subjects, evaluate at the fixed val threshold.
set -euo pipefail
source "/project/detectanomaly/training/repos/Dinomaly2/slurm/config.sh"
source "${ENV_ACTIVATE}"
cd "${REPO}"

VAL_BANK="/project/detectanomaly/training/shared/brats_val"
VAL_IDS="${VAL_BANK}/splits/BraTS_T2_val24.csv"
TEST_IDS="${VAL_BANK}/312_test_subject_ids.txt"
CALIB_VALUES="${SAVE_DIR}/calib_values.npy"

for MOD in ${INFER_MODALITIES}; do
    RAW_DIR="${SAVE_DIR}/infer_${MOD}/raw_predictions"
    JSON="${SAVE_DIR}/valfit_${MOD}.json"
    TEST_DIR="${SAVE_DIR}/valfit_test_${MOD}"
    EVAL_DIR="${SAVE_DIR}/eval_valfit_${MOD}"
    mkdir -p "${TEST_DIR}" "${EVAL_DIR}"

    echo "=========== VAL-FIT ${DATASET} ${MOD} ==========="
    python tools/valfit.py \
        --raw_dir "${RAW_DIR}" \
        --calib_values "${CALIB_VALUES}" \
        --val_ids "${VAL_IDS}" \
        --test_ids "${TEST_IDS}" \
        --out_json "${JSON}" \
        --start_qs 0.5 0.9 \
        --end_qs 0.99 0.999 0.9999 0.99999 0.999999 1.0

    SQ=$(python -c "import json;print(json.load(open('${JSON}'))['start_q'])")
    EQ=$(python -c "import json;print(json.load(open('${JSON}'))['end_q'])")
    THR=$(python -c "import json;print(json.load(open('${JSON}'))['val_thr'])")
    CLIP=$(python -c "import json;print(json.load(open('${JSON}'))['clip_frac'])")
    echo "chosen start_q=${SQ} end_q=${EQ} val_thr=${THR} clip@1.0=${CLIP}"

    echo "--- apply to TEST subjects only ---"
    python tools/apply_norm.py \
        --raw_dir "${RAW_DIR}" \
        --calib_values "${CALIB_VALUES}" \
        --start_q "${SQ}" --end_q "${EQ}" \
        --ids_file "${TEST_IDS}" \
        --out_dir "${TEST_DIR}"

    N=$(ls "${TEST_DIR}"/*.npz 2>/dev/null | wc -l)
    echo "TEST npz written: ${N} (expected 312)"
    if [ "${N}" -ne 312 ]; then echo "ERROR: test count != 312"; exit 1; fi

    echo "--- eval at fixed val threshold ${THR} ---"
    python "${EVAL_SCRIPT}" \
        --raw_dir "${TEST_DIR}" \
        --threshold "${THR}" \
        --output_dir "${EVAL_DIR}"
    SUMMARY="${EVAL_DIR}/metrics_summary.csv"
    if [ -f "${SUMMARY}" ]; then
        cp "${SUMMARY}" "${OUT_BASE}/summary_valfit_${DATASET}_${MOD}.csv"
        echo "Summary -> ${OUT_BASE}/summary_valfit_${DATASET}_${MOD}.csv"
    fi
done

echo "=== val-fit eval done ==="
