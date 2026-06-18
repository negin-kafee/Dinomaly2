#!/usr/bin/env bash
# Shared configuration for the Dinomaly2 brain-MRI pipeline.
# Override any variable by exporting it before sourcing (e.g. in submit_pipeline.sh).

REPO="/project/detectanomaly/training/repos/Dinomaly2"
ENV_ACTIVATE="/project/detectanomaly/envs/strega-cu118-py311/bin/activate"
DATASETS_ROOT="/project/detectanomaly/training/datasets"
OUT_BASE="${REPO}/outputs"

# ---- experiment selection (override per dataset) ----
# DATASET       : short name used for folders / job names / W&B run id
# DATA_PATH     : training dataset root (must contain a raw/ folder of .nii.gz)
# INFER_MODALITIES : BraTS modalities to run inference on ("t1", "t2" or "t1 t2")
DATASET="${DATASET:-MOOD_IXI_all}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}/T1T2_combined/MOOD_IXI_all}"
INFER_MODALITIES="${INFER_MODALITIES:-t1 t2}"

BRATS_ROOT="${BRATS_ROOT:-${DATASETS_ROOT}/BraTS}"

# ---- model / training hyper-parameters (paper medical config) ----
BACKBONE="${BACKBONE:-dinov2reg_vit_small_14}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
CROP_SIZE="${CROP_SIZE:-280}"
TOTAL_ITERS="${TOTAL_ITERS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-16}"        # per-GPU
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1000}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"

# ---- W&B ----
WANDB_PROJECT="${WANDB_PROJECT:-Dinomaly2}"
WANDB_ENTITY="${WANDB_ENTITY:-negin-kafee2-politecnico-di-milano}"

# ---- derived paths ----
SAVE_NAME="${SAVE_NAME:-dinomaly2_brain_${DATASET}}"
SAVE_DIR="${OUT_BASE}/${DATASET}"
CKPT="${SAVE_DIR}/${SAVE_NAME}/model_final.pth"
CKPT_LAST="${SAVE_DIR}/${SAVE_NAME}/last.pth"
# Slice-index cache keyed by the *data folder* so smoke and full runs share it.
DATA_TAG="$(basename "${DATA_PATH}")"
SLICE_CACHE="${SLICE_CACHE:-${OUT_BASE}/cache/slice_index_${DATA_TAG}.json}"
EVAL_SCRIPT="/project/detectanomaly/training/repos/REFLECT/compute_metrics_soumick.py"
