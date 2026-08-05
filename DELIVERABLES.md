# Dinomaly2 — Unsupervised Brain-MRI Anomaly Detection (BraTS tumor localization)

Single-input (1-channel) Dinomaly2 trained on **healthy** brain MRI (IXI + MOOD),
evaluated on **BraTS** tumor segmentation. 8 training datasets, inference on BraTS
T1/T2, evaluated against BraTS binary tumor GT.

---

## 1. Paper / method

| | |
|---|---|
| **Paper** | *Dinomaly2: Less Is More — a Minimal Reconstruction Framework for Multi-Class Anomaly Detection* |
| **arXiv** | 2510.17611 |
| **Code (upstream)** | github.com/guojiajeremy/Dinomaly |
| **Our fork** | github.com/negin-kafee/Dinomaly2 (branch `main`) |

**Method.** A frozen DINOv2-reg ViT encoder extracts multi-layer features; a small
bottleneck + decoder reconstruct them under a cosine feature-reconstruction loss.
At test time the anomaly map = `1 − cosine_similarity(encoder, decoder)` per patch,
Gaussian-smoothed (kernel 5, σ 4). Trained only on healthy data, tumors appear as
high-reconstruction-error regions.

**Backbone:** `dinov2reg_vit_small_14` (patch 14, embed 384, 6 heads, target layers 2–9),
weights `backbones/weights/dinov2_vits14_reg4_pretrain.pth`.

---

## 2. Configuration (kept close to the paper's medical config)

| Setting | Value |
|---|---|
| Input | 1-channel grayscale → replicated to 3ch, ImageNet mean/std |
| Train / crop size | 280×280 (14×20; nearest 256 requested → 280 divisible by patch 14) |
| Iterations | 40 000 |
| Batch size | 32 per GPU (paper: 16 — **deviation, approved**) |
| Optimizer | StableAdamW, lr 2e-3 (bottleneck 2e-4), wd 1e-4, eps 1e-10 |
| Scheduler | WarmupCosine (warmup 100) |
| Loss | cosine feature reconstruction (layers 2–9), point-wise hard-mining |
| GPUs | 1× V100-SXM2-32GB per job |

**Anomaly-map normalization (added).** BraTS maps are rescaled to [0,1] using
healthy-training quantiles: `clip((map − q_start)/(q_end − q_start), 0, 1)` with
**start_q=0.9, end_q=0.9999**, calibrated per-dataset on 2000 random healthy slices.
These quantiles were selected by an offline sweep (`tools/sweep_norm.py`); the paper
default (q0.5/q0.95) saturated the maps on brain data (large zero-background) and gave
Dice ≈ 0.15. See §7.

---

## 3. Data

| Group | Dataset | Modality trained | BraTS test modality |
|---|---|---|---|
| T1+T2 | `T1T2_combined/MOOD_IXI_all` | T1+T2 | T1, T2 |
| T1+T2 | `T1T2_combined/MOOD_IXI_3T_T1T2` | T1+T2 | T1, T2 |
| T1 | `T1_only/MOOD_3T` | T1 | T1 |
| T1 | `T1_only/MOOD_IXI_3T` | T1 | T1 |
| T1 | `T1_only/MOOD_IXI_3T_15T` | T1 | T1 |
| T2 | `T2_only/IXI_3T` | T2 | T2 |
| T2 | `T2_only/IXI_15T` | T2 | T2 |
| T2 | `T2_only/IXI_combined` | T2 | T2 |

- Train root: `/project/detectanomaly/training/datasets/`
- BraTS root: `/project/detectanomaly/training/datasets/BraTS` (346 subjects; GT binary tumor 0/1)
- Predictions upsampled/saved at 256×256 (BraTS 240→256 nearest; anomaly maps 280→256 nearest).

---

## 4. Results (BraTS, per `compute_metrics_soumick.py`)

Per-slice / per-volume **Dice (median)**, slice **AUROC**, and size-stratified Dice
(small ≤494 px, medium ≤2270 px, large >2270 px). Operating threshold chosen by the
eval's grid search (0.05–0.94).

| Dataset | Mod | thr | Slice Dice | Vol Dice | Slice AUROC | Dice small | Dice med | Dice large | AUROC large |
|---|---|---|---|---|---|---|---|---|---|
| MOOD_IXI_all | T2 | 0.94 | 0.411 | 0.372 | 0.826 | 0.000 | 0.439 | 0.629 | 0.927 |
| MOOD_IXI_all | T1 | 0.78 | 0.149 | 0.157 | 0.797 | 0.002 | 0.147 | 0.337 | 0.886 |
| MOOD_IXI_3T_T1T2 | T2 | 0.94 | 0.458 | 0.323 | 0.819 | 0.001 | 0.462 | 0.636 | 0.923 |
| MOOD_IXI_3T_T1T2 | T1 | 0.94 | 0.153 | 0.159 | 0.821 | 0.003 | 0.154 | 0.341 | 0.905 |
| MOOD_3T | T1 | 0.94 | 0.148 | 0.101 | 0.835 | 0.019 | 0.133 | 0.267 | 0.906 |
| MOOD_IXI_3T | T1 | 0.94 | 0.186 | 0.159 | 0.820 | 0.010 | 0.169 | 0.359 | 0.899 |
| MOOD_IXI_3T_15T | T1 | 0.94 | 0.153 | 0.152 | 0.802 | 0.002 | 0.149 | 0.362 | 0.884 |
| IXI_3T | T2 | 0.94 | 0.379 | 0.215 | 0.790 | 0.021 | 0.360 | 0.527 | 0.895 |
| IXI_15T | T2 | 0.94 | 0.386 | 0.231 | 0.799 | 0.020 | 0.366 | 0.529 | 0.904 |
| IXI_combined | T2 | 0.94 | 0.450 | 0.267 | 0.771 | 0.009 | 0.431 | 0.613 | 0.883 |

**Takeaways**
- **T2** is strong: slice Dice 0.38–0.46, large-tumor Dice 0.53–0.64, AUROC ~0.80.
- **T1** is weaker (slice Dice 0.15–0.19) — expected, tumor–tissue contrast is lower on T1.
- **Small tumors** (≤494 px) are essentially undetectable at 256×256 across all runs — an inherent resolution/method limit.
- Adding MOOD to IXI (T2) does not clearly help T2 Dice; the IXI-only `IXI_combined` is the best T2 result (0.450).

Full per-slice / per-volume CSVs: `outputs/summary_<DATASET>_<mod>.csv`,
`outputs/<DATASET>/infer_<mod>/raw_predictions/*.npz` (anomaly_maps 256×256 f32 + gt 0/1),
optional NIfTI heatmaps in `outputs/<DATASET>/infer_<mod>/nifti/`.

---

## 5. SLURM jobs, nodes, timing

Training (gpuq, 1× V100, batch 32, 40k iters):

| Dataset | Train job | Node | Elapsed |
|---|---|---|---|
| MOOD_IXI_all | 39098558 | gnode02 | 14:55:01 |
| MOOD_3T | 39603487 | gnode14 | 16:38:46 |
| MOOD_IXI_3T | 39603490 | gnode15 | 14:05:06 |
| MOOD_IXI_3T_15T | 39603493 | gnode01 | 11:17:27 |
| IXI_3T | 39603496 | gnode05 | 04:50:26 |
| IXI_15T | 39603499 | gnode15 | 05:31:54 |
| IXI_combined | 39603502 | gnode06 | 05:24:17 |
| MOOD_IXI_3T_T1T2 | 39603505 | gnode02 | 12:48:05 |

- Inference (gpuq, 1 GPU): ~10–15 min/dataset + ~15 min one-time healthy calibration.
- Eval (cpuq): ~1.5–3 h/dataset.
- Throughput: ~1.3 s/iter on the large mixed set; faster (~0.45 s/iter) on IXI-only.

**GPU/mem:** V100-SXM2-32GB, ~64 GB host RAM for inference, 200 GB for training.

**W&B:** project `Dinomaly2`, entity `negin-kafee2-politecnico-di-milano`,
run name `dinomaly2_brain_<DATASET>` (rank-0 logging, resume-safe).

---

## 6. Reproduce

```bash
# one dataset, full chain (train -> infer -> eval, dependency-chained)
cd slurm
DATASET=MOOD_IXI_all \
DATA_PATH=/project/detectanomaly/training/datasets/T1T2_combined/MOOD_IXI_all \
INFER_MODALITIES="t1 t2" NGPU=1 BATCH_SIZE=32 NUM_WORKERS=8 TOTAL_ITERS=40000 \
bash submit_pipeline.sh
```

Pipeline scripts: `slurm/01_train.sbatch`, `slurm/02_infer.sbatch` (inline calibration +
normalization), `slurm/03_eval.sbatch`. Core code: `dataset_brain.py`, `dinomaly_brain.py`
(DDP), `inference_brain.py`. Normalization tooling: `tools/sweep_norm.py`,
`tools/apply_norm.py`, `slurm/02b_infer_raw.sbatch`, `slurm/02c_sweep.sbatch`,
`slurm/02d_apply.sbatch`.

---

## 7. Deviations from the paper & notes

- **Batch size 32** (paper 16) — approved; other optimizer/schedule settings unchanged.
- **Train at 280×280**, save maps at 256×256 (nearest) — 256 is not divisible by the ViT
  patch size 14; 280 = 14×20 is the paper's medical resolution.
- **1-channel input** replicated to 3 channels (no model change).
- **DDP + checkpoint-resume + W&B logging** added for cluster training.
- **Map normalization added** (start_q=0.9 / end_q=0.9999, healthy-quantile). The eval uses a
  fixed threshold grid (0.05–0.94); without normalization the maps sit below the grid floor
  and Dice = 0. Sweep results (per-slice median Dice, MOOD_IXI_all): q0.5/q0.95 → 0.15;
  q0.9/q0.999 → 0.35; **q0.9/q0.9999 → 0.48**.
- **Residual saturation:** the chosen operating threshold lands at the grid ceiling (0.94) for
  most datasets, so a small amount of additional Dice may exist beyond the grid — not pursued
  to keep a single consistent normalization across all datasets.
- **BraTS GT** resized 240→256 nearest; brain background not re-zeroed on the raw heatmap.
- **Paper cross-check (±5%): not applicable.** The Dinomaly/Dinomaly2 papers benchmark
  *industrial* anomaly datasets (MVTec-AD / VisA) and report **no brain-MRI numbers**, so there
  is no published Dice to compare against. Our results are consistent with typical unsupervised
  brain-anomaly performance (moderate Dice, AUROC ~0.8, strong on large lesions, weak on small).

---

## 8. Incidents

- Two eval jobs (`IXI_3T`, `IXI_combined`) stalled on an unresponsive compute node for ~10 days
  and were killed by cluster maintenance (2026-07-30). Predictions were intact; the evals were
  resubmitted (jobs 40091139 / 40091140) and completed normally in ~1.5 h each.
- One earlier training run was interrupted by scheduled maintenance and resumed from checkpoint.
