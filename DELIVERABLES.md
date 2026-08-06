# Dinomaly2 — Unsupervised Brain-MRI Anomaly Detection (BraTS tumor localization)

Single-input (1-channel) Dinomaly2 trained on **healthy** brain MRI (IXI + MOOD),
evaluated on **BraTS** tumor segmentation. 8 training datasets, inference on BraTS
T1/T2, evaluated against BraTS binary tumor GT.

> **Correction notice (2026-08-06).** An earlier version of this document reported
> numbers that were **test-fit on both the normalization quantile and the operating
> threshold** — the map-normalization quantile was chosen by maximizing Dice on the
> BraTS test predictions, and the eval's `find_optimal_threshold` additionally selected
> the threshold on the same test set. Both are data leaks. The numbers below are the
> **corrected, leak-free** values: the normalization quantile **and** the threshold are
> fit **only on the 24-subject validation bank** and applied unchanged to the **312
> held-out test subjects**. As expected the corrected numbers are **lower**, and the T1
> results in particular were largely an artifact of the leak.

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
Anomaly map = `1 − cosine_similarity(encoder, decoder)` per patch, Gaussian-smoothed
(kernel 5, σ 4). Trained only on healthy data, tumors appear as high error.

**Backbone:** `dinov2reg_vit_small_14` (patch 14, embed 384, 6 heads, layers 2–9),
weights `backbones/weights/dinov2_vits14_reg4_pretrain.pth`.

---

## 2. Configuration

| Setting | Value |
|---|---|
| Input | 1-channel grayscale → replicated to 3ch, ImageNet mean/std |
| Train / crop size | 280×280 (14×20; 256 not divisible by patch 14) |
| Iterations | 40 000 |
| Batch size | 32 per GPU (paper 16 — deviation, approved) |
| Optimizer | StableAdamW, lr 2e-3 (bottleneck 2e-4), wd 1e-4, eps 1e-10 |
| Scheduler | WarmupCosine (warmup 100); loss cosine feature recon (layers 2–9) |
| GPUs | 1× V100-SXM2-32GB per job |

---

## 3. Evaluation protocol (leak-free)

Val/test split from the shared bank `/project/detectanomaly/training/shared/brats_val/`:
**24 validation subjects** (`splits/BraTS_T2_val24.csv`) disjoint from **312 test
subjects** (`312_test_subject_ids.txt`). The same subject split is used for T1 and T2
(tumor GT is modality-independent).

Per dataset-modality:
1. **Calibrate** normalization bounds on 2000 random *healthy training* slices
   (`start = quantile(healthy, start_q)`, `end = quantile(healthy, end_q)`), map
   rescaled as `clip((raw − start)/(end − start), 0, 1)`.
2. **Fit on the 24 val subjects only:** sweep `start_q ∈ {0.5, 0.9}` and
   `end_q ∈ {0.99 … 1.0}`; pick the pair with best val Dice whose val-optimal threshold
   is **interior** (< 0.98, i.e. not pinned to the grid ceiling). Record the fraction of
   pixels that clip to exactly 1.0 (saturation).
3. **Apply** the fitted `(start_q, end_q)` + **fixed val threshold** to the **312 test
   subjects** and evaluate with the user's `compute_metrics_soumick.py`. Every run is
   asserted to contain exactly **312** subjects.

Tooling: `tools/valfit.py`, `tools/apply_norm.py`, `slurm/04_valfit_eval.sbatch`.
Why higher `end_q` than before: brain slices are ~half zero-background, which drags low
quantiles down; `q0.95` clipped a large fraction of *tissue* to 1.0 (saturation),
pinning the threshold at the grid ceiling. Raising `end_q` toward the healthy max
removes the saturation and yields interior thresholds.

---

## 4. Results (312 test subjects, leak-free)

Fitted normalization + threshold (all fit on 24 val, applied to 312 test):

| Dataset | Mod | start_q | end_q | thr | clip@1.0 | Saturation |
|---|---|---|---|---|---|---|
| MOOD_IXI_all | T2 | 0.9 | 0.99999 | 0.59 | 0.3% | ok |
| MOOD_IXI_all | T1 | 0.9 | 0.9999 | 0.95 | 4.1% | ok |
| MOOD_IXI_3T_T1T2 | T2 | 0.5 | 0.99999 | 0.85 | 0.7% | ok |
| MOOD_IXI_3T_T1T2 | T1 | 0.9 | 0.99999 | 0.77 | 1.6% | ok |
| MOOD_3T | T1 | 0.9 | 1.0 | 1.00 | 11.4% | **still saturating** |
| MOOD_IXI_3T | T1 | 0.5 | 0.999999 | 0.79 | 1.0% | ok |
| MOOD_IXI_3T_15T | T1 | 0.9 | 0.99999 | 0.76 | 1.5% | ok |
| IXI_3T | T2 | 0.5 | 0.999999 | 0.91 | 2.1% | ok |
| IXI_15T | T2 | 0.9 | 1.0 | 0.48 | 0.4% | ok |
| IXI_combined | T2 | 0.5 | 0.99999 | 0.95 | 2.7% | ok |

Metrics (median per-slice Dice, per-volume Dice, slice AUROC, size-stratified Dice):

| Dataset | Mod | Slice Dice | Vol Dice | Slice AUROC | Dice large | AUROC large | old Slice Dice (leaked) |
|---|---|---|---|---|---|---|---|
| MOOD_IXI_all | T2 | 0.379 | 0.372 | 0.823 | 0.621 | 0.926 | 0.411 |
| MOOD_IXI_3T_T1T2 | T2 | 0.347 | 0.375 | 0.812 | 0.623 | 0.915 | 0.458 |
| IXI_15T | T2 | 0.405 | 0.354 | 0.772 | 0.677 | 0.896 | 0.386 |
| IXI_3T | T2 | 0.381 | 0.313 | 0.706 | 0.661 | 0.846 | 0.379 |
| IXI_combined | T2 | 0.395 | 0.321 | 0.726 | 0.660 | 0.855 | 0.450 |
| MOOD_3T † | T1 | 0.167 | 0.134 | 0.823 | 0.347 | 0.907 | 0.148 |
| MOOD_IXI_all | T1 | 0.057 | 0.116 | 0.797 | 0.245 | 0.888 | 0.149 |
| MOOD_IXI_3T_T1T2 | T1 | 0.037 | 0.119 | 0.810 | 0.208 | 0.898 | 0.153 |
| MOOD_IXI_3T_15T | T1 | 0.006 | 0.089 | 0.786 | 0.141 | 0.872 | 0.153 |
| MOOD_IXI_3T | T1 | 0.000 | 0.068 | 0.799 | 0.126 | 0.879 | 0.186 |

† `MOOD_3T` remains saturating even at `end_q = 1.0` (the healthy-max, the ceiling of a
quantile normalization): 11.4% of pixels still clip and its val threshold stays pinned
at 1.0. Its number is reported but should be treated as unreliable.

**Interpretation**
- **T2 is a genuine result:** median slice Dice **0.35–0.41**, large-tumor Dice
  0.62–0.68, AUROC 0.71–0.82. It survives the leak-free protocol (some datasets up, some
  down vs the leaked numbers).
- **T1 largely collapses:** median slice Dice **0.00–0.06** once the threshold is fit on
  val rather than test. The T1 models only localize *large* tumors (large-Dice 0.13–0.35,
  slice AUROC ~0.80 unchanged); the previous non-zero T1 medians were essentially an
  artifact of the test-set threshold fit.
- **Small tumors** (≤494 px) remain undetectable at 256×256 across all runs.

Full CSVs: `outputs/summary_valfit_<DATASET>_<mod>.csv`; fitted params
`outputs/<DATASET>/valfit_<mod>.json`; normalized 312-subject test predictions
`outputs/<DATASET>/valfit_test_<mod>/`.

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

- Raw re-inference (gpuq, 1 GPU): jobs 40227150–40227157, ~15 min calibration + ~15 min inference each.
- Val-fit + 312-test eval (cpuq): jobs 40278909–40278916.
- **W&B:** project `Dinomaly2`, entity `negin-kafee2-politecnico-di-milano`, run `dinomaly2_brain_<DATASET>`.

---

## 6. Reproduce

```bash
# 1) train -> raw inference (per dataset)
cd slurm
DATASET=IXI_15T DATA_PATH=/project/detectanomaly/training/datasets/T2_only/IXI_15T \
INFER_MODALITIES="t2" NGPU=1 BATCH_SIZE=32 bash submit_pipeline.sh   # training
DATASET=IXI_15T DATA_PATH=... INFER_MODALITIES="t2" sbatch 02b_infer_raw.sbatch  # raw + calib dump

# 2) leak-free val-fit + 312-test eval
DATASET=IXI_15T DATA_PATH=... INFER_MODALITIES="t2" sbatch 04_valfit_eval.sbatch
```

Core code: `dataset_brain.py`, `dinomaly_brain.py` (DDP), `inference_brain.py`.
Leak-free eval: `tools/valfit.py`, `tools/apply_norm.py`, `slurm/04_valfit_eval.sbatch`.
Read-only diagnostics: `tools/diagnose_threshold.py`, `tools/sweep_norm.py`.

---

## 7. Deviations from the paper & notes

- **Batch size 32** (paper 16) — approved.
- **Train at 280×280**, maps saved at 256×256 (nearest); 256 not divisible by patch 14.
- **1-channel input** replicated to 3 channels (no model change).
- **DDP + checkpoint-resume + W&B** added for cluster training.
- **Map normalization** (healthy-quantile) added; **fit on the 24 val subjects** — both
  the quantile and the threshold — then applied to the 312 test subjects. Superseded a
  leaky earlier version (quantile + threshold both test-fit). `MOOD_3T` (T1) cannot be
  de-saturated even at `end_q = 1.0`, a model-quality limit.
- **Paper cross-check (±5%): not applicable.** Dinomaly/Dinomaly2 benchmark only
  *industrial* data (MVTec-AD / VisA); there are no brain-MRI numbers to compare to.

---

## 8. Incidents

- Two eval jobs (`IXI_3T`, `IXI_combined`) stalled on an unresponsive node for ~10 days
  and were killed by cluster maintenance (2026-07-30); resubmitted and completed.
- One training run was interrupted by maintenance and resumed from checkpoint.
