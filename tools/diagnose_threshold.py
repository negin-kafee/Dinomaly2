#!/usr/bin/env python
"""READ-ONLY diagnosis for the two methodology concerns.

1. Subject overlap: how many of each dataset's saved predictions fall in the
   24-subject val bank vs the 312-subject test list vs neither.
2. Threshold-grid ceiling: replicate the eval's global (micro) Dice threshold
   search (down-sampled [::4] pixels, as in find_optimal_threshold) but with an
   EXTENDED upper grid, and report where the interior optimum lands and how much
   Dice moves versus the 0.94-capped grid.

Uses cumulative histograms of the (already normalized) anomaly maps so it is
exact for the micro-Dice and cheap in memory. Modifies nothing.
"""
import glob
import os

import numpy as np

SHARED = "/project/detectanomaly/training/shared/brats_val"
OUT_BASE = "/project/detectanomaly/training/repos/Dinomaly2/outputs"

NBINS = 2000  # histogram resolution over [0, 1]


def load_ids(path):
    with open(path) as f:
        return set(l.strip() for l in f if l.strip())


def subject_id_from_file(f):
    return os.path.basename(f)[:-4]  # strip .npz


def accumulate_hist(npz_files):
    """Return (hist_all, hist_gt, n_gt_total) over [::4] down-sampled pixels."""
    hist_all = np.zeros(NBINS, dtype=np.int64)
    hist_gt = np.zeros(NBINS, dtype=np.int64)
    n_gt = 0
    for f in npz_files:
        d = np.load(f)
        a = d['anomaly_maps']
        g = d['gt_masks']
        for i in range(a.shape[0]):
            p = a[i][::4, ::4].astype(np.float32).ravel()
            m = (g[i][::4, ::4] > 0).ravel()
            idx = np.clip((p * NBINS).astype(np.int64), 0, NBINS - 1)
            hist_all += np.bincount(idx, minlength=NBINS)
            if m.any():
                hist_gt += np.bincount(idx[m], minlength=NBINS)
                n_gt += int(m.sum())
    return hist_all, hist_gt, n_gt


def best_threshold(hist_all, hist_gt, n_gt, t_lo, t_hi, step):
    # cumulative-from-top: P(>=t), TP(>=t)
    edges = np.arange(NBINS) / NBINS
    rev_all = np.cumsum(hist_all[::-1])[::-1].astype(np.float64)
    rev_gt = np.cumsum(hist_gt[::-1])[::-1].astype(np.float64)
    best = (0.0, 0.0)
    for t in np.arange(t_lo, t_hi, step):
        b = int(np.clip(round(t * NBINS), 0, NBINS - 1))
        P = rev_all[b]
        TP = rev_gt[b]
        denom = P + n_gt
        d = 0.0 if denom == 0 else 2.0 * TP / denom
        if d > best[1]:
            best = (float(t), float(d))
    return best  # (thr, dice)


def main():
    test_ids = load_ids(os.path.join(SHARED, "312_test_subject_ids.txt"))
    val_csv = os.path.join(SHARED, "splits", "BraTS_T2_val24.csv")
    val_ids = set()
    if os.path.exists(val_csv):
        with open(val_csv) as f:
            next(f)
            for line in f:
                if line.strip():
                    val_ids.add(line.split(',')[0].strip())
    print(f"val_ids={len(val_ids)}  test_ids={len(test_ids)}\n")

    dirs = sorted(glob.glob(os.path.join(OUT_BASE, "*", "infer_*", "raw_predictions")))
    print("%-34s %5s %4s %5s %5s | %-18s | %-18s %s" % (
        "dataset/infer", "npz", "val", "test", "othr",
        "grid0.05-0.94", "extended 0.05-0.999", "interior?"))
    for rd in dirs:
        parts = rd.split(os.sep)
        tag = parts[-3] + "/" + parts[-2]
        files = sorted(glob.glob(os.path.join(rd, "*.npz")))
        ids = [subject_id_from_file(f) for f in files]
        nval = sum(i in val_ids for i in ids)
        ntest = sum(i in test_ids for i in ids)
        nother = len(ids) - nval - ntest
        # test-only subset for a leak-free reporting view
        test_files = [f for f, i in zip(files, ids) if i in test_ids]
        use = test_files if test_files else files
        ha, hg, ng = accumulate_hist(use)
        thrA, dA = best_threshold(ha, hg, ng, 0.05, 0.95, 0.01)
        thrB, dB = best_threshold(ha, hg, ng, 0.05, 0.999, 0.001)
        interior = "yes" if thrB < 0.985 else "NO(ceiling)"
        print("%-34s %5d %4d %5d %5d | thr%.2f D%.4f | thr%.3f D%.4f  %s" % (
            tag, len(ids), nval, ntest, nother, thrA, dA, thrB, dB, interior))


if __name__ == "__main__":
    main()
