#!/usr/bin/env python
"""Offline sweep of map-normalization quantiles for brain anomaly maps.

Reads RAW (un-normalized) per-subject prediction npz files plus the dumped
healthy calibration-value array, then for a grid of (start_q, end_q) quantiles:
  bounds = (quantile(healthy, start_q), quantile(healthy, end_q))
  norm   = clip((raw - start)/(end - start), 0, 1)
and reports an eval-faithful score: a single global optimal threshold (same
0.05..0.94 grid as the eval script) maximizing the MEDIAN per-slice Dice over
tumor-containing slices.

This picks the normalization quantile choice to lock in before scaling. It does
NOT modify any files.
"""
import argparse
import glob
import os

import numpy as np


def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * inter / s


def load_tumor_slices(raw_dir, max_subjects, stride):
    fs = sorted(glob.glob(os.path.join(raw_dir, '*.npz')))
    if max_subjects:
        fs = fs[:max_subjects]
    slices = []  # list of (raw_map_flat, gt_bool_flat)
    for f in fs:
        d = np.load(f)
        a = d['anomaly_maps']
        g = d['gt_masks'].astype(bool)
        for k in range(a.shape[0]):
            if g[k].any():
                slices.append((a[k, ::stride, ::stride].reshape(-1),
                               g[k, ::stride, ::stride].reshape(-1)))
    return slices, len(fs)


def score_bounds(slices, start, end):
    denom = max(end - start, 1e-9)
    thrs = np.arange(0.05, 0.95, 0.05)
    best = (0.0, 0.0)
    # Pre-normalize once per slice, then sweep thresholds.
    norm = [(np.clip((a - start) / denom, 0.0, 1.0), g) for a, g in slices]
    for t in thrs:
        ds = [dice(a >= t, g) for a, g in norm]
        m = float(np.median(ds))
        if m > best[0]:
            best = (m, float(t))
    return best  # (median_dice, thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True,
                    help='Directory of RAW prediction npz (raw_predictions/).')
    ap.add_argument('--calib_values', required=True,
                    help='.npy of dumped healthy anomaly values.')
    ap.add_argument('--max_subjects', type=int, default=60)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--start_qs', type=float, nargs='+',
                    default=[0.5, 0.9, 0.95])
    ap.add_argument('--end_qs', type=float, nargs='+',
                    default=[0.95, 0.99, 0.995, 0.999, 0.9999])
    args = ap.parse_args()

    healthy = np.load(args.calib_values)
    slices, nsub = load_tumor_slices(args.raw_dir, args.max_subjects, args.stride)
    print(f'loaded {len(slices)} tumor slices from {nsub} subjects; '
          f'healthy calib values: n={healthy.size}', flush=True)

    results = []
    for sq in args.start_qs:
        for eq in args.end_qs:
            if eq <= sq:
                continue
            start = float(np.quantile(healthy, sq))
            end = float(np.quantile(healthy, eq))
            if end <= start:
                end = start + 1e-6
            med, thr = score_bounds(slices, start, end)
            results.append((med, sq, eq, start, end, thr))
            print(f'start_q={sq:<6} end_q={eq:<7} '
                  f'start={start:.5f} end={end:.5f} '
                  f'-> median-Dice={med:.4f} @thr={thr:.2f}', flush=True)

    results.sort(reverse=True)
    best = results[0]
    print('\nBEST: start_q=%.4g end_q=%.4g start=%.5f end=%.5f '
          'median-Dice=%.4f @thr=%.2f' %
          (best[1], best[2], best[3], best[4], best[0], best[5]), flush=True)


if __name__ == '__main__':
    main()
