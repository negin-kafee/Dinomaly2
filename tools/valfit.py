#!/usr/bin/env python
"""Leak-free val-fit of the map-normalization quantile AND operating threshold.

Fits ONLY on the 24-subject validation bank (disjoint from the 312-subject test
set). For a grid of (start_q, end_q) healthy-quantile pairs it:
  - normalizes the VAL maps with clip((raw-start)/(end-start), 0, 1),
  - finds the val-optimal threshold by global (micro) Dice over an EXTENDED grid,
  - measures the fraction of pixels that clip to exactly 1.0 (saturation),
and selects the config with the best val Dice among those whose val-optimal
threshold is INTERIOR (not pinned to the ceiling). If none is interior, it warns
that the normalization is still saturating and the end quantile must go higher.

Writes the chosen params to --out_json. Modifies no predictions.
"""
import argparse
import glob
import json
import os

import numpy as np

NBINS = 2000
CEILING = 0.98  # a val-optimal threshold >= this is treated as "pinned"


def load_ids(path):
    ids = set()
    with open(path) as f:
        if path.endswith('.csv'):
            next(f)
            for line in f:
                if line.strip():
                    ids.add(line.split(',')[0].strip())
        else:
            for line in f:
                if line.strip():
                    ids.add(line.strip())
    return ids


def sid(f):
    return os.path.basename(f)[:-4]


def val_hist(files, start, end):
    denom = max(end - start, 1e-9)
    hist_all = np.zeros(NBINS, dtype=np.int64)
    hist_gt = np.zeros(NBINS, dtype=np.int64)
    n_gt = 0
    clip = 0
    tot = 0
    for f in files:
        d = np.load(f)
        a = d['anomaly_maps']
        g = d['gt_masks']
        for i in range(a.shape[0]):
            p = a[i][::4, ::4].astype(np.float32).ravel()
            n = np.clip((p - start) / denom, 0.0, 1.0)
            m = (g[i][::4, ::4] > 0).ravel()
            idx = np.clip((n * NBINS).astype(np.int64), 0, NBINS - 1)
            hist_all += np.bincount(idx, minlength=NBINS)
            if m.any():
                hist_gt += np.bincount(idx[m], minlength=NBINS)
                n_gt += int(m.sum())
            clip += int((n >= 1.0).sum())
            tot += n.size
    return hist_all, hist_gt, n_gt, clip, tot


def best_thr(hist_all, hist_gt, n_gt, t_lo, t_hi, step):
    rev_all = np.cumsum(hist_all[::-1])[::-1].astype(np.float64)
    rev_gt = np.cumsum(hist_gt[::-1])[::-1].astype(np.float64)
    best = (0.0, 0.0)
    for t in np.arange(t_lo, t_hi, step):
        b = int(np.clip(round(t * NBINS), 0, NBINS - 1))
        P = rev_all[b]
        TP = rev_gt[b]
        den = P + n_gt
        d = 0.0 if den == 0 else 2.0 * TP / den
        if d > best[1]:
            best = (float(t), float(d))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--calib_values', required=True)
    ap.add_argument('--val_ids', required=True)
    ap.add_argument('--test_ids', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--start_qs', type=float, nargs='+', default=[0.5, 0.9])
    ap.add_argument('--end_qs', type=float, nargs='+',
                    default=[0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0])
    args = ap.parse_args()

    healthy = np.load(args.calib_values)
    val_ids = load_ids(args.val_ids)
    test_ids = load_ids(args.test_ids)
    files = sorted(glob.glob(os.path.join(args.raw_dir, '*.npz')))
    val_files = [f for f in files if sid(f) in val_ids]
    test_files = [f for f in files if sid(f) in test_ids]
    print(f'raw={len(files)}  val_files={len(val_files)}  test_files={len(test_files)}',
          flush=True)

    results = []
    for sq in args.start_qs:
        for eq in args.end_qs:
            if eq <= sq:
                continue
            start = float(np.quantile(healthy, sq))
            end = float(np.quantile(healthy, eq))
            if end <= start:
                end = start + 1e-6
            ha, hg, ng, clip, tot = val_hist(val_files, start, end)
            thr, dice = best_thr(ha, hg, ng, 0.05, 0.998, 0.002)
            clip_frac = clip / max(tot, 1)
            interior = thr < CEILING
            results.append(dict(start_q=sq, end_q=eq, start=start, end=end,
                                val_thr=thr, val_dice=dice, clip_frac=clip_frac,
                                interior=interior))
            print(f'start_q={sq:<8} end_q={eq:<9} start={start:.5f} end={end:.5f} '
                  f'val_thr={thr:.3f} val_Dice={dice:.4f} clip@1.0={clip_frac:.4f} '
                  f'interior={interior}', flush=True)

    interior_res = [r for r in results if r['interior']]
    pool = interior_res if interior_res else results
    best = max(pool, key=lambda r: r['val_dice'])
    best['n_val'] = len(val_files)
    best['n_test'] = len(test_files)
    best['saturating_warning'] = not bool(interior_res)
    if not interior_res:
        print('WARNING: no interior threshold at any end_q -> still saturating; '
              'extend end_qs higher.', flush=True)
    print('\nCHOSEN:', json.dumps(best, indent=2), flush=True)
    with open(args.out_json, 'w') as f:
        json.dump(best, f, indent=2)


if __name__ == '__main__':
    main()
