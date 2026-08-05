#!/usr/bin/env python
"""Apply map-normalization to RAW prediction npz files in place.

Given a directory of RAW per-subject npz (anomaly_maps un-normalized) and the
dumped healthy calibration values, compute bounds from (start_q, end_q) and
rewrite each npz with anomaly_maps rescaled to [0, 1] via
    clip((raw - start)/(end - start), 0, 1)
The eval script then consumes the normalized maps directly.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--calib_values', required=True)
    ap.add_argument('--start_q', type=float, required=True)
    ap.add_argument('--end_q', type=float, required=True)
    ap.add_argument('--out_json', default='',
                    help='Optional path to record the applied bounds.')
    ap.add_argument('--ids_file', default='',
                    help='If set, only subjects whose id is listed here are '
                         'processed (e.g. the 312 test subjects).')
    ap.add_argument('--out_dir', default='',
                    help='If set, write normalized npz here instead of '
                         'overwriting in place.')
    args = ap.parse_args()

    healthy = np.load(args.calib_values)
    start = float(np.quantile(healthy, args.start_q))
    end = float(np.quantile(healthy, args.end_q))
    if end <= start:
        end = start + 1e-6
    denom = end - start
    print(f'applying start_q={args.start_q} end_q={args.end_q} '
          f'-> start={start:.6f} end={end:.6f}', flush=True)

    keep = None
    if args.ids_file:
        with open(args.ids_file) as fh:
            keep = set(l.strip() for l in fh if l.strip())
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    fs = sorted(glob.glob(os.path.join(args.raw_dir, '*.npz')))
    n = 0
    for f in fs:
        sid = os.path.basename(f)[:-4]
        if keep is not None and sid not in keep:
            continue
        d = np.load(f)
        a = d['anomaly_maps'].astype(np.float32)
        a = np.clip((a - start) / denom, 0.0, 1.0).astype(np.float32)
        out = os.path.join(args.out_dir, os.path.basename(f)) if args.out_dir else f
        np.savez_compressed(out, anomaly_maps=a,
                            gt_masks=d['gt_masks'],
                            slice_ids=d['slice_ids'])
        n += 1
        if n % 50 == 0:
            print(f'  [{n}]', flush=True)
    dest = args.out_dir or args.raw_dir
    print(f'normalized {n} subjects -> {dest}', flush=True)

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump({'start': start, 'end': end,
                       'start_q': args.start_q, 'end_q': args.end_q,
                       'n_written': n}, fh)


if __name__ == '__main__':
    main()
