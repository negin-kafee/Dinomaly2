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
    args = ap.parse_args()

    healthy = np.load(args.calib_values)
    start = float(np.quantile(healthy, args.start_q))
    end = float(np.quantile(healthy, args.end_q))
    if end <= start:
        end = start + 1e-6
    denom = end - start
    print(f'applying start_q={args.start_q} end_q={args.end_q} '
          f'-> start={start:.6f} end={end:.6f}', flush=True)

    fs = sorted(glob.glob(os.path.join(args.raw_dir, '*.npz')))
    for i, f in enumerate(fs):
        d = np.load(f)
        a = d['anomaly_maps'].astype(np.float32)
        a = np.clip((a - start) / denom, 0.0, 1.0).astype(np.float32)
        np.savez_compressed(f, anomaly_maps=a,
                            gt_masks=d['gt_masks'],
                            slice_ids=d['slice_ids'])
        if (i + 1) % 50 == 0:
            print(f'  [{i + 1}/{len(fs)}]', flush=True)
    print(f'normalized {len(fs)} subjects in {args.raw_dir}', flush=True)

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump({'start': start, 'end': end,
                       'start_q': args.start_q, 'end_q': args.end_q}, fh)


if __name__ == '__main__':
    main()
