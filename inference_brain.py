"""Dinomaly2 inference on BraTS, saving raw per-subject predictions.

For each BraTS subject (one modality, t1 or t2) this script:
  * runs the trained model on every axial slice,
  * builds the anomaly heatmap with the paper's own pipeline
    (1 - cosine_similarity feature distance + Gaussian smoothing),
  * upsamples the heatmap from the model's native 280 to 256 with NEAREST
    interpolation (as requested),
  * saves per subject a ``.npz`` (consumed by compute_metrics_soumick.py) plus
    a continuous heatmap and a binary mask as ``.nii.gz``.

No metrics are computed here. Binarisation for the saved .nii.gz mask is for
visualisation only; the authoritative metrics use the continuous heatmap and the
evaluation script's own threshold search.
"""

import os
import argparse
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader

from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel
from dataset_brain import BraTSSubjectInferenceDataset

warnings.filterwarnings("ignore")


def build_model(args, device):
    if args.lc == 2:
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    elif args.lc == 1:
        fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    elif args.lc == 0:
        fuse_layer_encoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
        fuse_layer_decoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
    else:
        raise ValueError("loose constraint value not supported")

    encoder = vit_encoder.load(args.backbone)
    if 'small' in args.backbone:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'base' in args.backbone:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'large' in args.backbone:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    bottleneck = []
    dropout = args.dropout
    bottleneck.append(nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)))
    bottleneck.append(nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
                                    nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout)))
    bottleneck = nn.ModuleList(bottleneck)

    decoder = []
    for _ in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=partial(LinearAttention2, eps=1e-8) if args.la else Attention)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                     target_layers=target_layers, remove_class_token=False,
                     fuse_layer_encoder=fuse_layer_encoder,
                     fuse_layer_decoder=fuse_layer_decoder,
                     context_aware_recenter=args.cr)
    return model.to(device)


@torch.no_grad()
def infer_subject(model, images, device, gaussian, crop_size, save_size, batch_size):
    """images: [N, 3, crop, crop] -> heatmaps [N, save, save] float32."""
    n = images.shape[0]
    out = np.empty((n, save_size, save_size), dtype=np.float32)
    for s in range(0, n, batch_size):
        batch = images[s:s + batch_size].to(device, non_blocking=True)
        en, de = model(batch)
        amap, _ = cal_anomaly_maps(en, de, crop_size)        # [b, 1, crop, crop]
        amap = gaussian(amap)
        amap = F.interpolate(amap, size=save_size, mode='nearest')  # NN -> 256
        out[s:s + batch_size] = amap[:, 0].float().cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description='Dinomaly2 BraTS inference (raw predictions)')
    parser.add_argument('--brats_root', type=str, required=True)
    parser.add_argument('--modality', type=str, required=True, choices=['t1', 't2'])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=280)
    parser.add_argument('--save_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_nifti', action='store_true',
                        help='Also save heatmap/mask .nii.gz per subject')
    parser.add_argument('--mask_percentile', type=float, default=98.0,
                        help='Percentile threshold for the saved binary .nii.gz mask (viz only)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    npz_dir = os.path.join(args.out_dir, 'raw_predictions')
    os.makedirs(npz_dir, exist_ok=True)
    if args.save_nifti:
        nii_dir = os.path.join(args.out_dir, 'nifti')
        os.makedirs(nii_dir, exist_ok=True)

    model = build_model(args, device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[ckpt] loaded {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}',
          flush=True)
    model.eval()

    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    dataset = BraTSSubjectInferenceDataset(args.brats_root, args.modality,
                                           crop_size=args.crop_size,
                                           save_size=args.save_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, collate_fn=lambda b: b[0])

    print(f'[infer] {len(dataset)} subjects, modality={args.modality}', flush=True)
    for i, sample in enumerate(loader):
        sid = sample['subject_id']
        images = sample['images']
        gt_masks = sample['gt_masks'].numpy().astype(np.uint8)
        slice_ids = sample['slice_ids'].numpy()

        heatmaps = infer_subject(model, images, device, gaussian,
                                 args.crop_size, args.save_size, args.batch_size)

        np.savez_compressed(os.path.join(npz_dir, f'{sid}.npz'),
                            anomaly_maps=heatmaps.astype(np.float32),
                            gt_masks=gt_masks,
                            slice_ids=slice_ids)

        if args.save_nifti:
            thr = np.percentile(heatmaps, args.mask_percentile)
            binmask = (heatmaps >= thr).astype(np.uint8)
            # save as [H, W, Z]
            hm = np.moveaxis(heatmaps, 0, 2)
            bm = np.moveaxis(binmask, 0, 2)
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(hm.astype(np.float32), affine),
                     os.path.join(nii_dir, f'{sid}_heatmap.nii.gz'))
            nib.save(nib.Nifti1Image(bm.astype(np.uint8), affine),
                     os.path.join(nii_dir, f'{sid}_mask.nii.gz'))

        if (i + 1) % 20 == 0:
            print(f'  [{i + 1}/{len(dataset)}] {sid} done', flush=True)

    print(f'[infer] saved raw predictions to {npz_dir}', flush=True)


if __name__ == '__main__':
    main()
