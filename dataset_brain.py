"""Brain-MRI NIfTI datasets for Dinomaly2.

Minimal addition for reading 3D NIfTI (.nii.gz) brain-MRI volumes as 2D axial
slices, keeping the rest of the repo unchanged.

Design decisions (confirmed with project owner):
  * Single-modality input: each volume is one modality (T1 *or* T2). The single
    grayscale channel is replicated to 3 channels so the pretrained RGB DINOv2
    backbone can be used without any model surgery.
  * Native training resolution is 280x280 (= 14 * 20, divisible by the DINOv2
    patch size). Anomaly maps are interpolated to 256 at save time by the
    inference script.
  * ImageNet normalization (same mean/std as the original repo) is applied after
    replicating to 3 channels.
"""

import os
import json
import glob

import numpy as np
import cv2
import torch
import nibabel as nib


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_slice(slice2d, p_low=0.5, p_high=99.5):
    """Robust per-slice intensity normalization to [0, 1].

    Uses percentiles of the non-zero (brain) voxels so the background stays at 0.
    """
    slice2d = slice2d.astype(np.float32)
    fg = slice2d[slice2d > 0]
    if fg.size == 0:
        return np.zeros_like(slice2d)
    lo = np.percentile(fg, p_low)
    hi = np.percentile(fg, p_high)
    if hi <= lo:
        hi = fg.max()
        lo = fg.min()
    if hi <= lo:
        return np.zeros_like(slice2d)
    out = (slice2d - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _to_model_tensor(slice2d, size):
    """[H, W] float slice in [0,1] -> normalized [3, size, size] float tensor."""
    img = cv2.resize(slice2d, (size, size), interpolation=cv2.INTER_LINEAR)
    img = np.repeat(img[:, :, None], 3, axis=2)  # H, W, 3
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1)).copy()  # 3, H, W
    return torch.from_numpy(img).float()


def build_slice_index(raw_dir, cache_path, fg_thresh=0.01, slice_axis=2):
    """Scan all volumes once and cache (relpath, slice_idx) for brain slices.

    A slice is kept if the fraction of non-zero (brain) voxels exceeds
    ``fg_thresh``. The index is cached to JSON to avoid rescanning.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, '*.nii.gz')) +
                   glob.glob(os.path.join(raw_dir, '*.nii')))
    index = []
    for p in paths:
        rel = os.path.relpath(p, raw_dir)
        try:
            vol = np.asanyarray(nib.load(p).dataobj)
        except Exception as e:  # pragma: no cover - corrupt file guard
            print(f'[build_slice_index] skip {rel}: {e}')
            continue
        if vol.ndim != 3:
            continue
        vol = np.moveaxis(vol, slice_axis, 0)  # [Z, H, W]
        n_vox = vol.shape[1] * vol.shape[2]
        fg_frac = (vol > 0).reshape(vol.shape[0], -1).sum(axis=1) / float(n_vox)
        for k in np.where(fg_frac > fg_thresh)[0]:
            index.append([rel, int(k)])
    tmp = cache_path + f'.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(index, f)
    os.replace(tmp, cache_path)
    print(f'[build_slice_index] {len(index)} slices from {len(paths)} volumes '
          f'-> {cache_path}')
    return index


class BrainMRINiftiTrainDataset(torch.utils.data.Dataset):
    """Healthy brain-MRI volumes -> 2D axial slices for unsupervised training.

    Returns ``(img, label=0)`` to match the original ImageFolder-based loop.
    """

    def __init__(self, data_path, crop_size=280, fg_thresh=0.01,
                 slice_axis=2, cache_path=None):
        self.raw_dir = os.path.join(data_path, 'raw')
        if not os.path.isdir(self.raw_dir):
            # allow pointing directly at a folder of volumes
            self.raw_dir = data_path
        self.crop_size = crop_size
        self.slice_axis = slice_axis
        if cache_path is None:
            cache_path = os.path.join(
                data_path, f'.dinomaly_slice_index_fg{fg_thresh}_ax{slice_axis}.json')
        self.cache_path = cache_path
        if not os.path.exists(cache_path):
            build_slice_index(self.raw_dir, cache_path, fg_thresh, slice_axis)
        with open(cache_path) as f:
            self.index = json.load(f)
        if len(self.index) == 0:
            raise RuntimeError(f'No training slices found under {self.raw_dir}')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        rel, k = self.index[i]
        path = os.path.join(self.raw_dir, rel)
        vol = nib.load(path)
        sl = np.take(np.asanyarray(vol.dataobj), k, axis=self.slice_axis)
        sl = _normalize_slice(sl)
        img = _to_model_tensor(sl, self.crop_size)
        return img, 0


class BraTSSubjectInferenceDataset(torch.utils.data.Dataset):
    """One BraTS subject per item: all axial slices of a modality + binary GT.

    ``__getitem__`` returns a dict with the full subject volume so the inference
    script can batch slices internally and save one ``.npz`` per subject.
    """

    def __init__(self, brats_root, modality, crop_size=280, save_size=256,
                 slice_axis=2):
        assert modality in ('t1', 't2')
        self.brats_root = brats_root
        self.modality = modality
        self.crop_size = crop_size
        self.save_size = save_size
        self.slice_axis = slice_axis

        self.raw_root = os.path.join(brats_root, 'BraTS_raw')
        self.seg_dir = os.path.join(brats_root, f'BraTS_{modality.upper()}_seg')

        subjects = []
        for seg_path in sorted(glob.glob(os.path.join(self.seg_dir, '*.nii.gz'))):
            sid = os.path.basename(seg_path).replace(f'_{modality}_seg.nii.gz', '')
            vol_path = os.path.join(self.raw_root, sid, f'{sid}_{modality}.nii.gz')
            if os.path.exists(vol_path):
                subjects.append((sid, vol_path, seg_path))
        if len(subjects) == 0:
            raise RuntimeError(
                f'No BraTS subjects for modality {modality} under {brats_root}')
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, i):
        sid, vol_path, seg_path = self.subjects[i]
        vol = np.asanyarray(nib.load(vol_path).dataobj)
        seg = np.asanyarray(nib.load(seg_path).dataobj)
        vol = np.moveaxis(vol, self.slice_axis, 0)  # [Z, H, W]
        seg = np.moveaxis(seg, self.slice_axis, 0)

        n = vol.shape[0]
        imgs = np.empty((n, 3, self.crop_size, self.crop_size), dtype=np.float32)
        gts = np.empty((n, self.save_size, self.save_size), dtype=np.uint8)
        for k in range(n):
            sl = _normalize_slice(vol[k])
            imgs[k] = _to_model_tensor(sl, self.crop_size).numpy()
            g = (seg[k] > 0).astype(np.uint8)
            g = cv2.resize(g, (self.save_size, self.save_size),
                           interpolation=cv2.INTER_NEAREST)
            gts[k] = (g > 0).astype(np.uint8)

        return {
            'subject_id': sid,
            'images': torch.from_numpy(imgs),         # [N, 3, crop, crop]
            'gt_masks': torch.from_numpy(gts),        # [N, save, save] {0,1}
            'slice_ids': torch.arange(n, dtype=torch.long),
        }
