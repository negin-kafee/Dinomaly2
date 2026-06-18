"""Dinomaly2 training on healthy brain-MRI (NIfTI), with DDP + checkpoint-resume + W&B.

The model, loss, optimizer, scheduler and all hyper-parameters are kept identical
to the paper's 2D medical configuration (see dinomaly_2D.py). The only additions
relative to the original script are:
  * a NIfTI 2D-slice dataloader (dataset_brain.py),
  * multi-GPU DistributedDataParallel with a DistributedSampler,
  * checkpoint saving + automatic resume,
  * Weights & Biases logging.

Launch with torchrun, e.g.:
  torchrun --nproc_per_node=4 dinomaly_brain.py \
      --data_path /.../T1T2_combined/MOOD_IXI_all \
      --save_dir ./outputs/MOOD_IXI_all --save_name dinomaly2_brain_moodixiall
"""

import os
import argparse
import random
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from utils import global_cosine, global_cosine_hm_percent, WarmupCosineScheduler
from optimizers import StableAdamW
from dataset_brain import BrainMRINiftiTrainDataset, build_slice_index

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def build_model(args, device):
    """Identical architecture/hyper-params to dinomaly_2D.py."""
    if args.lc == 0:
        fuse_layer_encoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
        fuse_layer_decoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif args.lc == 1:
        fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    elif args.lc == 2:
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    elif args.lc == 3:
        fuse_layer_encoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
    elif args.lc == 4:
        fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
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
    model = model.to(device)
    model.init_weights()
    # Encoder stays frozen (as in the paper); freeze grads so DDP does not sync it.
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model, bottleneck, decoder


def main():
    parser = argparse.ArgumentParser(description='Dinomaly2 brain-MRI training (DDP)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Dataset root containing a raw/ folder of .nii.gz volumes')
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--save_name', type=str, default='dinomaly2_brain')
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_small_14')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--la', type=int, default=1)
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--ll', type=int, default=1)
    parser.add_argument('--ll_ratio', type=float, default=0.9)
    parser.add_argument('--ll_factor', type=float, default=0.1)
    parser.add_argument('--cr', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=280)
    parser.add_argument('--crop_size', type=int, default=280)
    parser.add_argument('--total_iters', type=int, default=40000)
    parser.add_argument('--lr_decay_ratio', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Per-GPU batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fg_thresh', type=float, default=0.01)
    parser.add_argument('--ckpt_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    # W&B
    parser.add_argument('--wandb_project', type=str, default='Dinomaly2')
    parser.add_argument('--wandb_entity', type=str, default='negin-kafee2-politecnico-di-milano')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # ---- distributed init ----
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    rank = get_rank()

    setup_seed(1 + rank)

    save_path = os.path.join(args.save_dir, args.save_name)
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, 'last.pth')

    # ---- build slice index once on rank 0, then everyone loads the cache ----
    raw_dir = os.path.join(args.data_path, 'raw')
    if not os.path.isdir(raw_dir):
        raw_dir = args.data_path
    cache_path = os.path.join(args.data_path,
                              f'.dinomaly_slice_index_fg{args.fg_thresh}_ax2.json')
    if rank == 0 and not os.path.exists(cache_path):
        build_slice_index(raw_dir, cache_path, args.fg_thresh, slice_axis=2)
    if is_dist():
        dist.barrier()

    train_data = BrainMRINiftiTrainDataset(args.data_path, crop_size=args.crop_size,
                                           fg_thresh=args.fg_thresh, cache_path=cache_path)

    if world_size > 1:
        sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
    else:
        sampler = None
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, persistent_workers=args.num_workers > 0)

    model, bottleneck, decoder = build_model(args, device)
    trainable = nn.ModuleList([bottleneck, decoder])

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    core = model.module if isinstance(model, DDP) else model

    optimizer = StableAdamW([{'params': bottleneck[0].parameters(), 'lr': 2e-4},
                             {'params': bottleneck[1].parameters()},
                             {'params': decoder.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                            amsgrad=False, eps=1e-10)
    lr_scheduler = WarmupCosineScheduler(optimizer, final_ratio=args.lr_decay_ratio,
                                         total_epochs=args.total_iters, warmup_epochs=100)

    # ---- resume ----
    start_it = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        core.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        start_it = ckpt['iter']
        if rank == 0:
            print(f'[resume] loaded checkpoint at iter {start_it}', flush=True)

    # ---- W&B (rank 0 only) ----
    use_wandb = (rank == 0) and (not args.no_wandb)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.save_name, id=args.save_name, resume='allow',
                       config=vars(args))
        except Exception as e:
            print(f'[wandb] disabled ({e})', flush=True)
            use_wandb = False

    if rank == 0:
        print(f'train slices: {len(train_data)} | world_size: {world_size} | '
              f'per-gpu batch: {args.batch_size}', flush=True)

    # ---- train loop ----
    model.train()
    it = start_it
    epoch = 0
    loss_window = []
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for img, _ in train_loader:
            img = img.to(device, non_blocking=True)
            en, de = model(img)

            p_final = args.ll_ratio
            p = min(p_final * it / 1000, p_final)
            if args.ll:
                loss = global_cosine_hm_percent(en, de, p=p, factor=args.ll_factor)
            else:
                loss = global_cosine(en, de)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()

            loss_window.append(loss.item())
            it += 1

            if rank == 0 and it % args.log_interval == 0:
                mean_loss = float(np.mean(loss_window))
                loss_window = []
                lr = optimizer.param_groups[-1]['lr']
                print(f'iter [{it}/{args.total_iters}] loss:{mean_loss:.4f} lr:{lr:.2e}',
                      flush=True)
                if use_wandb:
                    wandb.log({'loss': mean_loss, 'lr': lr, 'iter': it}, step=it)

            if rank == 0 and (it % args.ckpt_interval == 0 or it >= args.total_iters):
                torch.save({'model': core.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': lr_scheduler.state_dict(),
                            'iter': it, 'args': vars(args)}, ckpt_path)

            if it >= args.total_iters:
                done = True
                break
        epoch += 1

    if rank == 0:
        torch.save({'model': core.state_dict(), 'iter': it, 'args': vars(args)},
                   os.path.join(save_path, 'model_final.pth'))
        print('Training complete.', flush=True)
        if use_wandb:
            wandb.finish()

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
