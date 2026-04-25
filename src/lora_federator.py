#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for Heterogeneous LoRA Federated Fine-tuning of Diffusion Models.

This is a NEW, standalone entry point that does NOT modify any existing files.
It reuses the existing Diffuser, data loading utilities, and evaluation tools,
but replaces the training/aggregation pipeline with LoRA-based federated learning.

Usage:
    # Training with homogeneous LoRA (all clients use same rank)
    python src/lora_federator.py --dataset=celeba --train=1 --lora_rank=8 \
        --num_channels=3 --image_size=64 --load_model=model_celeba.pth \
        --rounds=10 --num_users=5 --local_ep=3 --local_bs=64
    
    # Training with heterogeneous ranks (different clients have different ranks)
    python src/lora_federator.py --dataset=celeba --train=1 \
        --lora_ranks="8,16,4,8,16" --global_lora_rank=16 \
        --num_channels=3 --image_size=64 --load_model=model_celeba.pth
    
    # Inference / Sampling only (no training)
    python src/lora_federator.py --dataset=celeba --train=0 \
        --lora_rank=8 --num_channels=3 --image_size=64 \
        --load_model=models/lora_...pth --export_samples=16 --show_samples=0
"""

import copy
import csv
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


# ============================================================
# Add src/ to path for imports
# ============================================================
SRC_DIR = str(Path(__file__).parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from diffuser import Diffuser
from hetero_aggregation import HeteroLoRAAggregator, prefix_slice_distribution
from lora import (
    inject_lora_into_unet,
    extract_all_lora_factors,
    set_all_lora_factors,
    get_lora_state_dict,
    count_communication_bytes,
    freeze_non_lora_params,
    count_lora_params,
)
from lora_client import LoRAClient, create_heterogeneous_clients
from options import args_parser
from unet import UnetConditional, Unet
from utils import (
    exp_details,
    export_dataset,
    count_update_parameters,
    get_partitioned_dataset,
    export_samples,
    show_samples,
    perform_evaluation,
)

# Legacy diff_model module shim (for loading old .pth files)
import types as _legacy_types
from unet import (SinusoidalPositionEmbeddings, Residual, Upsample, Downsample,
                  Block, ResnetBlock, ConvNextBlock, Attention, LinearAttention,
                  PreNorm, Unet, UnetConditional)
_legacy_diff_model = _legacy_types.ModuleType('diff_model')
_legacy_diff_model.SinusoidalPositionEmbeddings = SinusoidalPositionEmbeddings
_legacy_diff_model.Residual = Residual
_legacy_diff_model.Upsample = Upsample
_legacy_diff_model.Downsample = Downsample
_legacy_diff_model.Block = Block
_legacy_diff_model.ResnetBlock = ResnetBlock
_legacy_diff_model.ConvNextBlock = ConvNextBlock
_legacy_diff_model.Attention = Attention
_legacy_diff_model.LinearAttention = LinearAttention
_legacy_diff_model.PreNorm = PreNorm
_legacy_diff_model.Unet = Unet
_legacy_diff_model.UnetConditional = UnetConditional
if 'diff_model' not in sys.modules:
    sys.modules['diff_model'] = _legacy_diff_model


def parse_lora_ranks(rank_str: str, num_clients: int, default_rank: int = 8) -> Dict[int, int]:
    """Parse comma-separated rank string into per-client dict."""
    if not rank_str or rank_str.strip() == '':
        return {i: default_rank for i in range(num_clients)}
    if ',' in rank_str:
        ranks = [int(x.strip()) for x in rank_str.split(',')]
        if len(ranks) != num_clients:
            print(f'[Warning] lora_ranks has {len(ranks)} entries but num_users={num_clients}')
            # Pad or truncate
            if len(ranks) < num_clients:
                ranks = ranks + [ranks[-1]] * (num_clients - len(ranks))
            else:
                ranks = ranks[:num_clients]
        return {i: r for i, r in enumerate(ranks)}
    else:
        return {i: int(rank_str.strip()) for i in range(num_clients)}


def factors_list_to_nested_dict(factors):
    """
    Convert [(layer_name, B_dict, A_dict), ...] to
    B_nested={layer_name:{factor_key:B}}, A_nested={layer_name:{factor_key:A}}.
    """
    B_nested = {}
    A_nested = {}
    for layer_name, B_dict, A_dict in factors:
        B_nested[layer_name] = {k: v.detach().clone() for k, v in B_dict.items()}
        A_nested[layer_name] = {k: v.detach().clone() for k, v in A_dict.items()}
    return B_nested, A_nested


def build_base_model(args, device: str):
    """
    Build and optionally load the base U-Net model (WITHOUT LoRA injection).
    LoRA injection happens later via inject_lora_into_unet().
    """
    is_conditional = args.conditional == 1
    channels = args.num_channels
    image_size = args.image_size

    model = (
        UnetConditional(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,),
            num_classes=args.num_classes,
        ) if is_conditional else Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,),
        )
    )

    # Load pre-trained weights if specified
    parent_path = Path(__file__).parent.parent
    existing_model_path = parent_path / args.load_model if args.load_model else None
    
    if existing_model_path and os.path.isfile(existing_model_path):
        print(f'Loading base model from {existing_model_path}')
        checkpoint = torch.load(
            existing_model_path, map_location=torch.device(device), weights_only=False
        )
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        elif hasattr(checkpoint, 'state_dict'):
            try:
                model.load_state_dict(checkpoint.state_dict())
                print('Loaded base model weights')
            except RuntimeError:
                print('Architecture mismatch - using loaded model directly')
                model = checkpoint.to(device)
        else:
            raise ValueError(f'Unsupported model format')
    elif args.train == 1:
        print('[Info] No pre-trained model found, training from random init (NOT recommended)')
    
    return model.to(device)


def save_lora_checkpoint(model, path: Path, round_num: int = 0):
    """Save a complete checkpoint (base + LoRA params)."""
    # Save full model for sampling/inference
    torch.save(model, path)
    print(f'Saved checkpoint: {path}')


def run_training(args):
    """Main LoRA federated training loop."""
    start_time = time.time()
    parent_path = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    exp_details(args)

    # ---- Set custom data root if provided ----
    data_root = getattr(args, 'data_root', '')
    if data_root:
        # Override CelebA data path via module-level patch
        import utils as _utils
        _utils._CUSTOM_DATA_ROOT = data_root
        print(f'Using custom data root: {data_root}')

    # ---- Load dataset & partition ----
    train_dataset, client_groups, data_stats = get_partitioned_dataset(args)
    print(f'Dataset stats: {data_stats}')

    # ---- Build clean base diffusion model ----
    base_model_clean = build_base_model(args, device)

    # ---- Detect if loaded model is a LoRA-trained checkpoint ----
    _load_path = parent_path / args.load_model if args.load_model else None
    _is_lora_checkpoint = False
    if _load_path and _load_path.exists():
        try:
            _ck = torch.load(_load_path, map_location=device, weights_only=False)
            # flora checkpoints save full model object (not dict)
            if not isinstance(_ck, dict):
                _is_lora_checkpoint = True
                print(f'[Info] Loaded model is a LoRA checkpoint – reusing existing LoRA')
        except Exception:
            pass

    # ---- Parse heterogeneous rank configuration first ----
    lora_rank_str = getattr(args, 'lora_ranks', str(getattr(args, 'lora_rank', 8)))
    client_rank_map = parse_lora_ranks(
        lora_rank_str,
        args.num_users,
        default_rank=getattr(args, 'lora_rank', 8),
    )

    global_rank = int(getattr(args, 'global_lora_rank', max(client_rank_map.values())))
    max_client_rank = max(client_rank_map.values())
    if global_rank < max_client_rank:
        raise ValueError(
            f'global_lora_rank must be >= max client rank. '
            f'Got global_lora_rank={global_rank}, max_client_rank={max_client_rank}'
        )

    # ---- Server model: global-rank LoRA model for aggregation/checkpoint/sampling ----
    server_model = copy.deepcopy(base_model_clean)
    if not _is_lora_checkpoint:
        inject_lora_into_unet(
            server_model,
            rank=global_rank,
            alpha=getattr(args, 'lora_alpha', 1.0),
            dropout=getattr(args, 'lora_dropout', 0.0),
            target_layers=getattr(args, 'lora_target_layers', 'attention'),
        )
    else:
        # Ensure existing LoRA params are trainable
        for p in server_model.parameters():
            p.requires_grad = True
        # Re-inject fresh LoRA on top of existing one (for continued training)
        inject_lora_into_unet(
            server_model,
            rank=global_rank,
            alpha=getattr(args, 'lora_alpha', 1.0),
            dropout=getattr(args, 'lora_dropout', 0.0),
            target_layers=getattr(args, 'lora_target_layers', 'attention'),
        )

    num_params = sum(p.numel() for p in server_model.parameters())
    lora_params = sum(p.numel() for p in server_model.parameters() if p.requires_grad)
    print(f'\nServer model: {num_params:,} total params, {lora_params:,} global LoRA trainable '
          f'({100*lora_params/num_params:.2f}%)')

    # ---- Initialize diffuser ----
    diffuser = Diffuser(int(args.time_steps))

    # ---- Heterogeneous rank configuration ----
    print(f'\nClient LoRA rank config:')
    for cid, r in sorted(client_rank_map.items()):
        n_data = len(client_groups.get(cid, []))
        print(f'  Client {cid}: rank={r}, samples={n_data}')

    # ---- Create aggregator ----
    aggregator = HeteroLoRAAggregator(
        global_rank=global_rank,
        rank_correction=True,
        use_procrustes=True,
        svd_method='rsvd',
    )

    # Initialize client LoRA factors from the same server global subspace, even in round 0.
    # This is necessary for nested prefix distribution to be true from the beginning.
    init_factors = extract_all_lora_factors(server_model)
    B_init, A_init = factors_list_to_nested_dict(init_factors)
    aggregator._last_distribution = prefix_slice_distribution(B_init, A_init, client_rank_map)

    # ---- Setup output directories ----
    _lr = getattr(args, 'lora_rank', 8)
    model_name = (
        f'lora_{args.dataset}'
        f'_R[{args.rounds}]'
        f'_K[{args.num_users}]'
        f'_r[{_lr}g{global_rank}]'
        f'_E[{args.local_ep}]'
        f'_B[{args.local_bs}]'
        f'_T[{int(args.time_steps)}]'
        f'_I[{args.iid},{args.unequal}]'
    )

    result_folder = parent_path / f'results/{time.strftime("%Y%m%d-%H%M%S")}_{model_name}'
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(parent_path / 'models', exist_ok=True)

    # ---- Training rounds ----
    train_losses = []
    comm_stats_per_round = []

    # Create clients (lazy creation on first use, then cached)
    client_cache = {}

    for round_idx in tqdm(range(args.rounds), desc='Federated Rounds'):
        print(f'\n{"="*60}')
        print(f'Global Training Round [{round_idx}]/[{args.rounds-1}]')
        print(f'{"="*60}')

        # Sample participating clients (C fraction)
        m = max(int(args.frac * args.num_users), 1)
        user_indices = np.random.choice(range(args.num_users), m, replace=False).tolist()
        
        # Collect client updates
        local_factors_list = []  # [(client_id, factors), ...]
        local_losses = []
        data_sizes = {}
        active_ranks = {}

        for i, client_id in enumerate(user_indices):
            print(f'\n--- Client {client_id} ---')
            
            # Create or retrieve cached client
            if client_id not in client_cache:
                client_cache[client_id] = LoRAClient(
                    args=args,
                    dataset=train_dataset,
                    base_model=base_model_clean,  # clean model; LoRAClient injects local rank itself
                    indices=client_groups[client_id],
                    time_steps=int(args.time_steps),
                    diffuser=diffuser,
                    lora_rank=client_rank_map.get(client_id, 8),
                    lora_alpha=getattr(args, 'lora_alpha', 1.0),
                )
            
            client = client_cache[client_id]

            # Receive server prefix distribution. Round 0 also receives the initial prefix distribution.
            server_upd = getattr(aggregator, '_last_distribution', {}).get(client_id)
            if server_upd is not None:
                B_serv, A_serv = server_upd
                server_factors = [
                    (layer_name, B_serv[layer_name], A_serv[layer_name])
                    for layer_name in B_serv
                ]
                client.receive_server_update(server_factors)
            else:
                client.receive_server_update(None)

            # Local training
            factors, loss = client.train_local()
            local_factors_list.append((client_id, factors))
            local_losses.append(loss)
            
            data_sizes[client_id] = len(client_groups[client_id])
            active_ranks[client_id] = client_rank_map.get(client_id, 8)

        # ---- Server-side aggregation ----
        agg_result = aggregator.aggregate_round(
            client_factors_list=local_factors_list,
            data_sizes=data_sizes,
            client_ranks=active_ranks,
        )
        comm_stats_per_round.append(agg_result['stats'])

        # Cache distribution for next round's client receive step
        aggregator._last_distribution = agg_result['client_updates']

        # Apply aggregated global factors back to server_model for checkpointing/sampling
        set_all_lora_factors(server_model, [
            (name, agg_result['global_B'][name], agg_result['global_A'][name])
            for name in agg_result['global_B']
        ])

        avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0
        train_losses.append(avg_loss)

        print(f'\nRound {round_idx} summary:')
        print(f'  Avg loss: {avg_loss:.4f}')
        print(f'  Upload bytes: {agg_result["stats"].get("upload_bytes", 0)/1024:.1f} KB')
        print(f'  Download bytes: {agg_result["stats"].get("download_bytes", 0)/1024:.1f} KB')
        elapsed = time.time() - start_time
        print(f'  Wall-clock: {elapsed:.1f}s')

        # ---- Save checkpoints ----
        if round_idx % max(1, args.rounds // 5) == 0 or round_idx == args.rounds - 1:
            ckpt_path = result_folder / f'{model_name}_R[{round_idx}].pth'
            save_lora_checkpoint(server_model, ckpt_path, round_idx)

        # ---- Intermediate sampling ----
        if (args.exp_rounds > 0 and 
            round_idx >= 5 and 
            round_idx % 2 == 1):
            sample_folder = parent_path / f'exports/{model_name}_R[{round_idx}]'
            export_samples(
                diffuser, sample_folder,
                bool(args.conditional), server_model, int(args.time_steps),
                args.image_size, args.num_channels, args.num_classes,
                args.exp_rounds,
                use_ddim=bool(getattr(args, 'use_ddim', 1)),
                ddim_steps=getattr(args, 'ddim_steps', 100)
            )

    # ---- Final save ----
    final_path = result_folder / f'{model_name}.pth'
    save_lora_checkpoint(server_model, final_path)
    
    # Save to project root with hyperparam tags
    _lr = getattr(args, 'lora_rank', 8)
    final_model_name = f'flora_model_{args.dataset}_R[{args.rounds}]_K[{args.num_users}]_E[{args.local_ep}]'
    final_model_path = parent_path / f'{final_model_name}.pth'
    save_lora_checkpoint(server_model, final_model_path)
    print(f'\nFinal model saved: {final_model_path}')

    # ---- Export loss CSV ----
    csv_path = result_folder / f'{model_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'loss', 'upload_kb', 'download_kb'])
        for idx, (loss, stats) in enumerate(zip(train_losses, comm_stats_per_round)):
            writer.writerow([
                idx,
                f'{loss:.6f}',
                stats.get('upload_bytes', 0) / 1024,
                stats.get('download_bytes', 0) / 1024,
            ])
        writer.writerow(['runtime', time.time() - start_time])

    print(f'\n{"="*60}')
    print(f'Training Complete!')
    
    # Calculate and display total communication volume
    total_up = sum(s.get('upload_bytes', 0) for s in comm_stats_per_round)
    total_down = sum(s.get('download_bytes', 0) for s in comm_stats_per_round)
    total_total = total_up + total_down
    
    print(f'Total Communication Volume:')
    print(f'  Upload   : {total_up / (1024*1024):.2f} MB')
    print(f'  Download : {total_down / (1024*1024):.2f} MB')
    print(f'  Total    : {total_total / (1024*1024):.2f} MB')
    
    print(f'\nTotal runtime: {time.time()-start_time:.1f}s')
    print(f'Model saved to: {final_path}')
    print(f'Results folder: {result_folder}')
    print(f'Loss CSV: {csv_path}')
    print(f'{"="*60}')


def run_inference(args):
    """Run inference / sampling only (no training)."""
    parent_path = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Set custom data root if provided (same as run_training) ----
    data_root = getattr(args, 'data_root', '')
    if data_root:
        import utils as _utils
        _utils._CUSTOM_DATA_ROOT = data_root
        print(f'Using custom data root: {data_root}')

    image_size = args.image_size
    is_conditional = args.conditional == 1
    channels = args.num_channels

    # ---- Build and load model ----
    lora_rank = getattr(args, 'lora_rank', 8)

    # Load the saved model (already contains base + trained LoRA)
    load_path = parent_path / args.load_model if args.load_model else None
    if load_path and os.path.isfile(load_path):
        print(f'Loading trained LoRA model from {load_path}')
        model = torch.load(load_path, map_location=torch.device(device), weights_only=False)
        if isinstance(model, dict):
            raise ValueError('Expected full model object, got state_dict. '
                             'Use a .pth saved by save_lora_checkpoint().')
        model = model.to(device)
        model.eval()
        print(f'Loaded model with {sum(p.numel() for p in model.parameters()):,} params')
    else:
        # Fallback: build from scratch + inject LoRA (untrained)
        print('[Warning] No trained model found, building untrained model')
        model = build_base_model(args, device)
        inject_lora_into_unet(model, rank=lora_rank,
                              alpha=getattr(args, 'lora_alpha', 1.0),
                              target_layers='attention')
        model.eval()

    # ---- Diffuser ----
    diffuser = Diffuser(int(args.time_steps))

    # Derive export folder name from loaded model (or default)
    if args.load_model:
        _stem = Path(args.load_model).stem
        model_name = f'infer_{_stem}'
    else:
        model_name = f'lora_{args.dataset}_inference'

    # ---- Export samples ----
    if args.export_samples > 0:
        export_dir = parent_path / f'exports/{model_name}'
        export_samples(diffuser, export_dir, is_conditional, model,
                       int(args.time_steps), image_size, channels,
                       args.num_classes, args.export_samples,
                       use_ddim=bool(getattr(args, 'use_ddim', 1)),
                       ddim_steps=getattr(args, 'ddim_steps', 100))
        print(f'Exported {args.export_samples} samples to {export_dir}')

    # ---- Show samples ----
    if args.show_samples > 0:
        train_dataset, _, _ = get_partitioned_dataset(args)
        show_samples(diffuser, is_conditional, model, train_dataset,
                     int(args.time_steps), image_size, channels,
                     use_ddim=bool(getattr(args, 'use_ddim', 1)),
                     ddim_steps=getattr(args, 'ddim_steps', 100))

    # ---- Export dataset samples (for FID reference) ----
    if args.export_dataset > 0:
        real_dir = parent_path / f'exports/{args.dataset}/dataset'
        export_dataset(real_dir, args.dataset, args.export_dataset, train=False)
        print(f'Exported {args.export_dataset} real dataset samples to {real_dir}')

    # ---- Auto FID evaluation (when both real and fake exist) ----
    if args.export_samples > 0 and args.export_dataset > 0:
        fake_dir = parent_path / f'exports/{model_name}'
        real_dir = parent_path / f'exports/{args.dataset}/dataset'
        print(f'\n{"="*50}')
        print(f'Computing FID & Precision/Recall...')
        print(f'  Real (reference): {real_dir}')
        print(f'  Fake (generated): {fake_dir}')
        print(f'{"="*50}')

        import datetime as _dt
        _ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        _lora_rank = getattr(args, 'lora_rank', 8)
        _global_rank = getattr(args, 'global_lora_rank', _lora_rank)
        _iid_tag = f'I[{args.iid},{args.unequal}]'
        _tag = (f'lora_{args.dataset}_R[{args.rounds}]_K[{args.num_users}]'
                f'_r[{_lora_rank}g{_global_rank}]_E[{args.local_ep}]'
                f'_B[{args.local_bs}]_T[{int(args.time_steps)}]'
                f'_{_iid_tag}')
        _eval_log_dir = parent_path / 'results' / 'eval_logs'

        perform_evaluation(real_path=str(real_dir), fake_path=str(fake_dir),
                           num_samples=min(args.export_samples, args.export_dataset),
                           eval_log_dir=str(_eval_log_dir), config_tag=_tag)


def add_lora_arguments(parser):
    """Add LoRA-specific arguments to the argument parser."""
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='Base LoRA rank for all clients (default: 8)')
    parser.add_argument('--lora_ranks', type=str, default='',
                        help='Comma-separated per-client ranks (e.g., "4,8,16,8,4"). '
                             'Overrides lora_rank if set.')
    parser.add_argument('--global_lora_rank', type=int, default=16,
                        help='Server-side global LoRA rank for aggregation (default: 16)')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA scaling factor alpha (default: 1.0)')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='Dropout probability for LoRA path (default: 0.0)')


def main(external_args=None):
    import argparse as _argparse

    # Parse base arguments
    p = _argparse.ArgumentParser(description='Heterogeneous LoRA Federated Diffusion Training')

    # Federated args
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--num_users', type=int, default=5)
    p.add_argument('--frac', type=float, default=1)
    p.add_argument('--local_ep', type=int, default=3)
    p.add_argument('--local_bs', type=int, default=128)
    p.add_argument('--train_mode', type=str, default='full')  # unused but kept for compat

    # Diffusion args
    p.add_argument('--train', type=int, default=1)
    p.add_argument('--load_model', type=str, default='')
    p.add_argument('--time_steps', type=float, default=1000)
    p.add_argument('--conditional', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.5)
    p.add_argument('--optimizer', type=str, default='adam')

    # Data args
    p.add_argument('--dataset', type=str, default='fmnist')
    p.add_argument('--data_root', type=str, default='',
                    help='Custom data root path (overrides default). '
                         'E.g., --data_root=D:\\\\data\\\\CelebA')
    p.add_argument('--image_size', type=int, default=28)
    p.add_argument('--num_channels', type=int, default=1)
    p.add_argument('--iid', type=int, default=1)
    p.add_argument('--unequal', type=int, default=0)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--round_offset', type=int, default=0)

    # Export args
    p.add_argument('--export_samples', type=int, default=0)
    p.add_argument('--export_dataset', type=int, default=0)
    p.add_argument('--show_samples', type=int, default=0)
    p.add_argument('--exp_rounds', type=int, default=0)

    # Add LoRA-specific args
    add_lora_arguments(p)

    if external_args is not None:
        args = external_args
    else:
        args = p.parse_args()

    if args.train == 1:
        run_training(args)
    else:
        run_inference(args)


if __name__ == '__main__':
    main()
