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
import json
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
from diffusers_backend import DiffusersDiffuser, DiffusersUNetWrapper
from hetero_aggregation import HeteroLoRAAggregator, prefix_slice_distribution
from hetero_aggregation import count_client_distribution_bytes, count_factor_list_bytes
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


def parse_dim_mults(dim_mults_str: str):
    if not dim_mults_str:
        return (1, 2, 4)
    return tuple(int(x.strip()) for x in dim_mults_str.split(',') if x.strip())


def resolve_lora_alpha(args, rank: int) -> float:
    """
    Default alpha=rank so LoRA scaling is 1 and the effective update is B @ A.
    """
    mode = getattr(args, 'lora_alpha_mode', 'rank')
    if mode == 'rank':
        return float(rank)
    if mode == 'fixed':
        return float(getattr(args, 'lora_alpha', 1.0))
    raise ValueError(f'Unsupported lora_alpha_mode={mode}')


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


def _pad_factor_to_rank(B: torch.Tensor, A: torch.Tensor, target_rank: int):
    r = B.shape[1]
    if r == target_rank:
        return B, A
    if r > target_rank:
        raise ValueError(f'Cannot pad rank {r} down to target_rank={target_rank}')
    B_pad = torch.zeros(B.shape[0], target_rank, dtype=B.dtype, device=B.device)
    A_pad = torch.zeros(target_rank, A.shape[1], dtype=A.dtype, device=A.device)
    B_pad[:, :r] = B
    A_pad[:r, :] = A
    return B_pad, A_pad


def average_lora_factors_to_global(client_factors_list, data_sizes, target_rank, pad_to_rank):
    """
    Direct factor-space averaging baseline.

    Use pad_to_rank=False for homogeneous FedAvg-LoRA and pad_to_rank=True for
    heterogeneous naive factor averaging.
    """
    total = sum(data_sizes[cid] for cid, _ in client_factors_list)
    if total <= 0:
        raise ValueError('Cannot average LoRA factors with zero total data size.')

    B_global = {}
    A_global = {}
    expected_shapes = {}

    for cid, factors in client_factors_list:
        weight = data_sizes[cid] / total
        for layer_name, B_dict, A_dict in factors:
            if layer_name not in B_global:
                B_global[layer_name] = {}
                A_global[layer_name] = {}
            for key in B_dict:
                B = B_dict[key]
                A = A_dict[key]
                if pad_to_rank:
                    B, A = _pad_factor_to_rank(B, A, target_rank)
                else:
                    shape_key = (layer_name, key)
                    shape = (tuple(B.shape), tuple(A.shape))
                    if shape_key in expected_shapes and expected_shapes[shape_key] != shape:
                        raise ValueError(
                            'fedavg_lora requires homogeneous factor shapes. '
                            f'Layer {layer_name}/{key} got {shape}, expected {expected_shapes[shape_key]}.'
                        )
                    expected_shapes[shape_key] = shape

                if key not in B_global[layer_name]:
                    B_global[layer_name][key] = torch.zeros_like(B)
                    A_global[layer_name][key] = torch.zeros_like(A)
                B_global[layer_name][key] += weight * B
                A_global[layer_name][key] += weight * A

    return B_global, A_global


def build_base_model(args, device: str):
    """
    Build and optionally load the base U-Net model (WITHOUT LoRA injection).
    LoRA injection happens later via inject_lora_into_unet().
    """
    model_backend = getattr(args, 'model_backend', 'custom')
    if model_backend == 'diffusers':
        model_id = getattr(args, 'hf_model_id', 'google/ddpm-cifar10-32')
        print(f'[Model] diffusers backend: {model_id}')
        return DiffusersUNetWrapper(model_id=model_id).to(device)

    is_conditional = args.conditional == 1
    channels = args.num_channels
    image_size = args.image_size
    model_dim = int(getattr(args, 'model_dim', image_size))
    if model_dim <= 0:
        model_dim = image_size
    dim_mults = parse_dim_mults(getattr(args, 'dim_mults', '1,2,4'))

    model = (
        UnetConditional(
            dim=model_dim,
            channels=channels,
            dim_mults=dim_mults,
            num_classes=args.num_classes,
        ) if is_conditional else Unet(
            dim=model_dim,
            channels=channels,
            dim_mults=dim_mults,
        )
    )
    print(f'[Model] image_size={image_size}, model_dim={model_dim}, dim_mults={dim_mults}, conditional={is_conditional}')

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


def build_diffuser(args):
    if getattr(args, 'model_backend', 'custom') == 'diffusers':
        return DiffusersDiffuser(
            model_id=getattr(args, 'hf_model_id', 'google/ddpm-cifar10-32'),
            time_steps=int(args.time_steps),
        )
    return Diffuser(int(args.time_steps))


def save_lora_checkpoint(model, path: Path, round_num: int = 0):
    """Save a complete checkpoint (base + LoRA params)."""
    # Save full model for sampling/inference
    torch.save(model, path)
    print(f'Saved checkpoint: {path}')


def save_lora_metadata(path: Path, model, args, client_rank_map, comm_stats, round_num: int):
    """Save reproducibility metadata without changing the full-model checkpoint format."""
    factors = extract_all_lora_factors(model)
    torch.save({
        'round': round_num,
        'global_lora_factors': factors,
        'client_ranks': dict(client_rank_map),
        'communication_stats': list(comm_stats),
        'experiment_args': vars(args) if hasattr(args, '__dict__') else {},
    }, path)
    print(f'Saved LoRA metadata: {path}')


def save_experiment_protocol(result_folder: Path, args, data_stats, num_params, lora_params):
    protocol = {
        "protocol_name": "FedPhD-protocol-aligned FDLR",
        "dataset": args.dataset,
        "model_backend": getattr(args, "model_backend", "custom"),
        "hf_model_id": getattr(args, "hf_model_id", ""),
        "partition": getattr(args, "partition", ""),
        "num_users": args.num_users,
        "frac": args.frac,
        "rounds": args.rounds,
        "local_ep": args.local_ep,
        "local_bs": args.local_bs,
        "image_size": args.image_size,
        "model_dim": getattr(args, "model_dim", None),
        "dim_mults": getattr(args, "dim_mults", None),
        "num_channels": args.num_channels,
        "conditional": args.conditional,
        "time_steps": args.time_steps,
        "use_ddim": getattr(args, "use_ddim", 1),
        "ddim_steps": getattr(args, "ddim_steps", 100),
        "eval_num_samples": getattr(args, "eval_num_samples", 30000),
        "eval_batch_size": getattr(args, "eval_batch_size", 256),
        "compute_is": getattr(args, "compute_is", 1),
        "eval_real_split": getattr(args, "eval_real_split", "train"),
        "data_range": getattr(args, "data_range", "minus1_1"),
        "resume_model": getattr(args, "resume_model", ""),
        "start_round": getattr(args, "start_round", 0),
        "cache_clients": getattr(args, "cache_clients", 0),
        "lora_ranks": getattr(args, "lora_ranks", ""),
        "global_lora_rank": getattr(args, "global_lora_rank", None),
        "lora_alpha_mode": getattr(args, "lora_alpha_mode", "rank"),
        "rank_beta": getattr(args, "rank_beta", 0.5),
        "rank_correction": getattr(args, "rank_correction", 1),
        "use_procrustes": getattr(args, "use_procrustes", 1),
        "use_prefix_init": getattr(args, "use_prefix_init", 1),
        "agg_mode": getattr(args, "agg_mode", "fdlr"),
        "seed": getattr(args, "seed", 2023),
        "total_params": int(num_params),
        "trainable_params": int(lora_params),
        "data_stats": {
            str(k): {str(kk): int(vv) for kk, vv in val.items()}
            for k, val in data_stats.items()
        },
    }
    path = result_folder / "fedphd_protocol_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)
    print(f"Saved FedPhD protocol config: {path}")


def cleanup_fixed_result_folder(result_folder: Path):
    """Remove generated artifacts for the same fixed experiment name before a new run."""
    if not result_folder.exists():
        return
    for pattern in ('*.pth', '*.metadata.pt', '*.json', '*.csv', '*.log'):
        for path in result_folder.glob(pattern):
            if path.is_file():
                path.unlink()


def run_training(args):
    """Main LoRA federated training loop."""
    start_time = time.time()
    parent_path = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    seed = int(getattr(args, 'seed', 2023))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(getattr(args, 'disable_cudnn', 0)):
        torch.backends.cudnn.enabled = False
        print('[Info] cuDNN disabled for this run.')

    exp_details(args)

    # ---- Set custom data root if provided ----
    data_root = getattr(args, 'data_root', '')
    download_dataset = bool(getattr(args, 'download_dataset', 0))
    import utils as _utils
    if data_root:
        # Override CelebA data path via module-level patch
        _utils._CUSTOM_DATA_ROOT = data_root
        print(f'Using custom data root: {data_root}')
    _utils._DOWNLOAD_DATASET = download_dataset
    _utils._PARTITION_RULE = getattr(args, 'partition', '')
    _utils._DATA_RANGE = getattr(args, 'data_range', 'minus1_1')

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
            if (
                not isinstance(_ck, dict)
                and (
                    hasattr(_ck, '_lora_config')
                    or any('lora_' in name for name, _ in _ck.named_parameters())
                )
            ):
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
    server_alpha = resolve_lora_alpha(args, global_rank)
    if not _is_lora_checkpoint:
        inject_lora_into_unet(
            server_model,
            rank=global_rank,
            alpha=server_alpha,
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
            alpha=server_alpha,
            dropout=getattr(args, 'lora_dropout', 0.0),
            target_layers=getattr(args, 'lora_target_layers', 'attention'),
        )

    resume_factors = None
    resume_model = getattr(args, 'resume_model', '')
    if resume_model:
        resume_path = parent_path / resume_model
        if not resume_path.exists():
            raise FileNotFoundError(f'--resume_model not found: {resume_path}')
        print(f'[Resume] Loading LoRA factors from {resume_path}')
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(resume_checkpoint, dict):
            raise ValueError('--resume_model should be a full LoRA model checkpoint, not a state_dict.')
        resume_factors = extract_all_lora_factors(resume_checkpoint)
        if not resume_factors:
            raise ValueError(f'No LoRA factors found in resume checkpoint: {resume_path}')
        set_all_lora_factors(server_model, resume_factors)
        del resume_checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'[Resume] Applied {len(resume_factors)} LoRA layers to server model.')

    num_params = sum(p.numel() for p in server_model.parameters())
    lora_params = sum(p.numel() for p in server_model.parameters() if p.requires_grad)
    print(f'\nServer model: {num_params:,} total params, {lora_params:,} global LoRA trainable '
          f'({100*lora_params/num_params:.2f}%)')

    # ---- Initialize diffuser ----
    diffuser = build_diffuser(args)

    # ---- Heterogeneous rank configuration ----
    print(f'\nClient LoRA rank config:')
    for cid, r in sorted(client_rank_map.items()):
        n_data = len(client_groups.get(cid, []))
        print(f'  Client {cid}: rank={r}, samples={n_data}')

    # ---- Create aggregator ----
    agg_mode = getattr(args, 'agg_mode', 'fdlr')
    use_procrustes = bool(getattr(args, 'use_procrustes', 1))
    if agg_mode == 'update_space':
        use_procrustes = False

    aggregator = HeteroLoRAAggregator(
        global_rank=global_rank,
        rank_correction=bool(getattr(args, 'rank_correction', 1)),
        use_procrustes=use_procrustes,
        svd_method='rsvd',
        rank_beta=float(getattr(args, 'rank_beta', 0.5)),
    )

    # Initialize client LoRA factors from the same server global subspace, even in round 0.
    # This is necessary for nested prefix distribution to be true from the beginning.
    init_factors = resume_factors if resume_factors is not None else extract_all_lora_factors(server_model)
    B_init, A_init = factors_list_to_nested_dict(init_factors)
    if bool(getattr(args, 'use_prefix_init', 1)):
        aggregator._last_distribution = prefix_slice_distribution(B_init, A_init, client_rank_map)
    else:
        aggregator._last_distribution = {}

    # ---- Setup output directories ----
    final_model_name = f'flora_model_{args.dataset}_R[{args.rounds}]_K[{args.num_users}]_E[{args.local_ep}]'
    model_name = final_model_name
    result_folder = parent_path / 'results' / final_model_name
    os.makedirs(result_folder, exist_ok=True)
    is_resume_run = bool(getattr(args, 'resume_model', '')) or int(getattr(args, 'start_round', 0)) > 0
    if not is_resume_run:
        cleanup_fixed_result_folder(result_folder)
    os.makedirs(parent_path / 'models', exist_ok=True)
    save_experiment_protocol(result_folder, args, data_stats, num_params, lora_params)

    # ---- Training rounds ----
    train_losses = []
    comm_stats_per_round = []

    # Create clients (lazy creation on first use, then cached)
    cache_clients = bool(getattr(args, 'cache_clients', 0))
    client_cache = {}
    agg_round_num = 0

    start_round = int(getattr(args, 'start_round', 0))
    if start_round:
        print(f'[Resume] Continuing from round {start_round} to {args.rounds - 1}')
        agg_round_num = start_round

    for round_idx in tqdm(range(start_round, args.rounds), desc='Federated Rounds'):
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
        client_trainable_counts = {}

        for i, client_id in enumerate(user_indices):
            print(f'\n--- Client {client_id} ---')
            
            # Create or retrieve cached client
            if cache_clients and client_id in client_cache:
                client = client_cache[client_id]
            else:
                client_rank = client_rank_map.get(client_id, 8)
                client_alpha = resolve_lora_alpha(args, client_rank)
                client = LoRAClient(
                    args=args,
                    dataset=train_dataset,
                    base_model=base_model_clean,  # clean model; LoRAClient injects local rank itself
                    indices=client_groups[client_id],
                    time_steps=int(args.time_steps),
                    diffuser=diffuser,
                    lora_rank=client_rank,
                    lora_alpha=client_alpha,
                )
                if cache_clients:
                    client_cache[client_id] = client

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
            client_trainable_counts[client_id] = int(client.get_trainable_param_count())
            if not cache_clients:
                del client
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---- Server-side aggregation ----
        agg_t0 = time.time()
        if agg_mode in ('fedavg_lora', 'factor_avg'):
            B_global, A_global = average_lora_factors_to_global(
                local_factors_list,
                data_sizes=data_sizes,
                target_rank=global_rank,
                pad_to_rank=(agg_mode == 'factor_avg'),
            )
            client_updates = prefix_slice_distribution(B_global, A_global, client_rank_map)
            agg_result = {
                'global_B': B_global,
                'global_A': A_global,
                'client_updates': client_updates,
                'stats': {
                    'round': agg_round_num,
                    'num_clients': len(local_factors_list),
                    'agg_mode': agg_mode,
                    'upload_bytes': sum(count_factor_list_bytes(factors) for _, factors in local_factors_list),
                    'download_bytes': count_client_distribution_bytes(client_updates),
                },
            }
            agg_round_num += 1
        else:
            agg_result = aggregator.aggregate_round(
                client_factors_list=local_factors_list,
                data_sizes=data_sizes,
                client_ranks=active_ranks,
            )
        agg_time = time.time() - agg_t0
        round_wall = time.time() - start_time
        agg_result['stats']['server_aggregation_time_sec'] = float(agg_time)
        agg_result['stats']['round_wall_clock_sec'] = float(round_wall)
        agg_result['stats']['client_trainable_params'] = {
            str(cid): client_trainable_counts.get(cid, 0)
            for cid in user_indices
        }
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
            ckpt_path = result_folder / f'{final_model_name}_round[{round_idx}].pth'
            save_lora_checkpoint(server_model, ckpt_path, round_idx)
            save_lora_metadata(
                ckpt_path.with_suffix('.metadata.pt'),
                server_model, args, client_rank_map, comm_stats_per_round, round_idx
            )

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
                ddim_steps=getattr(args, 'ddim_steps', 100),
                batch_size=getattr(args, 'eval_batch_size', 256),
                data_range=getattr(args, 'data_range', 'minus1_1'),
            )

    # ---- Final save ----
    final_path = result_folder / f'{final_model_name}.pth'
    save_lora_checkpoint(server_model, final_path)
    save_lora_metadata(
        final_path.with_suffix('.metadata.pt'),
        server_model, args, client_rank_map, comm_stats_per_round, args.rounds - 1
    )
    
    # Save to project root with hyperparam tags
    final_model_path = parent_path / f'{final_model_name}.pth'
    root_metadata_path = final_model_path.with_suffix('.metadata.pt')
    if root_metadata_path.exists():
        root_metadata_path.unlink()
    save_lora_checkpoint(server_model, final_model_path)
    print(f'\nFinal model saved: {final_model_path}')

    # ---- Export training/communication log ----
    central_interval = int(getattr(args, 'central_agg_interval', 5))
    eval_log_dir = parent_path / 'results' / 'eval_logs'
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    log_ts = time.strftime('%Y%m%d-%H%M%S')
    train_log_path = eval_log_dir / f'{final_model_name}_train_{log_ts}.log'
    with open(train_log_path, 'w', encoding='utf-8') as f:
        f.write(f'Training Log  {log_ts}\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Model name    : {final_model_name}\n')
        f.write(f'Dataset       : {args.dataset}\n')
        f.write(f'Rounds        : {args.rounds}\n')
        f.write(f'Clients       : {args.num_users}\n')
        f.write(f'Local epochs  : {args.local_ep}\n')
        f.write(f'Local batch   : {args.local_bs}\n')
        f.write(f'Results folder: {result_folder}\n')
        f.write(f'Root model    : {final_model_path}\n')
        f.write(f'Result model  : {final_path}\n\n')
        f.write('[Per-round]\n')
        f.write('round\tloss\tupload_kb\tdownload_kb\tserver_aggregation_time_sec\tround_wall_clock_sec\n')
        for idx, (loss, stats) in enumerate(zip(train_losses, comm_stats_per_round)):
            round_num = start_round + idx
            f.write(
                f'{round_num}\t'
                f'{loss:.6f}\t'
                f'{stats.get("upload_bytes", 0) / 1024:.6f}\t'
                f'{stats.get("download_bytes", 0) / 1024:.6f}\t'
                f'{stats.get("server_aggregation_time_sec", 0.0):.6f}\t'
                f'{stats.get("round_wall_clock_sec", 0.0):.6f}\n'
            )
        f.write(f'\nruntime_sec\t{time.time() - start_time:.6f}\n\n')
        f.write('[Central-window communication]\n')
        f.write('window_start_round\tupload_MB\tdownload_MB\ttotal_MB\n')
        for start in range(0, len(comm_stats_per_round), central_interval):
            chunk = comm_stats_per_round[start:start + central_interval]
            up = sum(s.get('upload_bytes', 0) for s in chunk)
            down = sum(s.get('download_bytes', 0) for s in chunk)
            f.write(
                f'{start_round + start}\t'
                f'{up / (1024 * 1024):.6f}\t'
                f'{down / (1024 * 1024):.6f}\t'
                f'{(up + down) / (1024 * 1024):.6f}\n'
            )

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
    print(f'Root model: {final_model_path}')
    print(f'Result model: {final_path}')
    print(f'Results folder: {result_folder}')
    print(f'Training log: {train_log_path}')
    print(f'{"="*60}')


def run_inference(args):
    """Run inference / sampling only (no training)."""
    parent_path = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Set custom data root if provided (same as run_training) ----
    data_root = getattr(args, 'data_root', '')
    download_dataset = bool(getattr(args, 'download_dataset', 0))
    import utils as _utils
    if data_root:
        _utils._CUSTOM_DATA_ROOT = data_root
        print(f'Using custom data root: {data_root}')
    _utils._DOWNLOAD_DATASET = download_dataset
    _utils._PARTITION_RULE = getattr(args, 'partition', '')

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
                              alpha=resolve_lora_alpha(args, lora_rank),
                              target_layers='attention')
        model = model.to(device)
        model.eval()

    # ---- Diffuser ----
    diffuser = build_diffuser(args)

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
                       ddim_steps=getattr(args, 'ddim_steps', 100),
                       batch_size=getattr(args, 'eval_batch_size', 256),
                       data_range=getattr(args, 'data_range', 'minus1_1'))
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
        real_split = getattr(args, 'eval_real_split', 'train')
        real_train = real_split == 'train'
        real_dir = parent_path / f'exports/{args.dataset}/dataset_{real_split}'
        export_dataset(real_dir, args.dataset, args.export_dataset, train=real_train,
                       data_range=getattr(args, 'data_range', 'minus1_1'))
        print(f'Exported {args.export_dataset} real {real_split} samples to {real_dir}')

    # ---- Auto FID evaluation (when both real and fake exist) ----
    if args.export_samples > 0 and args.export_dataset > 0:
        fake_dir = parent_path / f'exports/{model_name}'
        real_split = getattr(args, 'eval_real_split', 'train')
        real_dir = parent_path / f'exports/{args.dataset}/dataset_{real_split}'
        print(f'\n{"="*50}')
        print(f'Computing FID, IS & Precision/Recall...')
        print(f'  Real (reference): {real_dir}')
        print(f'  Fake (generated): {fake_dir}')
        print(f'{"="*50}')

        import datetime as _dt
        _ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        _lora_rank = getattr(args, 'lora_rank', 8)
        _global_rank = getattr(args, 'global_lora_rank', _lora_rank)
        _agg_mode = getattr(args, 'agg_mode', 'fdlr')
        if getattr(args, 'load_model', ''):
            _model_stem = Path(args.load_model).stem
            _tag = f'eval_lora_{args.dataset}_{_model_stem}_{_agg_mode}_{_ts}.log'
        else:
            _tag = (f'eval_lora_{args.dataset}_R[{args.rounds}]_K[{args.num_users}]'
                    f'_r[{_lora_rank}g{_global_rank}]_E[{args.local_ep}]'
                    f'_B[{args.local_bs}]_{_agg_mode}_{_ts}.log')
        _eval_log_dir = parent_path / 'results' / 'eval_logs'
        _lora_ranks = getattr(args, 'lora_ranks', '')
        _client_mode = 'Heterogeneous' if _lora_ranks else 'Homogeneous'
        _experiment_meta = {
            'Method': 'lora',
            'Load model': getattr(args, 'load_model', '') or '(none)',
            'Agg mode': getattr(args, 'agg_mode', 'fdlr'),
            'Data dist.': 'IID' if getattr(args, 'iid', 0) else 'Non-IID',
            'Partition': getattr(args, 'partition', '') or '(default)',
            'Client mode': _client_mode,
            'LoRA ranks': _lora_ranks or str(getattr(args, 'lora_rank', 8)),
        }

        perform_evaluation(real_path=str(real_dir), fake_path=str(fake_dir),
                           num_samples=min(
                               int(getattr(args, 'eval_num_samples', 30000)),
                               args.export_samples,
                               args.export_dataset,
                           ),
                           eval_log_dir=str(_eval_log_dir), config_tag=_tag,
                           batch_size=getattr(args, 'eval_batch_size', 256),
                           compute_is=bool(getattr(args, 'compute_is', 1)),
                           experiment_meta=_experiment_meta)


def add_lora_arguments(parser):
    """Add LoRA-specific arguments to the argument parser."""
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='Base LoRA rank for all clients (default: 8)')
    parser.add_argument('--lora_ranks', type=str, default='',
                        help='Comma-separated per-client ranks (e.g., "4,8,16,8,4"). '
                             'Overrides lora_rank if set.')
    parser.add_argument('--global_lora_rank', type=int, default=16,
                        help='Server-side global LoRA rank for aggregation (default: 16)')
    parser.add_argument('--lora_alpha', type=float, default=-1.0,
                        help='LoRA alpha. Use <=0 for alpha=rank, so scaling=1 (default: -1).')
    parser.add_argument('--lora_alpha_mode', type=str, default='rank',
                        choices=['rank', 'fixed'],
                        help='rank: alpha=rank so LoRA scaling=1; fixed: use --lora_alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='Dropout probability for LoRA path (default: 0.0)')
    parser.add_argument('--rank_beta', type=float, default=0.5,
                        help='Rank correction beta in eta(r)=r^(-beta). Use 0, 0.5, or 1 for ablations.')
    parser.add_argument('--rank_correction', type=int, default=1,
                        help='Whether to use rank correction in aggregation.')
    parser.add_argument('--use_procrustes', type=int, default=1,
                        help='Whether to use Procrustes alignment.')
    parser.add_argument('--use_prefix_init', type=int, default=1,
                        help='Whether clients receive prefix initialization before round 0.')
    parser.add_argument('--agg_mode', type=str, default='fdlr',
                        choices=['fedavg_lora', 'factor_avg', 'update_space', 'fdlr'],
                        help='Aggregation mode. Currently fdlr and update_space are implemented.')


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
    p.add_argument('--resume_model', type=str, default='',
                   help='Full LoRA checkpoint to resume from; keeps --load_model as clean base.')
    p.add_argument('--start_round', type=int, default=0,
                   help='First round index to run when resuming, e.g. 19 after R[18].')
    p.add_argument('--cache_clients', type=int, default=0,
                   help='Cache client models across rounds. 0 saves memory and is recommended for long runs.')
    p.add_argument('--time_steps', type=float, default=1000)
    p.add_argument('--conditional', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.5)
    p.add_argument('--optimizer', type=str, default='adam')

    # Data args
    p.add_argument('--dataset', type=str, default='fmnist')
    p.add_argument('--model_backend', type=str, default='custom',
                   choices=['custom', 'diffusers'],
                   help='custom uses src/unet.py; diffusers uses Hugging Face UNet2DModel.')
    p.add_argument('--hf_model_id', type=str, default='google/ddpm-cifar10-32',
                   help='Hugging Face model id for --model_backend=diffusers.')
    p.add_argument('--disable_cudnn', type=int, default=0,
                   help='Disable cuDNN kernels. Useful for CUDNN_STATUS_INTERNAL_ERROR workarounds.')
    p.add_argument('--data_root', type=str, default='',
                    help='Custom data root path (overrides default). '
                         'E.g., --data_root=D:\\\\data\\\\CelebA')
    p.add_argument('--download_dataset', type=int, default=0,
                   help='Whether torchvision should download the dataset.')
    p.add_argument('--partition', type=str, default='',
                   help='Partition rule: fedphd-cifar2, fedphd-celeba4, or empty for existing split.')
    p.add_argument('--image_size', type=int, default=28)
    p.add_argument('--num_channels', type=int, default=1)
    p.add_argument('--iid', type=int, default=1)
    p.add_argument('--unequal', type=int, default=0)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--round_offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--model_dim', type=int, default=0,
                   help='Base channel dimension of U-Net. If 0, fallback to image_size.')
    p.add_argument('--dim_mults', type=str, default='1,2,4',
                   help='Comma-separated U-Net dim multipliers, e.g. 1,2,2,2.')

    # Export args
    p.add_argument('--export_samples', type=int, default=0)
    p.add_argument('--export_dataset', type=int, default=0)
    p.add_argument('--show_samples', type=int, default=0)
    p.add_argument('--exp_rounds', type=int, default=0)
    p.add_argument('--use_ddim', type=int, default=1,
                   help='Use DDIM sampling for evaluation/export.')
    p.add_argument('--ddim_steps', type=int, default=100,
                   help='DDIM sampling steps.')
    p.add_argument('--eval_num_samples', type=int, default=30000,
                   help='Number of generated samples for FedPhD-aligned evaluation.')
    p.add_argument('--eval_batch_size', type=int, default=256,
                   help='Batch size for FedPhD-aligned generation/evaluation.')
    p.add_argument('--central_agg_interval', type=int, default=5,
                   help='For FedPhD-style communication reporting. Default 5.')
    p.add_argument('--compute_is', type=int, default=1,
                   help='Whether to compute Inception Score during evaluation.')
    p.add_argument('--eval_real_split', type=str, default='train',
                   choices=['train', 'test'],
                   help='Which real split to export for FID/IS reference. FedPhD-style uses train.')
    p.add_argument('--data_range', type=str, default='minus1_1',
                   choices=['minus1_1', '0_1'],
                   help='Training/sample tensor range. Use 0_1 for legacy model_cifar.pth compatibility.')

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
