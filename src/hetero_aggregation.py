#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heterogeneous LoRA Aggregation Engine for Federated Diffusion Model Fine-tuning.

Core Algorithm:
  1. Recover true updates ΔW = B @ A from each client's local LoRA factors
  2. Compute rank-corrected aggregation weights α_{i,l}
  3. Aggregate in update space: ΔW̄ = Σ α_i · ΔW_i
  4. Re-project to low-rank via RSVD (Randomized SVD)
  5. Stabilize with Procrustes alignment against previous round factors
  6. Distribute nested subspace prefix slices to heterogeneous clients

References:
  [1] FLoRG: Federated Fine-tuning with Low-rank Gram Matrices and Procrustes Alignment
  [2] LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Step 1: Recover True Updates from LoRA Factors
# ============================================================

def recover_true_updates(
    client_factors_list: List[Tuple[int, List[Tuple[str, Dict[str, torch.Tensor], 
                                                    Dict[str, torch.Tensor]]]]],
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Tuple[int, int]]]:
    """
    For each client, compute ΔW_l = B_l @ A_l for every LoRA layer.
    
    Args:
        client_factors_list: List of (client_id, factors) where factors is
            the output of extract_all_lora_factors(): [(layer_name, B_dict, A_dict), ...]
    
    Returns:
        true_deltas: {layer_key: [ΔW_client1, ΔW_client2, ...]} — each ΔW is a 2D tensor
        layer_shapes: {layer_key: (out_dim, in_dim)} — reference shapes
    """
    true_deltas = {}
    layer_shapes = {}
    
    for client_id, factors in client_factors_list:
        for layer_name, B_dict, A_dict in factors:
            for factor_key in B_dict:
                full_key = f'{layer_name}_{factor_key}'
                B = B_dict[factor_key]  # [out, rank]
                A = A_dict[factor_key]  # [rank, in]
                
                # True update: ΔW = B @ A  (scaled by alpha/rank is already baked into gradients)
                delta = B @ A  # [out, in]
                
                if full_key not in true_deltas:
                    true_deltas[full_key] = []
                    layer_shapes[full_key] = (delta.shape[0], delta.shape[1])
                
                true_deltas[full_key].append((client_id, delta))
    
    return true_deltas, layer_shapes


def _extract_per_layer_ranks(
    client_factors_list: List[Tuple[int, List[Tuple[str, Dict[str, torch.Tensor],
                                                    Dict[str, torch.Tensor]]]]],
) -> Dict[str, Dict[int, int]]:
    """
    Extract per-layer per-client rank from client LoRA factors.

    For each layer_key (e.g., 'downs.0.2.fn_q'), determine the rank that
    each client used at that layer by inspecting B matrix shape [out, rank].

    Returns:
        {layer_key: {client_id: lora_rank}}
    """
    per_layer_ranks = {}
    for client_id, factors in client_factors_list:
        for layer_name, B_dict, A_dict in factors:
            for factor_key in B_dict:
                full_key = f'{layer_name}_{factor_key}'
                B = B_dict[factor_key]  # [out, rank]
                r = B.shape[1]
                if full_key not in per_layer_ranks:
                    per_layer_ranks[full_key] = {}
                per_layer_ranks[full_key][client_id] = r
    return per_layer_ranks


# ============================================================
# Step 2: Rank-Corrected AggregationWeights (Per-Layer α_{i,l})
# ============================================================

def compute_aggregation_weights(
    data_sizes: Dict[int, int],
    client_ranks: Dict[int, int],  # client_id -> rank used by this client (backward compat)
    rank_correction: bool = True,
    rank_beta: float = 0.5,
    per_layer_ranks: Optional[Dict[str, Dict[int, int]]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute **per-layer** aggregation weights with optional rank correction.

    Paper formula (Eq. for α_{i,l}^{(t)}):
        α_{i,l} = n_i · η(r_{i,l}) / Σ_{j∈S_t} n_j · η(r_{j,l})

    where η(r) = 1/√r prevents high-rank clients from dominating.

    When per_layer_ranks is provided (recommended), returns weights indexed
    by (layer, client) — strictly matching the paper's α_{i,l} notation.

    When per_layer_ranks is None (legacy mode), falls back to per-client
    uniform weights across all layers.

    Args:
        data_sizes: {client_id: num_samples}
        client_ranks: {client_id: lora_rank} (used only in legacy/fallback mode)
        rank_correction: Whether to apply η(r) = 1/√r correction
        rank_beta: Exponent in η(r) = r^(-rank_beta). Default 0.5 → 1/√r
        per_layer_ranks: {layer_key: {client_id: rank}} from _extract_per_layer_ranks()

    Returns:
        {layer_key: {client_id: alpha_i_l}}, each layer's weights sum to 1.0
    """
    # --- Paper-aligned: per-layer weights α_{i,l} ---
    if per_layer_ranks is not None:
        layer_weights = {}
        for layer_key, layer_client_ranks in per_layer_ranks.items():
            raw_weights = {}
            for cid in layer_client_ranks:
                n_i = data_sizes.get(cid, 1)
                r_il = layer_client_ranks[cid]

                if rank_correction and r_il > 0:
                    eta = float(r_il) ** (-rank_beta)
                else:
                    eta = 1.0

                raw_weights[cid] = n_i * eta

            # Normalize per layer: Σ_i α_{i,l} = 1
            total_w = sum(raw_weights.values())
            if total_w > 0:
                layer_weights[layer_key] = {
                    cid: w / total_w for cid, w in raw_weights.items()
                }
            else:
                layer_weights[layer_key] = {}

        return layer_weights

    # --- Legacy fallback: per-client uniform weights ---
    weights = {}
    for cid in data_sizes:
        n_i = data_sizes[cid]
        r_i = client_ranks.get(cid, 1)

        if rank_correction and r_i > 0:
            eta = float(r_i) ** (-rank_beta)
        else:
            eta = 1.0

        weights[cid] = n_i * eta

    total_weight = sum(weights.values())
    if total_weight > 0:
        for cid in weights:
            weights[cid] /= total_weight

    # Legacy format: wrap into per-layer dict for downstream compatibility.
    # All layers share the same per-client weight vector.
    legacy_wrapped = {}  # will be populated by caller if needed
    # Store flat weights for backward compat; caller should migrate to per-layer API
    return weights  # backward-compat return: Dict[int, float]


# ============================================================
# Step 3: Weighted Aggregation in Update Space (Per-Layer)
# ============================================================

def aggregate_true_updates(
    true_deltas: Dict[str, List[Tuple[int, torch.Tensor]]],
    agg_weights,  # Dict[str, Dict[int, float]] per-layer OR Dict[int, float] legacy
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted average of true updates in the dense update space.

    Paper formula:
        ΔW̄_l = Σ_{i∈S_t} α_{i,l} · ΔW_{i,l}

    When agg_weights is per-layer {layer_key: {cid: alpha}}, uses α_{i,l}
    strictly as defined in the paper. When it's a flat {cid: alpha}, falls
    back to uniform-per-layer weights (legacy behavior).

    Args:
        true_deltas: output from recover_true_updates()
                   {layer_key: [(client_id, delta_tensor), ...]}
        agg_weights: output from compute_aggregation_weights()
                    - New: {layer_key: {client_id: alpha_i_l}}
                    - Legacy: {client_id: alpha_i}

    Returns:
        aggregated: {layer_key: aggregated ΔW tensor}
    """
    aggregated = {}

    # Detect format
    _is_per_layer = (
        isinstance(agg_weights, dict) and
        bool(agg_weights) and
        isinstance(next(iter(agg_weights.values())), dict)
    )

    for layer_key, client_delta_list in true_deltas.items():
        if not client_delta_list:
            continue

        ref_delta = client_delta_list[0][1]
        agg_delta = torch.zeros_like(ref_delta)

        for client_id, delta in client_delta_list:
            if _is_per_layer:
                # Paper-aligned: α_{i,l} — weight specific to this layer
                layer_w = agg_weights.get(layer_key, {})
                w = layer_w.get(client_id, 0.0)
            else:
                # Legacy fallback: same α_i for all layers
                w = agg_weights.get(client_id, 0.0)

            agg_delta += w * delta

        aggregated[layer_key] = agg_delta

    return aggregated


# ============================================================
# Step 4: Low-Rank Reprojection via SVD / Randomized SVD
# ============================================================

def randomized_svd(matrix: torch.Tensor, rank: int, n_oversamples: int = 10,
                   n_power_iter: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficient approximate SVD for large matrices using randomized algorithm.
    
    Falls back to exact SVD for small matrices.
    
    Returns:
        U, S, Vt such that matrix ≈ U @ diag(S) @ Vt
        U: [m, rank], S: [rank], Vt: [rank, n]
    """
    m, n = matrix.shape
    device = matrix.device
    
    # For small matrices, use exact SVD
    if min(m, n) <= 256 or m * n <= 65536:
        U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
        return U[:, :rank], S[:rank], Vt[:rank, :]
    
    # Randomized SVD for larger matrices
    # Use PyTorch implementation (similar to sklearn.extmath.randomized_svd)
    target_rank = min(rank + n_oversamples, min(m, n))
    
    # Step 1: Random projection
    omega = torch.randn(n, target_rank, device=device, dtype=matrix.dtype)
    
    # Step 2: Form sample matrix Y = M Ω
    Y = matrix @ omega  # [m, target_rank]
    
    # Step 3: Power iteration for better accuracy
    for _ in range(n_power_iter):
        Y = matrix @ (matrix.T @ Y)
    
    # Step 4: QR decomposition
    Q, _ = torch.linalg.qr(Y)  # [m, target_rank]
    
    # Step 5: Project: B = Q^T M
    B = Q.T @ matrix  # [target_rank, n]
    
    # Step 6: Exact SVD of small matrix B
    U_small, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Step 7: U = Q U_small
    U = Q @ U_small  # [m, target_rank]
    
    # Truncate to desired rank
    k = min(rank, S.shape[0])
    return U[:, :k], S[:k], Vt[:k, :]  # FIX: U[:,:k] not U[:,k] (was taking single column)


def reproject_to_low_rank(
    aggregated_delta: torch.Tensor,
    global_rank: int,
    method: str = 'rsvd',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project aggregated dense update back to low-rank LoRA form.
    
    ΔW̄ ≈ U Σ V^T → B_new = U √Σ,  A_new = √Σ V^T
    
    The symmetric split ensures B and A contribute equally.
    
    Args:
        aggregated_delta: [out_channels, in_channels] dense update matrix
        global_rank: Target rank for the new global LoRA
        method: 'rsvd' (randomized SVD) or 'exact' (full SVD)
    
    Returns:
        B_global: [out_channels, global_rank]
        A_global: [global_rank, in_channels]
    """
    if method == 'rsvd':
        U, S, Vt = randomized_svd(aggregated_delta, global_rank)
    else:
        U, S, Vt = torch.linalg.svd(aggregated_delta, full_matrices=False)
        U = U[:, :global_rank]
        S = S[:global_rank]
        Vt = Vt[:global_rank, :]
    
    sqrt_S = torch.sqrt(S.clamp(min=0))  # Numerical safety
    
    B_global = U * sqrt_S.unsqueeze(0)   # [m, r] * [1, r] -> broadcast
    A_global = sqrt_S.unsqueeze(1) * Vt   # [r, 1] * [r, n] -> broadcast
    
    return B_global, A_global


def reproject_all_layers(
    aggregated_deltas: Dict[str, torch.Tensor],
    global_rank: int,
    method: str = 'rsvd',
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Reproject all aggregated layers back to (B, A) factor pairs.
    
    Returns:
        B_global_dict: {layer_name: {factor_key: B_tensor}}
        A_global_dict: {layer_name: {factor_key: A_tensor}}
    """
    B_global_all = {}  # {layer_name: {key: B}}
    A_global_all = {}  # {layer_name: {key: A}}
    
    # Group by layer name (split off factor_key suffix like _q, _k, etc.)
    layer_groups = OrderedDict()
    for full_key, delta in aggregated_deltas.items():
        # full_key is like "downs.0.2.fn_q" or "mid_attn_out"
        # Find the last underscore that separates layer from factor type
        parts = full_key.rsplit('_', 1)
        if len(parts) == 2:
            layer_name, factor_key = parts
        else:
            layer_name, factor_key = full_key, ''
        
        if layer_name not in layer_groups:
            layer_groups[layer_name] = {}
        layer_groups[layer_name][factor_key] = delta
    
    for layer_name, factor_deltas in layer_groups.items():
        B_global_all[layer_name] = {}
        A_global_all[layer_name] = {}
        
        for factor_key, delta in factor_deltas.items():
            B_g, A_g = reproject_to_low_rank(delta, global_rank, method)
            B_global_all[layer_name][factor_key] = B_g
            A_global_all[layer_name][factor_key] = A_g
    
    return B_global_all, A_global_all


# ============================================================
# Step 5: Orthogonal Procrustes Alignment
# ============================================================

def procrustes_alignment(
    B_new: Dict[str, torch.Tensor],
    A_new: Dict[str, torch.Tensor],
    B_prev: Optional[Dict[str, torch.Tensor]],
    A_prev: Optional[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Align current LoRA factors with previous round's factors using
    orthogonal Procrustes analysis to mitigate representation drift.
    
    Solves: Q* = argmin ||B_new Q - B_prev||²_F + ||Q^T A_new - A_prev||²_F
           s.t. Q^T Q = I
    
    When no previous factors exist (first round), returns unchanged.
    
    Args:
        B_new, A_new: Current round global factors (after RSVD reprojection)
        B_prev, A_prev: Previous round global factors (can be None on round 0)
    
    Returns:
        B_aligned, A_aligned: Aligned factors (product B@A is unchanged)
    """
    if B_prev is None or A_prev is None:
        return B_new, A_new
    
    # Stack all B matrices and all A matrices for joint alignment
    B_stack = []
    A_stack = []
    B_prev_stack = []
    A_prev_stack = []
    
    for key in B_new:
        if key in B_prev and key in A_prev and key in A_new:
            B_stack.append(B_new[key].flatten())
            A_stack.append(A_new[key].flatten())
            B_prev_stack.append(B_prev[key].flatten())
            A_prev_stack.append(A_prev[key].flatten())
    
    if not B_stack:
        return B_new, A_new
    
    B_mat = torch.stack(B_stack)       # [L, D_B]
    A_mat = torch.stack(A_stack)       # [L, D_A]
    Bp_mat = torch.stack(B_prev_stack) # [L, D_B]
    Ap_mat = torch.stack(A_prev_stack) # [L, D_A]
    
    # Joint Procrustes: combine all factor pairs into one optimization.
    # Since each (B,A) pair has shape ([out,r], [r,in]) with shared rank r,
    # we construct M = sum over all factors of (Bprev' @ Bnew + Anew @ Aprev')
    # Each term is [r, r], so they're directly summable.
    
    device = B_mat.device
    
    # Compute M as sum of per-factor [r, r] matrices from BOTH B and A terms
    M_joint = None
    valid_keys = []
    
    for key in B_new:
        if key in B_prev and key in A_prev and key in A_new:
            b_n = B_new[key]       # [out, r]
            b_p = B_prev[key]      # [out, r]
            a_n = A_new[key]       # [r,  in]
            a_p = A_prev[key]      # [r,  in]
            
            if b_n.shape[1] == b_p.shape[1]:  # same rank
                r = b_n.shape[1]
                m_k = b_n.t() @ b_p + a_n @ a_p.t()  # [r, r]
                if M_joint is None:
                    M_joint = torch.zeros(r, r, device=device)
                M_joint += m_k
                valid_keys.append(key)
    
    if M_joint is None or M_joint.abs().sum() == 0:
        return B_new, A_new
    
    # SVD to find optimal rotation Q
    try:
        U, _, Vt = torch.linalg.svd(M_joint, full_matrices=False)
        Q = U @ Vt  # [r, r] orthogonal
    except Exception:
        r = B_new[list(B_new.keys())[0]].shape[1]
        Q = torch.eye(r, device=device)
    
    # Apply alignment to each factor pair
    B_aligned = {}
    A_aligned = {}
    
    for key in B_new:
        if key in valid_keys:
            r = B_new[key].shape[1]
            if Q.shape[0] >= r:
                Q_k = Q[:r, :r].to(B_new[key].device)
                B_aligned[key] = B_new[key] @ Q_k
                A_aligned[key] = Q_k.t().to(A_new[key].device) @ A_new[key]
            else:
                B_aligned[key] = B_new[key].clone()
                A_aligned[key] = A_new[key].clone()
        else:
            B_aligned[key] = B_new[key].clone()
            A_aligned[key] = A_new[key].clone()
    
    return B_aligned, A_aligned


def procrustes_alignment_per_layer(
    B_new: Dict[str, Dict[str, torch.Tensor]],   # {layer: {factor: B_tensor [out, r]}}
    A_new: Dict[str, Dict[str, torch.Tensor]],   # {layer: {factor: A_tensor [r, in]}}
    B_prev: Optional[Dict[str, Dict[str, torch.Tensor]]],
    A_prev: Optional[Dict[str, Dict[str, torch.Tensor]]],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Per-factor Procrustes alignment.
    
    For each (B, A) pair within each layer, solve:
        min_Q ||B_new Q - B_prev||_F^2   s.t.  Q^T Q = I
    
    Closed-form via SVD of M = B_prev^T @ B_new:
        U, S, Vt = SVD(M)  =>  Q = U @ Vt
    
    Then:  B_aligned = B_new @ Q
           A_aligned = Q^T @ A_new
    Product preserved: B_aligned @ A_aligned = B_new @ A_new
    """
    if B_prev is None or A_prev is None:
        return B_new, A_new

    B_aligned = {}
    A_aligned = {}

    for layer_name in B_new:
        if layer_name not in B_prev:
            B_aligned[layer_name] = dict(B_new[layer_name])
            A_aligned[layer_name] = dict(A_new[layer_name])
            continue

        Bl = B_new[layer_name]
        Al = A_new[layer_name]
        Bl_prev = B_prev[layer_name]

        keys = set(Bl.keys())
        if not keys:
            B_aligned[layer_name] = dict(Bl)
            A_aligned[layer_name] = dict(Al)
            continue

        B_aligned[layer_name] = {}
        A_aligned[layer_name] = {}

        for k in keys:
            b = Bl[k]          # [out_dim, r]
            a = Al[k]          # [r, in_dim]

            if k not in Bl_prev:
                B_aligned[layer_name][k] = b.clone()
                A_aligned[layer_name][k] = a.clone()
                continue

            b_prev = Bl_prev[k] # [out_dim, r_prev]

            # Only align when rank matches between rounds
            if b.shape[1] != b_prev.shape[1]:
                B_aligned[layer_name][k] = b.clone()
                A_aligned[layer_name][k] = a.clone()
                continue

            device = b.device
            r = b.shape[1]

            try:
                # Full objective from paper: min ||BQ - Bprev||^2 + ||Q'A - Aprev||^2
                # After derivation (using Q'Q=I), equivalent to:
                #   max_Q tr(Q' (Bnew.T @ Bprev + Anew @ Aprev.T))  =>  Q = U V'T from SVD(M)
                #
                # Both terms are [r, r]:
                #   Bnew.T @ Bprev:  [r, out] @ [out, r] = [r, r]
                #   Anew     @ Aprev': [r, in ] @ [in,  r] = [r, r]
                if layer_name in A_prev and k in A_prev[layer_name]:
                    a_prev = A_prev[layer_name][k]
                    M = b.t() @ b_prev + a @ a_prev.t()  # [r, r]
                else:
                    M = b.t() @ b_prev  # fallback: B-only
                U, _, Vt = torch.linalg.svd(M, full_matrices=False)
                Q = U @ Vt  # [r, r], orthogonal

                # Apply same rotation to both factors
                B_aligned[layer_name][k] = (b @ Q).contiguous()
                A_aligned[layer_name][k] = (Q.t() @ a).contiguous()
            except Exception as e:
                print(f'  [Procrustes Warning] Factor {layer_name}/{k} failed: {e}')
                B_aligned[layer_name][k] = b.clone()
                A_aligned[layer_name][k] = a.clone()

    return B_aligned, A_aligned


# ============================================================
# Step 6: Nested Subspace Prefix Distribution
# ============================================================

def prefix_slice_distribution(
    B_global: Dict[str, Dict[str, torch.Tensor]],
    A_global: Dict[str, Dict[str, torch.Tensor]],
    client_ranks: Dict[int, Dict[str, int]],
) -> Dict[int, Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]]:
    """
    Distribute global LoRA factors to heterogeneous clients using prefix slicing.
    
    For client i with local rank r_{i,l} at layer l:
        B_i = B_global[:, :r_{i,l}]  (take first r columns)
        A_i = A_global[:r_{i,l}, :]  (take first r rows)
    
    This ensures all clients receive prefixes of the same global principal subspace,
    mitigating subspace mismatch in Non-IID settings.
    
    Args:
        B_global: {layer_name: {factor_key: B_tensor [out, R_global]}}
        A_global: {layer_name: {factor_key: A_tensor [R_global, in]}}
        client_ranks: {client_id: {layer_name: rank}} or {client_id: int} for uniform rank
    
    Returns:
        client_factors: {
            client_id: (
                B_client: {layer: {factor: B_sliced}},
                A_client: {layer: {factor: A_sliced}}
            )
        }
    """
    client_factors = {}
    
    for client_id, rank_config in client_ranks.items():
        B_client = {}
        A_client = {}
        
        # Support both uniform rank (int) and per-layer rank (dict)
        if isinstance(rank_config, int):
            uniform_rank = rank_config
            rank_config = {layer: uniform_rank for layer in B_global}
        
        for layer_name in B_global:
            client_rank = rank_config.get(layer_name, 4)  # default fallback
            
            B_client[layer_name] = {}
            A_client[layer_name] = {}
            
            for factor_key in B_global[layer_name]:
                B_full = B_global[layer_name][factor_key]  # [out, R_global]
                A_full = A_global[layer_name][factor_key]  # [R_global, in]
                
                R_global = B_full.shape[1]
                r = min(client_rank, R_global)
                
                if r > 0:
                    B_client[layer_name][factor_key] = B_full[:, :r].clone()
                    A_client[layer_name][factor_key] = A_full[:r, :].clone()
                else:
                    B_client[layer_name][factor_key] = B_full.clone()
                    A_client[layer_name][factor_key] = A_full.clone()
        
        client_factors[client_id] = (B_client, A_client)
    
    return client_factors


# ============================================================
# Complete Aggregation Pipeline (End-to-End)
# ============================================================

def count_factor_list_bytes(
    factors: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    bytes_per_param: int = 4,
) -> int:
    total_elems = 0
    for _, B_dict, A_dict in factors:
        total_elems += sum(t.numel() for t in B_dict.values())
        total_elems += sum(t.numel() for t in A_dict.values())
    return total_elems * bytes_per_param


def count_client_distribution_bytes(
    client_updates: Dict[int, Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]],
    bytes_per_param: int = 4,
) -> int:
    total_elems = 0
    for B_client, A_client in client_updates.values():
        for layer_name, B_dict in B_client.items():
            A_dict = A_client.get(layer_name, {})
            total_elems += sum(t.numel() for t in B_dict.values())
            total_elems += sum(t.numel() for t in A_dict.values())
    return total_elems * bytes_per_param

class HeteroLoRAAggregator:
    """
    Full heterogeneous LoRA aggregation pipeline combining all steps.
    
    Usage:
        aggregator = HeteroLoRAAggregator(global_rank=16)
        
        # Each round:
        result = aggregator.aggregate_round(
            client_factors=[(cid, factors) for each client],
            data_sizes={cid: n_samples},
            client_ranks={cid: rank},
        )
        
        # Distribute to clients:
        for cid, (B, A) in result['client_updates'].items():
            set_all_lora_factors(model, [(name, B[name], A[name]) for name in B])
    """

    def __init__(self, global_rank: int, rank_correction: bool = True,
                 use_procrustes: bool = True, svd_method: str = 'rsvd',
                 rank_beta: float = 0.5):
        self.global_rank = global_rank
        self.rank_correction = rank_correction
        self.use_procrustes = use_procrustes
        self.svd_method = svd_method
        self.rank_beta = rank_beta
        
        # Store previous round factors for Procrustes alignment
        self.B_prev = None
        self.A_prev = None
        self.round_num = 0

    def aggregate_round(
        self,
        client_factors_list: List[Tuple[int, List[Tuple[str, Dict, Dict]]]],
        data_sizes: Dict[int, int],
        client_ranks: Dict[int, int],
    ) -> dict:
        """
        Execute one complete aggregation round.
        
        Args:
            client_factors_list: [(client_id, extracted_factors), ...]
            data_sizes: {client_id: num_local_samples}
            client_ranks: {client_id: lora_rank_used_by_client}
        
        Returns:
            dict with keys:
                'global_B': {layer: {factor: B_tensor}} — aligned global B factors
                'global_A': {layer: {factor: A_tensor}} — aligned global A factors  
                'client_updates': {cid: (B_dict, A_dict)} — per-client prefix distributions
                'stats': aggregation statistics
        """
        stats = {'round': self.round_num, 'num_clients': len(client_factors_list)}
        
        # --- Step 1: Recover true updates ---
        true_deltas, layer_shapes = recover_true_updates(client_factors_list)
        stats['num_layers'] = len(true_deltas)
        print(f'[HeteroAgg Round {self.round_num}] '
              f'Recovered true updates from {len(client_factors_list)} clients, '
              f'{len(true_deltas)} LoRA layers')
        
        # --- Step 2: Compute per-layer aggregation weights α_{i,l} ---
        # Extract per-layer per-client ranks from LoRA factor shapes
        per_layer_ranks = _extract_per_layer_ranks(client_factors_list)
        agg_weights = compute_aggregation_weights(
            data_sizes, client_ranks, self.rank_correction, self.rank_beta,
            per_layer_ranks=per_layer_ranks,  # paper-aligned: α_{i,l}
        )
        
        # --- Step 3: Aggregate in update space (per-layer weighted) ---
        aggregated = aggregate_true_updates(true_deltas, agg_weights)
        stats['aggregated_layers'] = len(aggregated)
        
        # --- Step 4: RSVD reprojection to low-rank ---
        B_projected, A_projected = reproject_all_layers(aggregated, self.global_rank, self.svd_method)
        print(f'[HeteroAgg Round {self.round_num}] '
              f'RSVD reprojection to global_rank={self.global_rank}')
        
        # --- Step 5: Procrustes alignment ---
        if self.use_procrustes and self.B_prev is not None:
            B_aligned, A_aligned = procrustes_alignment_per_layer(
                B_projected, A_projected, self.B_prev, self.A_prev
            )
            print(f'[HeteroAgg Round {self.round_num}] Procrustes alignment applied')
        else:
            B_aligned, A_aligned = B_projected, A_projected
            if self.round_num == 0:
                print(f'[HeteroAgg Round {self.round_num}] Skipping Procrustes (round 0)')
        
        # Store for next round
        self.B_prev = B_aligned
        self.A_prev = A_aligned
        
        # --- Step 6: Prefix distribution to heterogeneous clients ---
        client_updates = prefix_slice_distribution(B_aligned, A_aligned, client_ranks)
        
        # Stats
        total_up_bytes = sum(
            count_factor_list_bytes(factors)
            for _, factors in client_factors_list
        ) if client_factors_list else 0

        total_down_bytes = count_client_distribution_bytes(client_updates) if client_updates else 0
        
        stats['upload_bytes'] = total_up_bytes
        stats['download_bytes'] = total_down_bytes
        stats['rank_beta'] = float(self.rank_beta)
        stats['rank_correction'] = bool(self.rank_correction)
        # Store per-layer weights for inspection (paper α_{i,l} format)
        if isinstance(agg_weights, dict) and agg_weights:
            _first_val = next(iter(agg_weights.values()))
            if isinstance(_first_val, dict):
                # Per-layer format: {layer_key: {cid: alpha}}
                stats['agg_weights_per_layer'] = {
                    lk: {str(cid): float(w) for cid, w in cw.items()}
                    for lk, cw in agg_weights.items()
                }
            else:
                # Legacy flat format
                stats['agg_weights'] = {str(cid): float(w) for cid, w in agg_weights.items()}
        else:
            stats['agg_weights'] = {}
        
        self.round_num += 1
        
        return {
            'global_B': B_aligned,
            'global_A': A_aligned,
            'global_raw_B': B_projected,
            'global_raw_A': A_projected,
            'client_updates': client_updates,
            'aggregated_deltas': aggregated,
            'stats': stats,
        }

    def reset(self):
        """Reset internal state (e.g., for a new experiment)."""
        self.B_prev = None
        self.A_prev = None
        self.round_num = 0
