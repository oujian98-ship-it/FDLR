#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) for U-Net Diffusion Models.

Supports:
  - Conv2d-based LoRA for 1x1 convolutions (Attention projections)
  - Automatic injection into Unet/UnetConditional Attention blocks
  - Splitting joint to_qkv into separate Q/K/V projections with individual LoRA
  - Heterogeneous rank configuration per-client
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Core LoRA Layer for Conv2d (1x1 conv == linear transform)
# ============================================================

class LoRAConv2d(nn.Module):
    """
    LoRA wrapper around a frozen nn.Conv2d (kernel_size=1).
    
    For a weight matrix W of shape [out_channels, in_channels]:
        output = W @ x + B @ A @ x
    where B: [out_channels, rank], A: [rank, in_channels]
    
    The original convolution weights are frozen.
    """

    def __init__(self, conv_layer: nn.Conv2d, rank: int, alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        assert conv_layer.kernel_size == (1, 1) or \
               (isinstance(conv_layer.kernel_size, int) and conv_layer.kernel_size == 1), \
            f"LoRAConv2d only supports 1x1 convolutions, got kernel_size={conv_layer.kernel_size}"

        self.conv = conv_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0

        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels

        # Freeze original weights
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        # Low-rank decomposition matrices
        # For Conv2d(1,1), weight shape is [out, in, 1, 1] -> treat as [out, in]
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))

        # Initialize A with Kaiming init (small random), B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        result = self.conv(x)
        
        # LoRA path: Conv2d(1x1) is equivalent to linear transform
        x_lora = self.dropout(x)
        bsz, c_in, height, width = x_lora.shape
        x_flat = x_lora.permute(0, 2, 3, 1).reshape(bsz * height * width, c_in)

        lora_mid = F.linear(x_flat, self.lora_A, None)       # [BHW, r]
        lora_out = F.linear(lora_mid, self.lora_B, None)     # [BHW, cout]
        lora_out = lora_out.reshape(bsz, height, width, self.out_channels).permute(0, 3, 1, 2)

        return result + self.scaling * lora_out

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def in_channels(self):
        return self.conv.in_channels


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear (used for time_mlp etc.)"""

    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        result = self.linear(x)
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A, None)
        lora_out = F.linear(lora_out, self.lora_B, None)
        return result + self.scaling * lora_out

    @property
    def out_features(self):
        return self.linear.out_features

    @property
    def in_features(self):
        return self.linear.in_features


# ============================================================
# Helper: Safe Factor Setter
# ============================================================

def _set_lora_pair_(module: LoRAConv2d, B: torch.Tensor, A: torch.Tensor) -> None:
    """Safely set one LoRA pair on a LoRAConv2d module and sync rank/scaling."""
    if not isinstance(module, LoRAConv2d):
        raise TypeError(f"Expected LoRAConv2d, got {type(module)}")
    if B.ndim != 2 or A.ndim != 2:
        raise ValueError(f"LoRA factors must be 2D, got B={tuple(B.shape)}, A={tuple(A.shape)}")
    if B.shape[1] != A.shape[0]:
        raise ValueError(f"Rank mismatch: B={tuple(B.shape)}, A={tuple(A.shape)}")
    if B.shape[0] != module.out_channels:
        raise ValueError(f"B out dim mismatch: expected {module.out_channels}, got {B.shape[0]}")
    if A.shape[1] != module.in_channels:
        raise ValueError(f"A in dim mismatch: expected {module.in_channels}, got {A.shape[1]}")

    device = next(module.parameters()).device
    dtype = module.lora_A.dtype
    B = B.detach().to(device=device, dtype=dtype).clone()
    A = A.detach().to(device=device, dtype=dtype).clone()
    rank = int(B.shape[1])

    if tuple(module.lora_B.shape) != tuple(B.shape):
        module.lora_B = nn.Parameter(torch.empty_like(B), requires_grad=True)
    if tuple(module.lora_A.shape) != tuple(A.shape):
        module.lora_A = nn.Parameter(torch.empty_like(A), requires_grad=True)

    module.lora_B.data.copy_(B)
    module.lora_A.data.copy_(A)

    module.rank = rank
    module.scaling = module.alpha / rank if rank > 0 else 0.0


# ============================================================
# Injected Attention Module (replaces to_qkv with separate Q/K/V + LoRA)
# ============================================================

class LoRAInjectedAttention(nn.Module):
    """
    Replaces the original Attention module by splitting to_qkv into
    separate Q/K/V Conv2d layers, each wrapped with optional LoRA.
    
    Also wraps to_out with LoRA.
    """

    def __init__(self, original_attn: 'Attention', rank_qkv: int, rank_out: int,
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()

        self.heads = original_attn.heads
        self.scale = original_attn.scale

        dim = original_attn.to_qkv.in_channels
        hidden_dim = original_attn.to_qkv.out_channels // 3

        # Copy frozen original weights and split them
        qkv_weight = original_attn.to_qkv.weight.data  # [hidden*3, dim, 1, 1]
        q_weight = qkv_weight[:hidden_dim].reshape(hidden_dim, dim, 1, 1).contiguous()
        k_weight = qkv_weight[hidden_dim:hidden_dim*2].reshape(hidden_dim, dim, 1, 1).contiguous()
        v_weight = qkv_weight[hidden_dim*2:].reshape(hidden_dim, dim, 1, 1).contiguous()

        # Create separate Q, K, V convolutions (frozen)
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)

        self.to_q.weight.data = q_weight.clone()
        self.to_k.weight.data = k_weight.clone()
        self.to_v.weight.data = v_weight.clone()

        # Wrap each with LoRA
        if rank_qkv > 0:
            self.lora_q = LoRAConv2d(self.to_q, rank=rank_qkv, alpha=alpha, dropout=dropout)
            self.lora_k = LoRAConv2d(self.to_k, rank=rank_qkv, alpha=alpha, dropout=dropout)
            self.lora_v = LoRAConv2d(self.to_v, rank=rank_qkv, alpha=alpha, dropout=dropout)
        else:
            self.lora_q = self.to_q
            self.lora_k = self.to_k
            self.lora_v = self.to_v

        # Handle to_out (may be plain Conv2d or Sequential with GroupNorm)
        orig_to_out = original_attn.to_out
        if isinstance(orig_to_out, nn.Sequential):
            # LinearAttention has: Sequential(Conv2d, GroupNorm)
            conv_part = orig_to_out[0]
            norm_part = orig_to_out[1]
            if rank_out > 0:
                self.lora_out_conv = LoRAConv2d(conv_part, rank=rank_out, alpha=alpha, dropout=dropout)
                self.to_out = nn.Sequential(self.lora_out_conv, norm_part)
            else:
                self.to_out = orig_to_out
        else:
            # Plain Attention has just Conv2d
            if rank_out > 0:
                self.lora_out_conv = LoRAConv2d(orig_to_out, rank=rank_out, alpha=alpha, dropout=dropout)
                self.to_out = self.lora_out_conv
            else:
                self.to_out = orig_to_out

        # Store config for later retrieval
        self._rank_qkv = rank_qkv
        self._rank_out = rank_out

    def forward(self, x):
        from einops import rearrange
        from torch import einsum

        bsz, channels, height, width = x.shape

        q = self.lora_q(x)
        k = self.lora_k(x)
        v = self.lora_v(x)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (heads dim) height width -> b heads dim (height width)",
                heads=self.heads,
            ),
            (q, k, v),
        )

        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Output is [b, heads, dim, tokens]
        out = einsum("b h i j, b h d j -> b h d i", attn, v)

        out = rearrange(
            out,
            "b heads dim (height width) -> b (heads dim) height width",
            height=height,
            width=width,
        )
        return self.to_out(out)

    def get_lora_factors(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract all LoRA (B, A) factor pairs from this attention module.
        Returns:
            B_dict: {name: B_matrix}  — left factors, shape [out, rank]
            A_dict: {name: A_matrix}  — right factors, shape [rank, in]
        """
        B_dict, A_dict = {}, {}

        if hasattr(self, 'lora_q') and isinstance(self.lora_q, LoRAConv2d):
            B_dict['q'] = self.lora_q.lora_B.detach().cpu().clone()
            A_dict['q'] = self.lora_q.lora_A.detach().cpu().clone()
        if hasattr(self, 'lora_k') and isinstance(self.lora_k, LoRAConv2d):
            B_dict['k'] = self.lora_k.lora_B.detach().cpu().clone()
            A_dict['k'] = self.lora_k.lora_A.detach().cpu().clone()
        if hasattr(self, 'lora_v') and isinstance(self.lora_v, LoRAConv2d):
            B_dict['v'] = self.lora_v.lora_B.detach().cpu().clone()
            A_dict['v'] = self.lora_v.lora_A.detach().cpu().clone()
        if hasattr(self, 'lora_out_conv') and isinstance(self.lora_out_conv, LoRAConv2d):
            B_dict['out'] = self.lora_out_conv.lora_B.detach().cpu().clone()
            A_dict['out'] = self.lora_out_conv.lora_A.detach().cpu().clone()

        return B_dict, A_dict

    def set_lora_factors(self, B_dict: Dict[str, torch.Tensor],
                         A_dict: Dict[str, torch.Tensor]):
        """Set LoRA factors from server distribution."""
        if 'q' in B_dict and hasattr(self, 'lora_q') and isinstance(self.lora_q, LoRAConv2d):
            _set_lora_pair_(self.lora_q, B_dict['q'], A_dict['q'])
        if 'k' in B_dict and hasattr(self, 'lora_k') and isinstance(self.lora_k, LoRAConv2d):
            _set_lora_pair_(self.lora_k, B_dict['k'], A_dict['k'])
        if 'v' in B_dict and hasattr(self, 'lora_v') and isinstance(self.lora_v, LoRAConv2d):
            _set_lora_pair_(self.lora_v, B_dict['v'], A_dict['v'])
        if 'out' in B_dict and hasattr(self, 'lora_out_conv') and isinstance(self.lora_out_conv, LoRAConv2d):
            _set_lora_pair_(self.lora_out_conv, B_dict['out'], A_dict['out'])


# Same treatment for LinearAttention
class LoRAInjectedLinearAttention(nn.Module):

    def __init__(self, original_attn: 'LinearAttention', rank_qkv: int, rank_out: int,
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()

        self.heads = original_attn.heads
        self.scale = original_attn.scale

        dim = original_attn.to_qkv.in_channels
        hidden_dim = original_attn.to_qkv.out_channels // 3

        qkv_weight = original_attn.to_qkv.weight.data
        q_weight = qkv_weight[:hidden_dim].reshape(hidden_dim, dim, 1, 1).contiguous()
        k_weight = qkv_weight[hidden_dim:hidden_dim*2].reshape(hidden_dim, dim, 1, 1).contiguous()
        v_weight = qkv_weight[hidden_dim*2:].reshape(hidden_dim, dim, 1, 1).contiguous()

        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)

        self.to_q.weight.data = q_weight.clone()
        self.to_k.weight.data = k_weight.clone()
        self.to_v.weight.data = v_weight.clone()

        if rank_qkv > 0:
            self.lora_q = LoRAConv2d(self.to_q, rank=rank_qkv, alpha=alpha, dropout=dropout)
            self.lora_k = LoRAConv2d(self.to_k, rank=rank_qkv, alpha=alpha, dropout=dropout)
            self.lora_v = LoRAConv2d(self.to_v, rank=rank_qkv, alpha=alpha, dropout=dropout)
        else:
            self.lora_q = self.to_q
            self.lora_k = self.to_k
            self.lora_v = self.to_v

        orig_to_out = original_attn.to_out
        if isinstance(orig_to_out, nn.Sequential):
            conv_part = orig_to_out[0]
            norm_part = orig_to_out[1]
            if rank_out > 0:
                self.lora_out_conv = LoRAConv2d(conv_part, rank=rank_out, alpha=alpha, dropout=dropout)
                self.to_out = nn.Sequential(self.lora_out_conv, norm_part)
            else:
                self.to_out = orig_to_out
        else:
            if rank_out > 0:
                self.lora_out_conv = LoRAConv2d(orig_to_out, rank=rank_out, alpha=alpha, dropout=dropout)
                self.to_out = self.lora_out_conv
            else:
                self.to_out = orig_to_out

        self._rank_qkv = rank_qkv
        self._rank_out = rank_out

    def forward(self, x):
        from einops import rearrange
        from torch import einsum

        b, c, h, w = x.shape
        q = self.lora_q(x)
        k = self.lora_k(x)
        v = self.lora_v(x)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            (q, k, v)
        )

        q = q.softmax(-2)
        k = k.softmax(-1)
        q = q * self.scale

        context = einsum("b h d n, b h e n -> b h d e", k, v)
        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

    def get_lora_factors(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        B_dict, A_dict = {}, {}
        if hasattr(self, 'lora_q') and isinstance(self.lora_q, LoRAConv2d):
            B_dict['q'] = self.lora_q.lora_B.detach().cpu().clone()
            A_dict['q'] = self.lora_q.lora_A.detach().cpu().clone()
        if hasattr(self, 'lora_k') and isinstance(self.lora_k, LoRAConv2d):
            B_dict['k'] = self.lora_k.lora_B.detach().cpu().clone()
            A_dict['k'] = self.lora_k.lora_A.detach().cpu().clone()
        if hasattr(self, 'lora_v') and isinstance(self.lora_v, LoRAConv2d):
            B_dict['v'] = self.lora_v.lora_B.detach().cpu().clone()
            A_dict['v'] = self.lora_v.lora_A.detach().cpu().clone()
        if hasattr(self, 'lora_out_conv') and isinstance(self.lora_out_conv, LoRAConv2d):
            B_dict['out'] = self.lora_out_conv.lora_B.detach().cpu().clone()
            A_dict['out'] = self.lora_out_conv.lora_A.detach().cpu().clone()
        return B_dict, A_dict

    def set_lora_factors(self, B_dict: Dict[str, torch.Tensor],
                         A_dict: Dict[str, torch.Tensor]):
        for key in ['q', 'k', 'v', 'out']:
            attr_name = f'lora_{key}' if key != 'out' else 'lora_out_conv'
            if key in B_dict and hasattr(self, attr_name):
                lora_module = getattr(self, attr_name)
                if isinstance(lora_module, LoRAConv2d):
                    _set_lora_pair_(lora_module, B_dict[key], A_dict[key])


# ============================================================
# Injection Engine: automatically injects LoRA into U-Net
# ============================================================

def inject_lora_into_unet(
    model: nn.Module,
    rank: int,
    rank_linear: int = 0,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_layers: str = 'attention',
) -> nn.Module:
    """
    Inject LoRA layers into a U-Net or UnetConditional model.

    Args:
        model: The base U-Net model (weights will be frozen after injection)
        rank: LoRA rank for Attention projection layers (Q/K/V/out)
        rank_linear: LoRA rank for time_mlp linear layers (default 0 = skip)
        alpha: LoRA scaling factor (typically 1.0 or 2*rank)
        dropout: Dropout probability on LoRA path
        target_layers: Which layers to inject:
            - 'attention': only Attention and LinearAttention (recommended)
            - 'all': also inject into time_mlp linear layers
    
    Returns:
        Modified model with LoRA injected (original model is modified in-place)
    """
    model_device = next(model.parameters()).device
    lora_config = {}  # Track injected modules for later factor extraction

    # --- Inject into downsampling blocks ---
    for block_idx, block_list in enumerate(model.downs):
        # Each down block: [conv_block, conv_block, attn, downsample]
        _inject_into_block_list(
            model, f'downs.{block_idx}', block_list,
            block_idx, rank, alpha, dropout, target_layers, lora_config, 'down'
        )

    # --- Inject into mid block ---
    if hasattr(model, 'mid_attn'):
        orig_mid_attn = model.mid_attn.fn
        if isinstance(orig_mid_attn, type(None)):
            pass  # already handled via Residual(PreNorm(...))
        elif hasattr(orig_mid_attn, 'to_qkv'):  # Direct Attention
            new_attn = LoRAInjectedAttention(orig_mid_attn, rank, rank, alpha, dropout)
            model.mid_attn.fn = new_attn
            lora_config[f'mid_attn'] = {'type': 'attention', 'rank_qkv': rank, 'rank_out': rank}
        elif hasattr(orig_mid_attn.fn, 'to_qkv'):  # Wrapped in PreNorm -> Residual -> PreNorm -> Attention
            inner = orig_mid_attn.fn
            if hasattr(inner, 'fn') and hasattr(inner.fn, 'to_qkv'):
                attn_inner = inner.fn
                from unet import Attention
                if isinstance(attn_inner, Attention):
                    new_attn = LoRAInjectedAttention(attn_inner, rank, rank, alpha, dropout)
                    inner.fn = new_attn
                    lora_config[f'mid_attn'] = {'type': 'attention', 'rank_qkv': rank, 'rank_out': rank}
                else:
                    from unet import LinearAttention
                    if isinstance(attn_inner, LinearAttention):
                        new_attn = LoRAInjectedLinearAttention(attn_inner, rank, rank, alpha, dropout)
                        inner.fn = new_attn
                        lora_config[f'mid_attn'] = {'type': 'linear_attention', 'rank_qkv': rank, 'rank_out': rank}

    # --- Inject into upsampling blocks ---
    for block_idx, block_list in enumerate(model.ups):
        _inject_into_block_list(
            model, f'ups.{block_idx}', block_list,
            block_idx, rank, alpha, dropout, target_layers, lora_config, 'up'
        )

    # --- Optionally inject into time_mlp ---
    if target_layers == 'all' and rank_linear > 0 and hasattr(model, 'time_mlp'):
        for idx, layer in enumerate(model.time_mlp):
            if isinstance(layer, nn.Linear):
                new_linear = LoRALinear(layer, rank=rank_linear, alpha=alpha, dropout=dropout)
                model.time_mlp[idx] = new_linear
                lora_config[f'time_mlp.{idx}'] = {'type': 'linear', 'rank': rank_linear}

    # --- Freeze ALL non-LoRA parameters ---
    freeze_non_lora_params(model)

    # Store config for later use
    model._lora_config = lora_config
    model._lora_rank = rank
    model._lora_alpha = alpha

    print(f'[LoRA] Injected LoRA (rank={rank}, alpha={alpha}) into {len(lora_config)} modules')
    count_lora_params(model)
    
    return model


def _inject_into_block_list(model, parent_name, block_list, block_idx,
                             rank, alpha, dropout, target_layers, lora_config, direction):
    """Inject LoRA into one down/up block list."""
    from unet import Attention, LinearAttention

    for sub_idx, sub_module in enumerate(block_list):
        full_path = f'{parent_name}.{sub_idx}'
        
        # Check if this module contains an Attention-like layer
        # The structure is typically: block_klass, block_klass, Residual(PreNorm(Attn)), Down/Upsample
        target = _find_attention_in_module(sub_module)
        
        if target is not None:
            layer_key = f'{direction}_{block_idx}_{sub_idx}'
            if isinstance(target, Attention):
                new_attn = LoRAInjectedAttention(target, rank, rank, alpha, dropout)
                _replace_attention_in_module(sub_module, new_attn)
                lora_config[layer_key] = {
                    'type': 'attention', 'path': full_path,
                    'rank_qkv': rank, 'rank_out': rank
                }
            elif isinstance(target, LinearAttention):
                new_attn = LoRAInjectedLinearAttention(target, rank, rank, alpha, dropout)
                _replace_attention_in_module(sub_module, new_attn)
                lora_config[layer_key] = {
                    'type': 'linear_attention', 'path': full_path,
                    'rank_qkv': rank, 'rank_out': rank
                }


def _find_attention_in_module(module):
    """Recursively find the innermost Attention/LinearAttention inside a module wrapper."""
    from unet import Attention, LinearAttention
    
    if isinstance(module, (Attention, LinearAttention)):
        return module
    
    if isinstance(module, nn.Sequential):
        for child in module.children():
            found = _find_attention_in_module(child)
            if found is not None:
                return found
    
    if hasattr(module, 'fn'):  # Residual or PreNorm
        return _find_attention_in_module(module.fn)
    
    for child in module.children():
        found = _find_attention_in_module(child)
        if found is not None:
            return found
    
    return None


def _replace_attention_in_module(wrapper_module, new_attention):
    """Replace the Attention inside a wrapper module with the LoRA-injected version."""
    if hasattr(wrapper_module, 'fn'):
        inner = wrapper_module.fn
        if hasattr(inner, 'fn') and hasattr(inner.fn, 'to_qkv'):
            # Residual -> PreNorm -> Attention
            inner.fn = new_attention
        elif hasattr(inner, 'to_qkv'):
            # PreNorm -> Attention (direct)
            wrapper_module.fn = new_attention


def freeze_non_lora_params(model: nn.Module):
    """Freeze all parameters that are not part of LoRA layers."""
    total = 0
    frozen = 0
    trainable = 0
    
    for name, param in model.named_parameters():
        total += param.numel()
        is_lora_param = ('lora_' in name and 
                        ('lora_A' in name or 'lora_B' in name))
        if is_lora_param:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    
    print(f'[LoRA] Parameters: {total:,} total | '
          f'{frozen:,} frozen ({100*frozen/total:.1f}%) | '
          f'{trainable:,} trainable LoRA ({100*trainable/total:.1f}%)')


def count_lora_params(model: nn.Module):
    """Count LoRA vs total parameters."""
    lora_total = 0
    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            lora_total += param.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f'[LoRA] Trainable: {lora_total:,} / {total:,} '
          f'({100*lora_total/total:.2f}% of base model)')


# ============================================================
# Factor Extraction & Setting utilities
# ============================================================

def extract_all_lora_factors(model: nn.Module) -> List[Tuple[str, Dict, Dict]]:
    """
    Extract all LoRA (B, A) factor pairs from an injected model.
    
    Returns:
        List of (layer_name, B_dict, A_dict) tuples
        - layer_name: identifier like 'down_0_2', 'mid_attn', 'up_1_2'
        - B_dict: {factor_name: tensor} e.g. {'q': B_q, 'k': B_k, ...}
        - A_dict: {factor_name: tensor} e.g. {'q': A_q, 'k': A_k, ...}
    """
    factors = []
    
    def _scan(module, prefix=''):
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            
            if isinstance(child, (LoRAInjectedAttention, LoRAInjectedLinearAttention)):
                B, A = child.get_lora_factors()
                if B:  # Only add if there are actual LoRA factors
                    factors.append((full_name, dict(B), dict(A)))
            
            _scan(child, full_name)
    
    _scan(model)
    return factors


def set_all_lora_factors(model: nn.Module, 
                         server_factors: List[Tuple[str, Dict, Dict]]):
    """
    Set all LoRA factors from server distribution (after prefix slicing).
    Each entry should be (layer_name, B_prefix_dict, A_prefix_dict)
    where the tensors have been sliced to match client's local rank.
    """
    # Build a lookup dict
    factor_map = {name: (B, A) for name, B, A in server_factors}
    
    def _apply(module, prefix=''):
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            
            if full_name in factor_map and isinstance(child, (LoRAInjectedAttention, LoRAInjectedLinearAttention)):
                B_dict, A_dict = factor_map[full_name]
                child.set_lora_factors(B_dict, A_dict)
            
            _apply(child, full_name)
    
    _apply(model)


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get state dict containing only LoRA parameters."""
    return {
        name: param for name, param in model.named_parameters()
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name)
    }


def compute_true_deltas_from_factors(factors: List[Tuple[str, Dict, Dict]]) -> Dict[str, torch.Tensor]:
    """
    Compute true weight updates ΔW = B @ A for each layer/factor pair.
    
    Args:
        factors: List of (layer_name, B_dict, A_dict) as returned by extract_all_lora_factors
    
    Returns:
        Dict mapping '{layer_name}_{factor_key}' → ΔW tensor (2D matrix)
    """
    deltas = {}
    for layer_name, B_dict, A_dict in factors:
        for key in B_dict:
            delta = B_dict[key] @ A_dict[key]  # [out, in]
            deltas[f'{layer_name}_{key}'] = delta
    return deltas


def count_communication_bytes(factors: List[Tuple[str, Dict, Dict]]) -> int:
    """Estimate upload communication cost in bytes (float32)."""
    total_elems = 0
    for _, B_dict, A_dict in factors:
        for t in list(B_dict.values()) + list(A_dict.values()):
            total_elems += t.numel()
    return total_elems * 4  # float32 = 4 bytes
