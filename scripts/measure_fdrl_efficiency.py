#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from lora import inject_lora_into_unet
    from lora_federator import build_base_model, resolve_lora_alpha
except ModuleNotFoundError as e:
    missing = e.name or str(e)
    print(f"[Error] Missing dependency while importing FDLR modules: {missing}")
    print("Install project requirements first, for example: pip install -r requirements.txt")
    raise SystemExit(2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--model_dim", type=int, default=128)
    p.add_argument("--dim_mults", type=str, default="1,2,4")
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--conditional", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha_mode", type=str, default="rank", choices=["rank", "fixed"])
    p.add_argument("--lora_alpha", type=float, default=-1.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_layers", type=str, default="attention")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # build_base_model expects these attrs.
    args.load_model = ""
    args.train = 1

    model = build_base_model(args, device)
    alpha = resolve_lora_alpha(args, args.lora_rank)
    inject_lora_into_unet(
        model,
        rank=args.lora_rank,
        alpha=alpha,
        dropout=args.lora_dropout,
        target_layers=args.lora_target_layers,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("Parameter statistics")
    print("=" * 60)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print(f"Trainable ratio  : {100 * trainable_params / total_params:.4f}%")

    try:
        from thop import profile

        x = torch.randn(1, args.num_channels, args.image_size, args.image_size).to(device)
        t = torch.randint(0, 100, (1,), device=device).long()
        if args.conditional:
            y = torch.randint(0, args.num_classes, (1,), device=device).long()
            macs, params = profile(model, inputs=(x, t, y), verbose=False)
        else:
            macs, params = profile(model, inputs=(x, t), verbose=False)

        print("=" * 60)
        print("MACs")
        print("=" * 60)
        print(f"MACs             : {macs:,}")
        print(f"MACs (G)         : {macs / 1e9:.4f}")
        print(f"THOP params      : {params:,}")
    except Exception as e:
        print("[Warning] MACs not computed.")
        print("Install thop with: pip install thop")
        print(f"Reason: {e}")


if __name__ == "__main__":
    main()
