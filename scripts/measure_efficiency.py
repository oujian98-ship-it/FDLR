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
    from unet import Unet, UnetConditional
except ModuleNotFoundError as e:
    missing = e.name or str(e)
    print(f"[Error] Missing dependency while importing U-Net: {missing}")
    print("Install project requirements first, for example: pip install -r requirements.txt")
    raise SystemExit(2)


def parse_dim_mults(s):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--conditional", type=int, default=0)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--dim_mults", type=str, default="1,2,2,2")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    dim_mults = parse_dim_mults(args.dim_mults)
    if args.conditional:
        model = UnetConditional(
            dim=args.model_dim,
            channels=args.num_channels,
            dim_mults=dim_mults,
            num_classes=args.num_classes,
        )
    else:
        model = Unet(
            dim=args.model_dim,
            channels=args.num_channels,
            dim_mults=dim_mults,
        )

    total, trainable = count_params(model)
    print(f"#Params total    : {total / 1e6:.3f} M")
    print(f"#Params trainable: {trainable / 1e6:.3f} M")

    try:
        from thop import profile

        x = torch.randn(args.batch_size, args.num_channels, args.image_size, args.image_size)
        t = torch.randint(0, 100, (args.batch_size,))
        if args.conditional:
            y = torch.randint(0, args.num_classes, (args.batch_size,))
            macs, params = profile(model, inputs=(x, t, y), verbose=False)
        else:
            macs, params = profile(model, inputs=(x, t), verbose=False)

        print(f"MACs             : {macs / 1e9:.3f} G")
        print(f"THOP Params      : {params / 1e6:.3f} M")
    except Exception as e:
        print("[Warning] MACs measurement failed.")
        print("Install thop: pip install thop")
        print(e)


if __name__ == "__main__":
    main()
