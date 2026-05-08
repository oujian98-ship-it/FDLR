#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean stale generated-image export directories.

By default this script is a dry run. Pass --execute to actually delete.
It only targets generated sample directories under ./exports, and skips
dataset reference exports such as exports/fmnist/dataset_train.
"""

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"


DEFAULT_PATTERNS = (
    "infer_*",
    "model_*",
    "flora_*",
    "fedavg_*",
)


def has_images(path: Path) -> bool:
    return any(path.glob("*.png")) or any(path.glob("*.jpg")) or any(path.glob("*.jpeg"))


def is_safe_export_dir(path: Path) -> bool:
    resolved = path.resolve()
    exports_root = EXPORTS_DIR.resolve()
    if not resolved.is_dir():
        return False
    if resolved == exports_root:
        return False
    if exports_root not in resolved.parents:
        return False

    # Keep real reference datasets, e.g. exports/fmnist/dataset_train.
    if resolved.parent.name in {"fmnist", "cifar", "cifar10", "celeba"}:
        return False

    return has_images(resolved)


def collect_targets(patterns, include_all_generated=False):
    candidates = set()
    if include_all_generated:
        candidates.update(p for p in EXPORTS_DIR.iterdir() if p.is_dir())
    else:
        for pattern in patterns:
            candidates.update(EXPORTS_DIR.glob(pattern))
    return sorted(p for p in candidates if is_safe_export_dir(p))


def main():
    parser = argparse.ArgumentParser(
        description="Delete stale generated image directories under ./exports."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched directories. Without this, only preview.",
    )
    parser.add_argument(
        "--all-generated",
        action="store_true",
        help="Clean every top-level generated export directory that contains images.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Additional top-level exports glob pattern, e.g. 'infer_fedavg_model_fmnist*'.",
    )
    args = parser.parse_args()

    if not EXPORTS_DIR.exists():
        print(f"Exports directory not found: {EXPORTS_DIR}")
        return

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    targets = collect_targets(patterns, include_all_generated=args.all_generated)

    if not targets:
        print("No generated export directories matched.")
        return

    mode = "DELETE" if args.execute else "DRY RUN"
    print(f"[{mode}] Matched {len(targets)} generated export directories:")
    for target in targets:
        image_count = sum(1 for _ in target.glob("*.png"))
        print(f"  - {target} ({image_count} png files)")

    if not args.execute:
        print("\nNo files were deleted. Re-run with --execute to delete these directories.")
        return

    for target in targets:
        shutil.rmtree(target)
    print(f"\nDeleted {len(targets)} directories.")


if __name__ == "__main__":
    main()
