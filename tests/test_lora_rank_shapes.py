import torch
import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from lora import extract_all_lora_factors


def assert_lora_rank(model, expected_rank: int):
    factors = extract_all_lora_factors(model)
    assert len(factors) > 0, 'No LoRA factors found'
    for layer_name, B_dict, A_dict in factors:
        for key, B in B_dict.items():
            A = A_dict[key]
            assert B.shape[1] == expected_rank, (layer_name, key, B.shape, expected_rank)
            assert A.shape[0] == expected_rank, (layer_name, key, A.shape, expected_rank)


def assert_product_preserved(B, A, Q, atol=1e-5):
    before = B @ A
    after = (B @ Q) @ (Q.t() @ A)
    assert torch.allclose(before, after, atol=atol), (before - after).abs().max().item()

if __name__ == "__main__":
    print("Test utilities ready.")
