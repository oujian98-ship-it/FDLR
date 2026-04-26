import torch
import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from lora import extract_all_lora_factors
from hetero_aggregation import (
    compute_aggregation_weights,
    prefix_slice_distribution,
    procrustes_alignment_per_layer,
)


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


def test_rank_beta_weights():
    weights = compute_aggregation_weights(
        data_sizes={0: 100, 1: 100, 2: 100},
        client_ranks={0: 4, 1: 16, 2: 64},
        rank_correction=True,
        rank_beta=0.5,
    )
    assert weights[0] > weights[1] > weights[2]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_prefix_distribution_shapes():
    B_global = {"layer": {"q": torch.randn(5, 8)}}
    A_global = {"layer": {"q": torch.randn(8, 7)}}
    updates = prefix_slice_distribution(B_global, A_global, {0: 2, 1: 6})
    assert updates[0][0]["layer"]["q"].shape == (5, 2)
    assert updates[0][1]["layer"]["q"].shape == (2, 7)
    assert updates[1][0]["layer"]["q"].shape == (5, 6)
    assert updates[1][1]["layer"]["q"].shape == (6, 7)


def test_procrustes_product_preserved():
    torch.manual_seed(0)
    B = {"layer": {"q": torch.randn(6, 4)}}
    A = {"layer": {"q": torch.randn(4, 5)}}
    Q, _ = torch.linalg.qr(torch.randn(4, 4))
    B_prev = {"layer": {"q": B["layer"]["q"] @ Q}}
    A_prev = {"layer": {"q": Q.t() @ A["layer"]["q"]}}

    B_aligned, A_aligned = procrustes_alignment_per_layer(B, A, B_prev, A_prev)
    before = B["layer"]["q"] @ A["layer"]["q"]
    after = B_aligned["layer"]["q"] @ A_aligned["layer"]["q"]
    assert torch.allclose(before, after, atol=1e-5)

if __name__ == "__main__":
    print("Test utilities ready.")
