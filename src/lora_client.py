#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA-aware Federated Client for Diffusion Model Training.

Key differences from original Client:
  - Base U-Net weights are frozen; only LoRA (B, A) factors are trained
  - Returns LoRA factor pairs instead of full state_dict
  - Accepts server-distributed LoRA factors as initialization
"""

from typing import Dict, List, Optional, Tuple

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lora import (
    extract_all_lora_factors,
    set_all_lora_factors,
    get_lora_state_dict,
    count_communication_bytes,
)


class LoRADatasetSplit(Dataset):
    """Wrapper around a subset of the base dataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.dataset[self.indices[item]]
        return image, torch.tensor(label)


class LoRAClient:
    """
    A federated client that trains only LoRA parameters on top of a frozen base model.
    
    The client:
      1. Receives global LoRA factor initialization from server (or uses random init)
      2. Trains only LoRA (B, A) on local data for local_ep epochs
      3. Extracts trained LoRA factor pairs to send back to server
    """

    def __init__(
        self,
        args,
        dataset,
        base_model,          # Frozen base U-Net with injected LoRA
        indices,             # Data partition indices for this client
        time_steps,          # Diffusion timesteps
        diffuser,            # Diffuser instance
        lora_rank: int = 8,  # This client's LoRA rank
        lora_alpha: float = 1.0,
    ):
        self.args = args
        self.indices = list(indices)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.time_steps = time_steps
        self.diffuser = diffuser

        # Create local model by deep-copying the injected base model
        self.local_model = copy.deepcopy(base_model)
        
        # Setup data loader
        self.train_loader = DataLoader(
            LoRADatasetSplit(dataset, indices),
            batch_size=self.args.local_bs,
            shuffle=True,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model.to(self.device)

        # Ensure only LoRA params are trainable (base should already be frozen)
        self._freeze_base_params()

    def _freeze_base_params(self):
        """Ensure all non-LoRA parameters are frozen."""
        for name, param in self.local_model.named_parameters():
            is_lora = ('lora_A' in name or 'lora_B' in name)
            param.requires_grad = is_lora

    def receive_server_update(
        self,
        server_factors: Optional[List[Tuple[str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
    ):
        """
        Receive and apply server's global LoRA factor distribution.
        
        Args:
            server_factors: List of (layer_name, B_dict, A_dict) from server prefix slicing.
                           If None, keeps current factors (for round 0 random init).
        """
        if server_factors is not None:
            set_all_lora_factors(self.local_model, server_factors)
            print(f'  [Client] Received server LoRA update ({len(server_factors)} layers)')

    @torch.no_grad()
    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.local_model.parameters() if p.requires_grad)

    def train_local(
        self,
    ) -> Tuple[List[Tuple[str, Dict, Dict]], float]:
        """
        Perform local training on LoRA parameters only.
        
        Returns:
            lora_factors: Extracted LoRA (B, A) factor pairs after training
            avg_loss: Average loss over all local epochs
        """
        self.local_model.train()

        # Optimizer only for trainable (LoRA) parameters
        trainable_params = [p for p in self.local_model.parameters() if p.requires_grad]
        
        if not trainable_params:
            print('  [Client] WARNING: No trainable parameters found!')
            return [], 0.0

        n_trainable = sum(p.numel() for p in trainable_params)
        print(f'  [Client] Training {len(trainable_params)} LoRA param groups '
              f'({n_trainable:,} params, rank={self.lora_rank})')

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr,
                                         weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr)

        # Cosine annealing LR scheduler for stability
        total_steps = self.args.local_ep * len(self.train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        epoch_losses = []

        for epoch in range(self.args.local_ep):
            batch_losses = []
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.local_model.zero_grad()

                # Sample timesteps uniformly
                t = torch.randint(0, int(self.time_steps),
                                  (images.shape[0],), device=self.device).long()

                # DDPM loss
                loss = self.diffuser.p_losses(
                    self.local_model, images, t,
                    loss_type="huber", labels=labels,
                )

                if batch_idx % 100 == 0:
                    print(f'    Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}')

                loss.backward()
                
                # Gradient clipping for LoRA training stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                batch_losses.append(loss.item())

            epoch_avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(epoch_avg)
            print(f'  [Client] Epoch {epoch} avg loss: {epoch_avg:.4f}')

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # --- Extract LoRA factors to send back ---
        lora_factors = extract_all_lora_factors(self.local_model)

        comm_bytes = count_communication_bytes(lora_factors)
        print(f'  [Client] Training complete. Avg loss: {avg_loss:.4f}, '
              f'Upload: {comm_bytes / 1024:.1f} KB, {len(lora_factors)} LoRA layers')

        return lora_factors, avg_loss


def create_heterogeneous_clients(
    args,
    dataset,
    base_model_with_lora,
    client_groups: dict,       # {client_id: indices}
    time_steps,
    diffuser,
    rank_config,               # {client_id: rank} or list of ranks or single int
    alpha: float = 1.0,
) -> Dict[int, LoRAClient]:
    """
    Create multiple LoRA clients with heterogeneous ranks.
    
    Args:
        args: Command line arguments
        dataset: Base dataset
        base_model_with_lora: U-Net with LoRA injected (will be deep-copied per client)
        client_groups: {client_id: data_indices}
        time_steps: Diffusion timesteps
        diffuser: Diffuser instance
        rank_config: 
            - int: uniform rank for all clients
            - list: per-client ranks (index = client_id)
            - dict: {client_id: rank}
    
    Returns:
        {client_id: LoRAClient}
    """
    clients = {}
    
    # Normalize rank_config to dict
    if isinstance(rank_config, int):
        rank_map = {cid: rank_config for cid in client_groups}
    elif isinstance(rank_config, (list, tuple)):
        rank_map = {cid: rank_config[cid] for cid in range(len(rank_config)) if cid in client_groups}
    elif isinstance(rank_config, dict):
        rank_map = rank_config
    else:
        raise ValueError(f"Unsupported rank_config type: {type(rank_config)}")
    
    for cid, indices in client_groups.items():
        r = rank_map.get(cid, 8)  # default rank
        print(f'[LoRA Client] Creating client {cid}: rank={r}, data={len(indices)} samples')
        
        clients[cid] = LoRAClient(
            args=args,
            dataset=dataset,
            base_model=base_model_with_lora,
            indices=indices,
            time_steps=time_steps,
            diffuser=diffuser,
            lora_rank=r,
            lora_alpha=alpha,
        )
    
    return clients
