#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedDiffuse 统一入口程序

统一管理两种训练方法的运行：
  - fedavg: 原始 FedAvg 全参训练 (federator.py)
  - lora:   异构 LoRA 联邦微调 (lora_federator.py)

用法:
    python main.py --dataset celeba
    python main.py --preset celeba_lora_quick
    python main.py --method fedavg --dataset celeba
"""

import argparse
import os
import sys
from pathlib import Path

# ============================================================
# 0. 路径设置
# ============================================================
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
str_src = str(SRC_DIR)
if str_src not in sys.path:
    sys.path.insert(0, str_src)


# ============================================================
# 1. 超参数配置 (修改这里的默认值即可)
# ============================================================

# ---- 数据集选择 ----
DATASET = 'fmnist'          # 数据集: celeba | fmnist | cifar10
DATA_ROOT = ''                 # 自定义数据路径 (留空则根据 DATASET 自动匹配)
DOWNLOAD_DATASET = 0           # torchvision 是否自动下载数据
PARTITION = ''                 # fedphd-cifar2 | fedphd-celeba4 | 空字符串
SEED = 42                    # 随机种子

# ---- 训练方法选择 ----
METHOD = 'lora'             # 方法: lora | fedavg

# ---- 联邦学习超参 ----
ROUNDS = 100                 # 全局训练轮数 R
NUM_USERS = 2               # 客户端数量 K
FRAC = 1.0                  # 每轮参与客户端比例 C
LOCAL_EP = 5                # 本地训练轮次 E
LOCAL_BS = 128              # 本地 batch size B
IID = 1                     # IID=1 / Non-IID=0
UNEQUAL = 0                 # 非均匀分布=1

# ---- 扩散模型超参 ----
RUN_MODE = ''               # train | eval | train_eval；留空则兼容旧的 TRAIN_MODE
TRAIN_MODE = 1              # 训练=1 / 推理=0
LOAD_MODEL = ''             # 预加载模型路径；FedAvg 训练留空表示从随机初始化开始
TIME_STEPS = 1000           # 扩散步数 T
CONDITIONAL = -1            # 条件生成 (=-1按数据集自动: FMNIST=1, CelebA=0)
LR = 0.01                   # 学习率
OPTIMIZER = 'adam'          # 优化器
MODEL_DIM = 0               # U-Net base channels；0 表示使用 image_size
DIM_MULTS = '1,2,4'         # U-Net dim multipliers，例如 FedPhD 对齐可试 1,2,2,2

# ---- LoRA 专用 (仅 METHOD=lora 时生效) ----
LORA_RANK = 32               # 基础 LoRA rank
LORA_RANKS = ''   # 各客户端逗号分隔的 rank, 如 "4,8,16,8,4" (留空则全部用 LORA_RANK)
GLOBAL_LORA_RANK = 32       # 服务端全局 rank，必须 >= 最大客户端 rank
LORA_ALPHA = -1.0           # <=0 表示 alpha=rank，使 LoRA scaling=1
LORA_ALPHA_MODE = 'rank'    # rank: alpha=rank；fixed: 使用 LORA_ALPHA
LORA_DROPOUT = 0.1          # LoRA dropout
RANK_BETA = 0.5             # rank 校正 eta(r)=r^(-beta)，消融可设 0/0.5/1
RANK_CORRECTION = 1         # 是否启用 rank 校正
USE_PROCRUSTES = 1          # 是否启用 Procrustes 对齐
USE_PREFIX_INIT = 1         # 第 0 轮前是否 prefix 初始化
AGG_MODE = 'fdlr'           # fdlr | update_space | factor_avg | fedavg_lora

FEDAVG_TRAIN_MODE = 'full'  # full | usplit | udec | ulatdec
MOMENTUM = 0.5              # SGD momentum
ROUND_OFFSET = 0            # 轮次编号偏移

# ---- 导出/采样 ----
EXPORT_SAMPLES = 0          # 生成图片数量
EXPORT_DATASET = 0          # 导出真实数据样本数 (FID参照)
SHOW_SAMPLES = 0            # 展示样本
EXP_ROUNDS = 0              # 中间轮次导出间隔
EVAL_NUM_SAMPLES = 30000    # FedPhD 对齐评估生成样本数
EVAL_BATCH_SIZE = 256       # FedPhD 对齐评估 batch size
CENTRAL_AGG_INTERVAL = 5    # FedPhD central aggregation 通信统计窗口
CHECKPOINT_INTERVAL = 10    # 中间 checkpoint 保存间隔；0 表示只保存最终模型
RUN_EVAL_AFTER_TRAIN = 0    # 训练完成后是否立刻导出/评估；0 表示训练和评估分开跑
LOG_TIMESTAMP = ''          # 可手动指定日志文件时间戳；留空则自动生成
COMPUTE_IS = 0              # 评估时计算 Inception Score
COMPUTE_PR = 0              # 评估时计算 Precision/Recall
EVAL_REAL_SPLIT = 'train'   # FedPhD-style 使用 train split 作为真实参考
DATA_RANGE = 'minus1_1'     # minus1_1=协议范围；0_1=兼容旧 model_cifar.pth
MODEL_BACKEND = 'custom'    # custom | diffusers
HF_MODEL_ID = 'google/ddpm-cifar10-32'

# ---- DDIM 加速采样 ----
USE_DDIM = 1                # 使用 DDIM 采样 (1=开启, 0=关闭/使用 DDPM)
DDIM_STEPS = 100            # DDIM 采样步数 (默认 100, 建议 50-100)


# ============================================================
# 2. 数据集自动匹配表
# ============================================================

DATASET_CONFIGS = {
    'fmnist': {
        'image_size': 28,
        'num_channels': 1,
        'num_classes': 10,
        'conditional': 1,           # FMNIST 必须用条件生成
        'default_model': 'model_fmnist.pth',
        'data_root': r'D:\data\fashion-mnist-master',
        'eval_num_samples': 5000,
        'eval_batch_size': 128,
    },
    'cifar10': {
        'image_size': 32,
        'num_channels': 3,
        'num_classes': 10,
        'conditional': 0,           # FedPhD CIFAR10 协议默认无条件生成
        'default_model': 'model_cifar.pth',
        'data_root': r'D:\data\cifar-10-python',
        'eval_num_samples': 30000,
        'eval_batch_size': 128,
    },
    'cifar': {
        'image_size': 32,
        'num_channels': 3,
        'num_classes': 10,
        'conditional': 0,
        'default_model': 'model_cifar.pth',
        'data_root': r'D:\data\cifar-10-python',
        'eval_num_samples': 30000,
        'eval_batch_size': 128,
    },
    'celeba': {
        'image_size': 64,
        'num_channels': 3,
        'num_classes': 16,
        'conditional': 0,           # CelebA 无条件即可
        'default_model': 'model_celeba.pth',
        'data_root': r'D:\data\CelebA',
        'eval_num_samples': 5000,
        'eval_batch_size': 64,
    },
}


def get_dataset_config(dataset_name):
    """根据数据集名称获取默认配置"""
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        print(f'[Error] Unknown dataset "{dataset_name}", available: {available}')
        sys.exit(1)
    return DATASET_CONFIGS[dataset_name]


def auto_data_root(dataset_name):
    """根据数据集自动匹配数据路径（用户未手动指定时使用）"""
    if DATA_ROOT:
        return DATA_ROOT  # 用户在顶部手动指定了，优先使用
    ds_cfg = get_dataset_config(dataset_name)
    return ds_cfg.get('data_root', '')


def auto_select_model(dataset_name):
    """根据数据集自动选择预加载模型"""
    if LOAD_MODEL:
        return LOAD_MODEL
    config = get_dataset_config(dataset_name)
    model_path = PROJECT_ROOT / config['default_model']
    if not model_path.exists():
        print(f'[Warning] Model file not found: {model_path}')
        return ''
    return config['default_model']


def is_fedavg_training_from_scratch(args):
    """FedAvg full/usplit/udec/ulatdec training starts from random init by default."""
    return (
        getattr(args, 'method', '') == 'fedavg'
        and int(getattr(args, 'train', 1)) == 1
        and getattr(args, 'train_mode', 'full') in {'full', 'usplit', 'udec', 'ulatdec'}
    )


def initial_load_model_for_config(method, train):
    """Choose the config-level load model before CLI/preset overrides."""
    if LOAD_MODEL:
        return LOAD_MODEL
    if method == 'fedavg' and int(train) == 1:
        return ''
    return auto_select_model(DATASET)


def apply_run_mode(args):
    """Map the clearer run_mode API onto legacy train/run_eval_after_train flags."""
    run_mode = getattr(args, 'run_mode', '') or ''
    if not run_mode:
        return args

    if run_mode == 'train':
        args.train = 1
        args.run_eval_after_train = 0
    elif run_mode == 'eval':
        args.train = 0
        args.run_eval_after_train = 0
    elif run_mode == 'train_eval':
        args.train = 1
        args.run_eval_after_train = 1
    else:
        raise ValueError(f'Unsupported --run_mode "{run_mode}". Use train, eval, or train_eval.')

    if run_mode in {'eval', 'train_eval'}:
        eval_count = int(getattr(args, 'eval_num_samples', 0) or 0)
        if int(getattr(args, 'export_samples', 0) or 0) <= 0:
            args.export_samples = eval_count
        if int(getattr(args, 'export_dataset', 0) or 0) <= 0:
            args.export_dataset = eval_count
    return args


# ============================================================
# 3. 预设配置 (Presets) — 可用 --preset 快速切换
# ============================================================

PRESETS = {
    'celeba_lora_quick_homo': {
        'description': 'CelebA 同构 LoRA 快速实验 (R=10, rank=8)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 10, 'num_users': 5, 'local_ep': 3, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '', 'global_lora_rank': 8,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1,
    },
    'celeba_lora_quick_hetero': {
        'description': 'CelebA 异构 LoRA 快速实验 (R=10, ranks=4,8,16,8,4)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 10, 'num_users': 5, 'local_ep': 3, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1,
    },
    'celeba_lora_paper_align_homo': {
        'description': 'CelebA 同构 LoRA 对齐实验 (R=30, E=5, B=64, rank=8)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '', 'global_lora_rank': 8,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1,
    },
    'celeba_lora_paper_align_hetero': {
        'description': 'CelebA 异构 LoRA 对齐实验 (R=30, E=5, B=64, ranks=4,8,16,8,4)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1,
    },
    'celeba_fedphd_protocol_homo': {
        'description': 'CelebA FedPhD-CelebA4 同构 LoRA 对齐协议 (R=2000, K=20, E=5, B=128, rank=8)',
        'method': 'lora', 'dataset': 'celeba',
        'load_model': 'model_celeba.pth',
        'rounds': 2000, 'num_users': 20, 'frac': 0.2, 'local_ep': 5, 'local_bs': 128,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-celeba4',
        'num_classes': 4, 'conditional': 0,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '', 'global_lora_rank': 8,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'eval_real_split': 'train',
        'train': 1,
    },
    'celeba_fedphd_protocol_hetero': {
        'description': 'CelebA FedPhD-CelebA4 异构 LoRA 对齐协议 (R=2000, K=20, E=5, B=128, ranks=4,8,16)',
        'method': 'lora', 'dataset': 'celeba',
        'load_model': 'model_celeba.pth',
        'rounds': 2000, 'num_users': 20, 'frac': 0.2, 'local_ep': 5, 'local_bs': 128,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-celeba4',
        'num_classes': 4, 'conditional': 0,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'eval_real_split': 'train',
        'train': 1,
    },
    'celeba_fedphd_smoke_homo': {
        'description': 'CelebA FedPhD-CelebA4 同构 LoRA smoke test (R=2, E=1)',
        'method': 'lora', 'dataset': 'celeba',
        'load_model': 'model_celeba.pth',
        'rounds': 2, 'num_users': 20, 'frac': 0.2, 'local_ep': 1, 'local_bs': 16,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-celeba4',
        'num_classes': 4, 'conditional': 0,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '', 'global_lora_rank': 8,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 500, 'eval_batch_size': 64,
        'eval_real_split': 'train',
        'train': 1,
    },
    'celeba_fedphd_smoke_hetero': {
        'description': 'CelebA FedPhD-CelebA4 异构 LoRA smoke test (R=2, E=1)',
        'method': 'lora', 'dataset': 'celeba',
        'load_model': 'model_celeba.pth',
        'rounds': 2, 'num_users': 20, 'frac': 0.2, 'local_ep': 1, 'local_bs': 16,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-celeba4',
        'num_classes': 4, 'conditional': 0,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 500, 'eval_batch_size': 64,
        'eval_real_split': 'train',
        'train': 1,
    },
    'celeba_lora_quick': {
        'description': 'CelebA LoRA 快速实验 alias: 异构 (R=10)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 10, 'num_users': 5, 'local_ep': 3, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1,
    },
    'celeba_lora_paper_align': {
        'description': 'CelebA LoRA 对齐实验 alias: 异构 (R=30, E=5, B=64)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1,
    },
    'celeba_fedavg_baseline': {
        'description': 'CelebA FedAvg Baseline 论文对齐 (R=30, ~37h)',
        'method': 'fedavg', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'train_mode': 'full',
        'train': 1,
    },
    'celeba_fedavg_quick': {
        'description': 'CelebA FedAvg 快速 Baseline (R=10, ~8h)',
        'method': 'fedavg', 'dataset': 'celeba',
        'rounds': 10, 'num_users': 5, 'local_ep': 3, 'local_bs': 128,
        'lr': 1e-4,
        'train_mode': 'full',
        'train': 1,
    },
    'fmnist_lora_quick_homo': {
        'description': 'Fashion-MNIST 同构 LoRA 快速实验 (rank=8)',
        'method': 'lora', 'dataset': 'fmnist',
        'load_model': 'model_fmnist.pth',
        'rounds': 15, 'num_users': 5, 'local_ep': 5, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 32, 'lora_ranks': '', 'global_lora_rank': 32,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 128,
        'train': 1,
    },
    'fmnist_lora_quick_hetero': {
        'description': 'Fashion-MNIST 异构 LoRA 快速实验 (ranks=4,8,16,8,4)',
        'method': 'lora', 'dataset': 'fmnist',
        'load_model': 'model_fmnist.pth',
        'rounds': 15, 'num_users': 5, 'local_ep': 5, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 128,
        'train': 1,
    },
    'fmnist_lora_quick': {
        'description': 'Fashion-MNIST LoRA 快速实验 alias: 异构',
        'method': 'lora', 'dataset': 'fmnist',
        'load_model': 'model_fmnist.pth',
        'rounds': 15, 'num_users': 5, 'local_ep': 5, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 128,
        'train': 1,
    },
    'cifar10_lora_quick': {
        'description': 'CIFAR10 LoRA 快速实验',
        'method': 'lora', 'dataset': 'cifar10',
        'rounds': 10, 'num_users': 5, 'local_ep': 3, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'train': 1,
    },
    'cifar10_fedphd_smoke': {
        'description': 'CIFAR10 FedPhD 协议 smoke test (R=2)',
        'method': 'lora', 'dataset': 'cifar10',
        'rounds': 2, 'num_users': 20, 'frac': 0.2, 'local_ep': 1, 'local_bs': 32,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-cifar2',
        'time_steps': 100, 'conditional': 0,
        'model_dim': 128, 'dim_mults': '1,2,2,2',
        'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8',
        'global_lora_rank': 16, 'lora_alpha_mode': 'rank',
        'use_ddim': 1, 'ddim_steps': 100, 'eval_num_samples': 1000, 'eval_batch_size': 256,
        'eval_real_split': 'train', 'central_agg_interval': 5, 'seed': 2023, 'train': 1,
    },
    'cifar10_fedphd_protocol': {
        'description': 'CIFAR10 FedPhD baseline-equivalent 协议 (R=2000, E=5)',
        'method': 'lora', 'dataset': 'cifar10',
        'rounds': 2000, 'num_users': 20, 'frac': 0.2, 'local_ep': 5, 'local_bs': 128,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-cifar2',
        'time_steps': 100, 'conditional': 0,
        'model_dim': 128, 'dim_mults': '1,2,2,2',
        'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8',
        'global_lora_rank': 16, 'lora_alpha_mode': 'rank',
        'use_ddim': 1, 'ddim_steps': 100, 'eval_num_samples': 30000, 'eval_batch_size': 256,
        'eval_real_split': 'train', 'central_agg_interval': 5, 'seed': 2023, 'train': 1,
    },
    'cifar10_hf_fedphd_smoke': {
        'description': 'CIFAR10 Hugging Face DDPM base smoke test (R=2)',
        'method': 'lora', 'dataset': 'cifar10',
        'model_backend': 'diffusers', 'hf_model_id': 'google/ddpm-cifar10-32',
        'load_model': '', 'data_range': 'minus1_1', 'disable_cudnn': 1,
        'rounds': 2, 'num_users': 20, 'frac': 0.05, 'local_ep': 1, 'local_bs': 8,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-cifar2',
        'time_steps': 1000, 'conditional': 0,
        'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8',
        'global_lora_rank': 16, 'lora_alpha_mode': 'rank',
        'use_ddim': 1, 'ddim_steps': 100, 'eval_num_samples': 1000, 'eval_batch_size': 128,
        'eval_real_split': 'train', 'central_agg_interval': 5, 'seed': 2023, 'train': 1,
    },
    'cifar10_hf_fedphd_protocol': {
        'description': 'CIFAR10 Hugging Face DDPM base FedPhD-aligned protocol (R=2000, E=5)',
        'method': 'lora', 'dataset': 'cifar10',
        'model_backend': 'diffusers', 'hf_model_id': 'google/ddpm-cifar10-32',
        'load_model': '', 'data_range': 'minus1_1',
        'rounds': 2000, 'num_users': 20, 'frac': 0.2, 'local_ep': 5, 'local_bs': 128,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-cifar2',
        'time_steps': 1000, 'conditional': 0,
        'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8',
        'global_lora_rank': 16, 'lora_alpha_mode': 'rank',
        'use_ddim': 1, 'ddim_steps': 100, 'eval_num_samples': 30000, 'eval_batch_size': 128,
        'eval_real_split': 'train', 'central_agg_interval': 5, 'seed': 2023, 'train': 1,
    },
    # ---- 推理评估 (train=0) ----
    # LoRA 推理
    'infer_celeba_lora': {
        'description': 'CelebA LoRA 推理: 生成图片 + FID评估',
        'method': 'lora', 'dataset': 'celeba', 'train': 0,
        'load_model': 'flora_model_celeba_R[30]_K[5]_E[5].pth',
        'export_samples': 5000, 'export_dataset': 5000,
    },
    'infer_fmnist_lora': {
        'description': 'FMNIST LoRA 推理: 生成图片 + FID评估',
        'method': 'lora', 'dataset': 'fmnist', 'train': 0,
        'load_model': 'flora_model_fmnist_R[15]_K[5]_E[5].pth',
        'export_samples': 5000, 'export_dataset': 5000,
    },
    'infer_cifar10_lora': {
        'description': 'CIFAR10 LoRA 推理: 生成图片 + FID评估',
        'method': 'lora', 'dataset': 'cifar10', 'train': 0,
        'load_model': 'flora_model_cifar10_R[2000]_K[20]_E[5].pth',
        'export_samples': 30000, 'export_dataset': 30000,
    },
    # FedAvg 推理
    'infer_celeba_fedavg': {
        'description': 'CelebA FedAvg 推理: 生成图片 + FID评估',
        'method': 'fedavg', 'dataset': 'celeba', 'train': 0,
        'train_mode': 'full',
        'load_model': 'fedavg_model_celeba_full_R[30]_K[5]_E[5].pth',
        'export_samples': 5000, 'export_dataset': 5000,
    },
    'infer_fmnist_fedavg': {
        'description': 'FMNIST FedAvg 推理: 生成图片 + FID评估',
        'method': 'fedavg', 'dataset': 'fmnist', 'train': 0,
        'train_mode': 'full',
        'load_model': 'fedavg_model_fmnist_full_R[15]_K[5]_E[5].pth',
        'export_samples': 5000, 'export_dataset': 5000,
    },
}

def apply_preset(preset_name):
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        print(f'[Error] Unknown preset "{preset_name}"')
        print(f'Available presets: {available}')
        sys.exit(1)
    p = PRESETS[preset_name]
    print(f'\n{"="*60}')
    print(f'Preset: {preset_name} → {p["description"]}')
    print(f'{"="*60}\n')
    return p


def build_args_from_config():
    """从顶部配置 + 数据集自动匹配 构建 Namespace"""
    ds_cfg = get_dataset_config(DATASET)

    args = argparse.Namespace(
        # 核心选择
        method=METHOD,
        run_mode=RUN_MODE,
        preset='',
        # 数据集 (自动填充 image_size/num_channels/num_classes/data_root)
        dataset=DATASET,
        model_backend=MODEL_BACKEND,
        hf_model_id=HF_MODEL_ID,
        disable_cudnn=0,
        data_root=auto_data_root(DATASET),
        download_dataset=DOWNLOAD_DATASET,
        partition=PARTITION,
        seed=SEED,
        image_size=ds_cfg['image_size'],
        num_channels=ds_cfg['num_channels'],
        num_classes=ds_cfg['num_classes'],
        # 联邦学习
        rounds=ROUNDS,
        num_users=NUM_USERS,
        frac=FRAC,
        local_ep=LOCAL_EP,
        local_bs=LOCAL_BS,
        iid=IID,
        unequal=UNEQUAL,
        # 扩散模型
        train=TRAIN_MODE,
        load_model=initial_load_model_for_config(METHOD, TRAIN_MODE),
        resume_model='',
        start_round=0,
        cache_clients=0,
        time_steps=TIME_STEPS,
        conditional=ds_cfg.get('conditional', CONDITIONAL),  # 按数据集自动匹配
        lr=LR,
        optimizer=OPTIMIZER,
        model_dim=MODEL_DIM,
        dim_mults=DIM_MULTS,
        # 导出
        export_samples=EXPORT_SAMPLES,
        export_dataset=EXPORT_DATASET,
        show_samples=SHOW_SAMPLES,
        exp_rounds=EXP_ROUNDS,
        eval_num_samples=ds_cfg.get('eval_num_samples', EVAL_NUM_SAMPLES),
        eval_batch_size=ds_cfg.get('eval_batch_size', EVAL_BATCH_SIZE),
        central_agg_interval=CENTRAL_AGG_INTERVAL,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        run_eval_after_train=RUN_EVAL_AFTER_TRAIN,
        log_timestamp=LOG_TIMESTAMP,
        compute_is=COMPUTE_IS,
        compute_pr=COMPUTE_PR,
        eval_real_split=EVAL_REAL_SPLIT,
        data_range=DATA_RANGE,
        # LoRA
        lora_rank=LORA_RANK,
        lora_ranks=LORA_RANKS,
        global_lora_rank=GLOBAL_LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_alpha_mode=LORA_ALPHA_MODE,
        lora_dropout=LORA_DROPOUT,
        rank_beta=RANK_BETA,
        rank_correction=RANK_CORRECTION,
        use_procrustes=USE_PROCRUSTES,
        use_prefix_init=USE_PREFIX_INIT,
        agg_mode=AGG_MODE,
        # FedAvg
        train_mode=FEDAVG_TRAIN_MODE,
        momentum=MOMENTUM,
        round_offset=ROUND_OFFSET,
        # DDIM
        use_ddim=USE_DDIM,
        ddim_steps=DDIM_STEPS,
    )
    return args


def apply_cli_overrides(args):
    """命令行参数覆盖配置文件默认值"""
    parser = argparse.ArgumentParser(description='FedDiffuse', add_help=False)

    # 只注册可覆盖的参数，不设 default（保留 args 中的值）
    parser.add_argument('--method', type=str)
    parser.add_argument('--run_mode', type=str, choices=['train', 'eval', 'train_eval'])
    parser.add_argument('--preset', type=str, default='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_backend', type=str)
    parser.add_argument('--hf_model_id', type=str)
    parser.add_argument('--disable_cudnn', type=int)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--download_dataset', type=int)
    parser.add_argument('--partition', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--num_channels', type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--frac', type=float)
    parser.add_argument('--local_ep', type=int)
    parser.add_argument('--local_bs', type=int)
    parser.add_argument('--iid', type=int)
    parser.add_argument('--unequal', type=int)
    parser.add_argument('--train', type=int)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--resume_model', type=str)
    parser.add_argument('--start_round', type=int)
    parser.add_argument('--cache_clients', type=int)
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--conditional', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--model_dim', type=int)
    parser.add_argument('--dim_mults', type=str)
    parser.add_argument('--export_samples', type=int)
    parser.add_argument('--export_dataset', type=int)
    parser.add_argument('--show_samples', type=int)
    parser.add_argument('--exp_rounds', type=int)
    parser.add_argument('--eval_num_samples', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--central_agg_interval', type=int)
    parser.add_argument('--checkpoint_interval', type=int)
    parser.add_argument('--run_eval_after_train', type=int)
    parser.add_argument('--log_timestamp', type=str)
    parser.add_argument('--compute_is', type=int)
    parser.add_argument('--compute_pr', type=int)
    parser.add_argument('--eval_real_split', type=str)
    parser.add_argument('--data_range', type=str)
    parser.add_argument('--lora_rank', type=int)
    parser.add_argument('--lora_ranks', type=str)
    parser.add_argument('--global_lora_rank', type=int)
    parser.add_argument('--lora_alpha', type=float)
    parser.add_argument('--lora_alpha_mode', type=str)
    parser.add_argument('--lora_dropout', type=float)
    parser.add_argument('--rank_beta', type=float)
    parser.add_argument('--rank_correction', type=int)
    parser.add_argument('--use_procrustes', type=int)
    parser.add_argument('--use_prefix_init', type=int)
    parser.add_argument('--agg_mode', type=str)
    parser.add_argument('--train_mode', type=str)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--round_offset', type=int)
    parser.add_argument('--use_ddim', type=int)
    parser.add_argument('--ddim_steps', type=int)

    cli_args, _ = parser.parse_known_args()

    if cli_args.preset is not None:
        args.preset = cli_args.preset

    # 处理预设：先应用 preset，再用显式 CLI 参数覆盖。
    preset_keys = set()
    if args.preset:
        preset = apply_preset(args.preset)
        preset_keys = set(preset.keys())
        for key, value in preset.items():
            if key != 'description' and hasattr(args, key):
                setattr(args, key, value)

    for key, val in vars(cli_args).items():
        if val is not None:
            setattr(args, key, val)

    args = apply_run_mode(args)

    # 如果通过 --dataset 切换了数据集，重新自动匹配
    if hasattr(args, 'dataset') and args.dataset:
        ds_cfg = get_dataset_config(args.dataset)
        _is_preset_loaded = bool(args.preset)  # preset 已设置过的不应被覆盖
        # 仅在用户未显式指定时自动填充
        # (CLI 中未传 image_size 等时保持 auto 匹配)
        if '--image_size' not in sys.argv and '--num_channels' not in sys.argv:
            args.image_size = ds_cfg['image_size']
            args.num_channels = ds_cfg['num_channels']
        if '--num_classes' not in sys.argv and 'num_classes' not in preset_keys:
            args.num_classes = ds_cfg['num_classes']
        if '--conditional' not in sys.argv and not _is_preset_loaded:
            args.conditional = ds_cfg.get('conditional', 0)
        explicit_load_model = '--load_model' in sys.argv
        preset_load_model = 'load_model' in preset_keys
        if is_fedavg_training_from_scratch(args) and not explicit_load_model and not preset_load_model and not LOAD_MODEL:
            args.load_model = ''
        elif not explicit_load_model and not preset_load_model and not getattr(args, 'load_model', ''):
            args.load_model = auto_select_model(args.dataset)
        if '--data_root' not in sys.argv and (not DATA_ROOT):
            args.data_root = ds_cfg.get('data_root', '')
        if '--eval_num_samples' not in sys.argv and 'eval_num_samples' not in preset_keys:
            args.eval_num_samples = ds_cfg.get('eval_num_samples', args.eval_num_samples)
        if '--eval_batch_size' not in sys.argv and 'eval_batch_size' not in preset_keys:
            args.eval_batch_size = ds_cfg.get('eval_batch_size', args.eval_batch_size)

    return args


def print_config_summary(args):
    """打印当前运行配置摘要"""
    method_tag = 'LoRA-FedDiffuse' if args.method == 'lora' else 'FedAvg-Baseline'
    if args.train == 1 and bool(getattr(args, 'run_eval_after_train', 0)):
        mode_tag = 'TRAIN+EVAL'
    else:
        mode_tag = 'TRAIN' if args.train == 1 else 'EVAL'
    ds_info = f"{args.dataset.upper()} {args.image_size}x{args.image_size} ch={args.num_channels}"

    extra = []
    if args.method == 'lora':
        extra.append(f"rank={args.lora_rank}(client)/{args.global_lora_rank}(global)")
        if getattr(args, 'lora_ranks', ''):
            extra.append(f"hetero={args.lora_ranks}")
        extra.append(f"agg={getattr(args, 'agg_mode', 'fdlr')}")
        extra.append(f"beta={getattr(args, 'rank_beta', 0.5)}")
        extra.append(f"alpha_mode={getattr(args, 'lora_alpha_mode', 'rank')}")
    elif hasattr(args, 'train_mode'):
        extra.append(f"mode={args.train_mode}")
    if getattr(args, 'partition', ''):
        extra.append(f"partition={args.partition}")
    if getattr(args, 'run_mode', ''):
        extra.append(f"run_mode={args.run_mode}")
    if getattr(args, 'model_backend', 'custom') != 'custom':
        extra.append(f"backend={args.model_backend}")
    if getattr(args, 'model_dim', 0):
        extra.append(f"model_dim={args.model_dim}, dim_mults={args.dim_mults}")

    print(f'\n{"═"*56}')
    print(f'  FedDiffuse  │  {method_tag:<20}│  {mode_tag:<10}')
    print(f'{"─"*56}')
    print(f'  Dataset     :  {ds_info}')
    print(f'  Rounds      :  R={args.rounds},  K={args.num_users},  E={args.local_ep},  B={args.local_bs}')
    print(f'  Data dist.  :  {"IID" if args.iid else "Non-IID"}  |  lr={args.lr}')
    if extra:
        print(f'  Method spec :  {"  |  ".join(extra)}')
    if args.data_root:
        print(f'  Data root   :  {args.data_root}')
    print(f'  Load model  :  {args.load_model or "(none)"}')
    print(f'  Sampling    :  {"DDIM (steps=" + str(args.ddim_steps) + ")" if args.use_ddim else "DDPM (steps=" + str(int(args.time_steps)) + ")"}')
    print(f'  Eval        :  samples={getattr(args, "eval_num_samples", 30000)}, batch={getattr(args, "eval_batch_size", 256)}')
    print(f'  Real split  :  {getattr(args, "eval_real_split", "train")}')
    print(f'  Data range  :  {getattr(args, "data_range", "minus1_1")}')
    print(f'{"═"*56}\n')


# ============================================================
# 5. 运行入口
# ============================================================

def run_fedavg(args):
    from src.federator import main as fedavg_main
    fedavg_main(args)


def run_lora(args):
    from src.lora_federator import main as lora_main
    lora_main(args)


def main():
    # 从配置文件构建基础参数
    args = build_args_from_config()
    # CLI 覆盖
    args = apply_cli_overrides(args)
    # 打印摘要
    print_config_summary(args)
    # 分发执行
    if args.method == 'fedavg':
        run_fedavg(args)
    else:
        run_lora(args)


if __name__ == '__main__':
    main()
