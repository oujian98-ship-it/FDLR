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




DATASET_CONFIGS = {
    'fmnist': {
        'image_size': 28,
        'num_channels': 1,
        'num_classes': 10,
        'conditional': 1,           # FMNIST 必须用条件生成
        'default_model': 'model_fmnist.pth',
        'hf_model_id': 'fusing/ddpm-fashion-mnist', # 高质量 FMNIST 底座
        'data_root': r'D:\data\fashion-mnist-master',
        'eval_num_samples': 5000,
        'eval_batch_size': 256,
    },
    'cifar10': {
        'image_size': 32,
        'num_channels': 3,
        'num_classes': 10,
        'conditional': 0,           # FedPhD CIFAR10 协议默认无条件生成
        'default_model': 'model_cifar.pth',
        'hf_model_id': 'google/ddpm-cifar10-32',    # 官方高质量 CIFAR10 底座
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
        'conditional': 0,           
        'default_model': 'model_celeba.pth',
        'hf_model_id': 'dimitrisp/ddpm-celeba-64',  
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



# ============================================================
# 3. 预设配置 (Presets) — 可用 --preset 快速切换
# ============================================================

PRESETS = {
    # ---- CelebA · LoRA ----
    'celeba_lora_homo': {
        'description': 'CelebA 同构 LoRA 正式实验 (R=30, E=5, B=64, rank=8)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '', 'global_lora_rank': 8,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1, 'auto_eval': 1,
    },
    'celeba_lora_hetero': {
        'description': 'CelebA 异构 LoRA 正式实验 (R=30, E=5, B=64, ranks=4,8,16,8,4)',
        'method': 'lora', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 64,
        'train': 1, 'auto_eval': 1,
    },
    # ---- CelebA · FedPhD 协议 ----
    'celeba_fedphd_homo': {
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
        'train': 1, 'auto_eval': 1,
    },
    'celeba_fedphd_hetero': {
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
        'train': 1, 'auto_eval': 1,
    },
    # ---- CelebA · FedAvg Baseline ----
    'celeba_fedavg': {
        'description': 'CelebA FedAvg Baseline 论文对齐 (R=30, K=5, E=5, B=64)',
        'method': 'fedavg', 'dataset': 'celeba',
        'rounds': 30, 'num_users': 5, 'local_ep': 5, 'local_bs': 64,
        'lr': 1e-4,
        'train_mode': 'full',
        'train': 1,
    },
    # ---- FMNIST · LoRA ----
    'fmnist_lora_homo': {
        'description': 'Fashion-MNIST 同构 LoRA 正式实验 (R=15, K=5, E=5, rank=32)',
        'method': 'lora', 'dataset': 'fmnist',
        'load_model': 'model_fmnist.pth',
        'rounds': 15, 'num_users': 5, 'local_ep': 5, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 32, 'lora_ranks': '', 'global_lora_rank': 32,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 256,
        'conditional': 1, 'num_classes': 10,
        'train': 1, 'auto_eval': 1,
    },
    'fmnist_lora_hetero': {
        'description': 'Fashion-MNIST 异构 LoRA 正式实验 (R=15, K=5, E=5, ranks=4,8,16,8,4)',
        'method': 'lora', 'dataset': 'fmnist',
        'load_model': 'model_fmnist.pth',
        'rounds': 15, 'num_users': 5, 'local_ep': 5, 'local_bs': 128,
        'lr': 1e-4,
        'lora_rank': 8, 'lora_ranks': '4,8,16,8,4', 'global_lora_rank': 16,
        'lora_alpha': -1.0, 'rank_beta': 0.5,
        'eval_num_samples': 5000, 'eval_batch_size': 256,
        'conditional': 1, 'num_classes': 10,
        'train': 1, 'auto_eval': 1,
    },
    # ---- FMNIST · FedAvg Baseline ----
    'fmnist_fedavg': {
        'description': 'Fashion-MNIST FedAvg Baseline 论文对齐 (R=15, K=5, E=5, B=128)',
        'method': 'fedavg', 'dataset': 'fmnist',
        'load_model': 'model_fmnist.pth',
        'rounds': 15, 'num_users': 5, 'local_ep': 5, 'local_bs': 128,
        'lr': 1e-4,
        'train_mode': 'full',
        'train': 1,
    },
    # ---- CIFAR10 · FedPhD 协议 ----
    'cifar10_fedphd': {
        'description': 'CIFAR10 FedPhD baseline-equivalent 协议 (R=2000, K=20, E=5)',
        'method': 'lora', 'dataset': 'cifar10',
        'rounds': 2000, 'num_users': 20, 'frac': 0.2, 'local_ep': 5, 'local_bs': 128,
        'iid': 0, 'unequal': 0, 'partition': 'fedphd-cifar2',
        'time_steps': 100, 'conditional': 0,
        'model_dim': 128, 'dim_mults': '1,2,2,2',
        'lora_ranks': '4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8',
        'global_lora_rank': 16, 'lora_alpha_mode': 'rank',
        'use_ddim': 1, 'ddim_steps': 100, 'eval_num_samples': 30000, 'eval_batch_size': 256,
        'eval_real_split': 'train', 'central_agg_interval': 5, 'seed': 2023,
        'train': 1, 'auto_eval': 1,
    },
    'cifar10_hf_fedphd': {
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
        'eval_real_split': 'train', 'central_agg_interval': 5, 'seed': 2023,
        'train': 1, 'auto_eval': 1,
    },
    # ---- 推理评估 (train=0) ----
    'infer_celeba': {
        'description': 'CelebA 推理: 生成图片 + FID评估',
        'method': 'lora', 'dataset': 'celeba',
        'train': 0,
        'load_model': 'flora_model_celeba_R[30]_K[5]_E[5].pth',
        'export_samples': 5000, 'export_dataset': 5000,
    },
    'infer_fmnist': {
        'description': 'FMNIST 推理: 生成图片 + FID评估',
        'method': 'lora', 'dataset': 'fmnist',
        'train': 0,
        'load_model': 'flora_model_fmnist_R[15]_K[5]_E[5].pth',
        'export_samples': 5000, 'export_dataset': 5000,
    },
    'infer_cifar10': {
        'description': 'CIFAR10 推理: 生成图片 + FID评估',
        'method': 'lora', 'dataset': 'cifar10',
        'train': 0,
        'load_model': 'flora_model_cifar10_R[2000]_K[20]_E[5].pth',
        'export_samples': 30000, 'export_dataset': 30000,
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







def parse_args():
    """
    解析所有命令行参数，所有超参均有默认值，无需修改源文件。
    优先级：CLI显式参数 > --preset预设 > dataset自动默认 > 全局默认

    常用示例:
      python main.py --method fedavg --dataset fmnist --num_users 5 --rounds 15
      python main.py --preset fmnist_lora_quick_hetero
      python main.py --preset fmnist_lora_quick_hetero --num_users 10  # preset + 单参数覆盖
    """
    # ---- 第一步: 预读 --preset / --dataset，确定基础默认值 ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--preset', type=str, default='')
    pre.add_argument('--dataset', type=str, default='fmnist')
    pre_args, _ = pre.parse_known_args()

    dataset_name = pre_args.dataset
    ds_cfg = get_dataset_config(dataset_name)

    # preset 值作为第二层默认（CLI 仍可覆盖）
    pv = {}  # preset values
    if pre_args.preset:
        pv = apply_preset(pre_args.preset)
        if 'dataset' in pv:
            dataset_name = pv['dataset']
            ds_cfg = get_dataset_config(dataset_name)

    def d(key, fallback):
        """preset优先，否则用fallback"""
        return pv.get(key, fallback)

    # ---- 第二步: 注册所有参数（含默认值）----
    parser = argparse.ArgumentParser(
        description='FedDiffuse - 联邦扩散模型训练框架',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 核心
    parser.add_argument('--method',   type=str, default=d('method', 'fedavg'),
                        choices=['fedavg', 'lora'], help='训练方法: fedavg | lora')
    parser.add_argument('--preset',   type=str, default='', help='快捷预设名 (见PRESETS)')
    parser.add_argument('--dataset',  type=str, default=d('dataset', 'fmnist'),
                        choices=list(DATASET_CONFIGS.keys()), help='数据集')

    # 数据
    parser.add_argument('--data_root',        type=str, default=d('data_root', ds_cfg.get('data_root', '')), help='数据路径（留空自动匹配）')
    parser.add_argument('--download_dataset', type=int, default=d('download_dataset', 0), help='自动下载=1')
    parser.add_argument('--partition',        type=str, default=d('partition', ''), help='分区规则 fedphd-cifar2|fedphd-celeba4')
    parser.add_argument('--seed',             type=int, default=d('seed', 42), help='随机种子')
    parser.add_argument('--image_size',       type=int, default=d('image_size', ds_cfg['image_size']))
    parser.add_argument('--num_channels',     type=int, default=d('num_channels', ds_cfg['num_channels']))
    parser.add_argument('--num_classes',      type=int, default=d('num_classes', ds_cfg['num_classes']))

    # 联邦学习超参
    parser.add_argument('--rounds',    type=int,   default=d('rounds', 15),    help='全局轮数 R')
    parser.add_argument('--num_users', type=int,   default=d('num_users', 5),  help='客户端数 K')
    parser.add_argument('--frac',      type=float, default=d('frac', 1.0),     help='参与比例 C')
    parser.add_argument('--local_ep',  type=int,   default=d('local_ep', 5),   help='本地轮次 E')
    parser.add_argument('--local_bs',  type=int,   default=d('local_bs', 128), help='本地 batch size B')
    parser.add_argument('--iid',       type=int,   default=d('iid', 1),        help='IID=1 / Non-IID=0')
    parser.add_argument('--unequal',   type=int,   default=d('unequal', 0),    help='非均匀分布=1')

    # 扩散模型
    parser.add_argument('--train',        type=int,   default=d('train', 1),         help='训练=1 推理=0')
    parser.add_argument('--load_model',   type=str,   default=d('load_model', ds_cfg.get('default_model', '')), help='预加载模型文件名')
    parser.add_argument('--resume_model', type=str,   default=d('resume_model', ''), help='断点续训模型路径')
    parser.add_argument('--start_round',  type=int,   default=d('start_round', 0),   help='续训起始轮次')
    parser.add_argument('--cache_clients',type=int,   default=d('cache_clients', 0))
    parser.add_argument('--time_steps',   type=int,   default=d('time_steps', 1000), help='扩散步数 T')
    parser.add_argument('--conditional',  type=int,   default=d('conditional', ds_cfg.get('conditional', 0)), help='条件生成=1')
    parser.add_argument('--lr',           type=float, default=d('lr', 1e-4),         help='学习率（扩散模型建议<=1e-4）')
    parser.add_argument('--optimizer',    type=str,   default=d('optimizer', 'sgd'), choices=['sgd', 'adam'])
    parser.add_argument('--model_dim',    type=int,   default=d('model_dim', 0),     help='U-Net base channels (0=image_size)')
    parser.add_argument('--dim_mults',    type=str,   default=d('dim_mults', '1,2,4'))
    parser.add_argument('--model_backend',type=str,   default=d('model_backend', 'custom'), choices=['custom', 'diffusers'])
    parser.add_argument('--hf_model_id',  type=str,   default=d('hf_model_id', ''))
    parser.add_argument('--disable_cudnn',type=int,   default=d('disable_cudnn', 0))

    # FedAvg 专用
    parser.add_argument('--train_mode',  type=str,   default=d('train_mode', 'full'),
                        choices=['full', 'usplit', 'udec', 'ulatdec'], help='FedAvg训练模式')
    parser.add_argument('--momentum',    type=float, default=d('momentum', 0.5))
    parser.add_argument('--round_offset',type=int,   default=d('round_offset', 0))

    # LoRA 专用
    parser.add_argument('--lora_rank',        type=int,   default=d('lora_rank', 32),       help='客户端 LoRA rank')
    parser.add_argument('--lora_ranks',       type=str,   default=d('lora_ranks', ''),       help='异构ranks 如 "4,8,16,8,4"')
    parser.add_argument('--global_lora_rank', type=int,   default=d('global_lora_rank', 32), help='服务端全局rank（>=最大客户端rank）')
    parser.add_argument('--lora_alpha',       type=float, default=d('lora_alpha', -1.0),     help='<=0表示alpha=rank')
    parser.add_argument('--lora_alpha_mode',  type=str,   default=d('lora_alpha_mode', 'rank'), choices=['rank', 'fixed'])
    parser.add_argument('--lora_dropout',     type=float, default=d('lora_dropout', 0.1))
    parser.add_argument('--rank_beta',        type=float, default=d('rank_beta', 0.5),       help='rank校正系数')
    parser.add_argument('--rank_correction',  type=int,   default=d('rank_correction', 1))
    parser.add_argument('--use_procrustes',   type=int,   default=d('use_procrustes', 1))
    parser.add_argument('--use_prefix_init',  type=int,   default=d('use_prefix_init', 1))
    parser.add_argument('--agg_mode',         type=str,   default=d('agg_mode', 'fdlr'),
                        choices=['fdlr', 'update_space', 'factor_avg', 'fedavg_lora'])

    # 导出 / 评估
    parser.add_argument('--export_samples',       type=int, default=d('export_samples', 5000))
    parser.add_argument('--export_dataset',       type=int, default=d('export_dataset', 5000))
    parser.add_argument('--show_samples',         type=int, default=d('show_samples', 0))
    parser.add_argument('--exp_rounds',           type=int, default=d('exp_rounds', 0))
    parser.add_argument('--eval_num_samples',     type=int, default=d('eval_num_samples', ds_cfg.get('eval_num_samples', 5000)))
    parser.add_argument('--eval_batch_size',      type=int, default=d('eval_batch_size', ds_cfg.get('eval_batch_size', 256)))
    parser.add_argument('--central_agg_interval', type=int, default=d('central_agg_interval', 5))
    parser.add_argument('--compute_is',           type=int, default=d('compute_is', 1),   help='评估时计算IS=1')
    parser.add_argument('--eval_real_split',      type=str, default=d('eval_real_split', 'train'), choices=['train', 'test'])
    parser.add_argument('--data_range',           type=str, default=d('data_range', 'minus1_1'))
    parser.add_argument('--auto_eval',            type=int, default=d('auto_eval', 0),    help='训练结束后自动评估FID/IS=1')

    # DDIM
    parser.add_argument('--use_ddim',   type=int, default=d('use_ddim', 1),   help='DDIM采样=1')
    parser.add_argument('--ddim_steps', type=int, default=d('ddim_steps', 100))

    args = parser.parse_args()

    # ---- 第三步: 若 load_model 默认值文件不存在，自动置空 ----
    if '--load_model' not in sys.argv and args.load_model:
        model_path = PROJECT_ROOT / args.load_model
        if not model_path.exists():
            print(f'[Warning] Default model not found: {model_path}，将从随机初始化开始')
            args.load_model = ''

    return args





def print_config_summary(args):
    """打印当前运行配置摘要"""
    method_tag = 'LoRA-FedDiffuse' if args.method == 'lora' else 'FedAvg-Baseline'
    mode_tag = 'TRAIN' if args.train == 1 else 'INFERENCE'
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
    from federator import main as fedavg_main
    fedavg_main(args)


def run_lora(args):
    from lora_federator import main as lora_main
    lora_main(args)


def main():
    # 解析所有参数（含默认值，支持 --preset 快速切换）
    args = parse_args()
    # 打印摘要
    print_config_summary(args)
    # 分发执行
    if args.method == 'fedavg':
        run_fedavg(args)
    else:
        run_lora(args)


if __name__ == '__main__':
    main()
