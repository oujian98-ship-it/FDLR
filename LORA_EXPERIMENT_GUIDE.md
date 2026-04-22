# 异构 LoRA 联邦微调扩散模型 — 完整使用指南

> **本文档说明如何使用新增的 LoRA 联邦训练功能。所有代码均为**全新文件**，不修改任何原有文件。

---

## 一、项目结构概览（新增文件）

```
FedDiffuse-main/
├── src/
│   ├── lora.py                    # 【新】LoRA 层定义 + 自动注入 U-Net
│   ├── hetero_aggregation.py      # 【新】异构聚合引擎 (ΔW恢复/加权/RSVD/Procrustes)
│   ├── lora_client.py             # 【新】LoRA 联邦客户端
│   ├── lora_federator.py          # 【新】⭐ 主入口（运行这个）
│   │
│   ├── federator.py               # 【未修改】原 FedAvg 入口
│   ├── client.py                  # 【未修改】原全参数训练客户端
│   ├── aggregation.py             # 【未修改】原 FedAvg 聚合
│   ├── unet.py                    # 【未修改】U-Net 模型
│   ├── diffuser.py                # 【未修改】扩散过程
│   └── utils.py                   # 【未修改】数据加载 / FID 等
│
├── LORA_EXPERIMENT_GUIDE.md       # 本文档
```

---

## 二、核心方法概述

### 2.1 解决的问题

在联邦学习场景下，多个客户端共同微调同一个扩散模型，但每个客户端的算力不同：

| 客户端 | 算力水平 | 本地 LoRA Rank |
|--------|---------|---------------|
| 客户端 A | 高（如 RTX 4090） | rank=16 |
| 客户端 B | 中（如 RTX 3090） | rank=8 |
| 客户端 C | 低（如 RTX 3060） | rank=4 |

传统 FedAvg 要求所有客户端训练完整模型，计算和通信代价极高。
本方案让每个客户端只训练适合自己能力的低秩 LoRA 参数，服务器通过专门的聚合算法合并异构更新。

### 2.2 算法流程（每轮）

```
服务器 → 分发 LoRA 因子前缀给各客户端
    ↓
各客户端 → 在本地数据上只训练 LoRA (B, A) 因子对
    ↓
客户端 → 上传 (B, A) 给服务器
    ↓
服务器执行:
  ① 恢复真实更新: ΔW_i = B_i @ A_i        (每个客户端, 每层)
  ② 计算聚合权重: α_i = n_i / √r_i         (数据量 × rank校正)
  ③ 更新空间加权平均: ΔW̄ = Σ α_i · ΔW_i
  ④ RSVD 低秩重投影: ΔW̄ → (B_global, A_global)
  ⑤ Procrustes 对齐: 与上一轮因子对齐, 抑制漂移
  ⑥ 前缀分发: 按客户端 rank 切片后发回
```

---

## 三、快速开始

### 3.1 前置条件

```bash
# 1. 进入项目根目录
cd d:\projects\FedDiffuse-main

# 2. 激活 conda 环境 (已有 dfg_env)
conda activate dfg_env

# 3. 确认依赖已安装 (einops, torch 等)
pip list | findstr -i "torch einops"
```

### 3.2 最简训练命令 — CelebA 人脸生成

**同构模式（所有客户端相同 rank=8）：**

```bash
cd src
python lora_federator.py ^
  --dataset=celeba ^
  --train=1 ^
  --lora_rank=8 --global_lora_rank=16 ^
  --num_channels=3 --image_size=64 ^
  --load_model=model_celeba.pth ^
  --rounds=10 --num_users=5 --local_ep=3 --local_bs=64 ^
  --lr=1e-4 ^
  --export_samples=0 --show_samples=0
```

**异构模式（不同客户端不同 rank）：**

```bash
python lora_federator.py ^
  --dataset=celeba ^
  --train=1 ^
  --lora_ranks="4,8,16,8,4" --global_lora_rank=16 ^
  --num_channels=3 --image_size=64 ^
  --load_model=model_celeba.pth ^
  --rounds=15 --num_users=5 --local_ep=3 --local_bs=64 ^
  --export_samples=0
```

### 3.3 推理/采样命令

```bash
python lora_federator.py ^
  --dataset=celeba ^
  --train=0 ^
  --lora_rank=8 --global_lora_rank=16 ^
  --num_channels=3 --image_size=64 ^
  --load_model=models/lora_celeba_R[10]_K[5]_r[8]g[16].pth ^
  --export_samples=16 --show_samples=0
```

---

## 四、参数详解

### 4.1 新增 LoRA 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `--lora_rank` | int | **8** | 所有客户端的基础 LoRA rank |
| `--lora_ranks` | str | **""** | **异构配置**: 逗号分隔的 per-client rank，如 `"4,8,16,8,4"` 表示5个客户端分别使用 rank 4/8/16/8/4 |
| `--global_lora_rank` | int | **16** | 服务器端全局 LoRA rank（RSVD 重投影目标秩）。应 ≥ 所有客户端的最大 rank |
| `--lora_alpha` | float | **1.0** | LoRA 缩放系数 (scaling = alpha/rank)。通常设为 1.0 或 2*rank |
| `--lora_dropout` | float | **0.0** | LoRA 路径上的 dropout 概率（正则化用，通常不需要改） |

### 4.2 复用原项目的参数（含义不变）

| 参数 | 说明 | 推荐值 (CelebA) | 推荐值 (FMNIST) |
|------|------|-----------------|-----------------|
| `--dataset` | 数据集 | `celeba` | `fmnist` |
| `--image_size` | 图像分辨率 | `64` | `28` |
| `--num_channels` | 通道数 (RGB=3, 灰度=1) | `3` | `1` |
| `--load_model` | 预训练基座模型路径 | `model_celeba.pth` | `model_fmnist.pth` |
| `--rounds` | 联邦训练轮数 | `10~20` | `10~15` |
| `--num_users` | 客户端数量 K | `5` | `5~10` |
| `--frac` | 每轮参与比例 C | `1.0` | `1.0` |
| `--local_ep` | 每轮本地训练 epoch E | `3` | `3` |
| `--local_bs` | 本地 batch size B | `64` | `128` |
| `--lr` | 学习率 | `1e-4` | `1e-4` |
| `--time_steps` | 扩散步数 T | `1000` | `1000` |
| `--conditional` | 是否类别条件生成 | `0` | `0` 或 `1` |
| `--iid` | 是否 IID 数据分布 | `1` | `1` |
| `--num_classes` | 类别数 (条件生成时) | `16` (CelebA) | `10` |

### 4.3 输出/采样参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--export_samples` | `0` | 训练结束后生成的样本数量（0=不生成） |
| `--show_samples` | `0` | 是否弹出 matplotlib 窗口显示样本（Windows 终端建议设为 0） |
| `--exp_rounds` | `0` | 是否在中间轮次导出样本（0=不导出，>0 则从第 5 轮开始每隔奇数轮导出） |
| `--export_dataset` | `0` | 导出真实数据集图像（用于 FID 计算） |

---

## 五、实验场景示例

### 场景 1：验证 LoRA 能工作（同构 + 少量轮次）

目的：确认 LoRA 注入正确、训练收敛、生成质量不退化。

```bash
# FMNIST (快速验证，几分钟跑完)
python lora_federator.py \
  --dataset=fmnist --train=1 \
  --lora_rank=4 --global_lora_rank=8 \
  --image_size=28 --num_channels=1 \
  --rounds=5 --num_users=3 --local_ep=2 --local_bs=128 \
  --lr=1e-4 \
  --export_samples=16 --show_samples=0
```

### 场景 2：CelebA 同构 LoRA 对比实验

对比 FedAvg 全量微调 vs LoRA 微调的效果差异：

```bash
# 基线：原版 FedAvg (全参数训练)
python federator.py --dataset=celeba --train=1 --num_channels=3 --image_size=64 \
  --load_model=model_celeba.pth --rounds=10 --num_users=5 --local_ep=3 --local_bs=64

# 实验：LoRA 微调 (rank=8)
python lora_federator.py --dataset=celeba --train=1 \
  --lora_rank=8 --global_lora_rank=16 \
  --num_channels=3 --image_size=64 --load_model=model_celeba.pth \
  --rounds=10 --num_users=5 --local_ep=3 --local_bs=64
```

### 场景 3：异构 LoRA 核心实验

这是论文的核心贡献——证明异构 rank 客户端的聚合效果：

```bash
python lora_federator.py --dataset=celeba --train=1 \
  --lora_ranks="4,8,16,8,4" --global_lora_rank=16 \
  --num_channels=3 --image_size=64 --load_model=model_celeba.pth \
  --rounds=15 --num_users=5 --local_ep=3 --local_bs=64 \
  --export_samples=50 --show_samples=0 --exp_rounds=20
```

### 场景 4：Non-IID 数据分布实验

测试方法在 Non-IID 数据下的鲁棒性：

```bash
python lora_federator.py --dataset=celeba --train=1 \
  --lora_ranks="4,8,12,8,4" --global_lora_rank=16 \
  --num_channels=3 --image_size=64 --load_model=model_celeba.pth \
  --rounds=20 --num_users=5 --local_ep=3 --local_bs=64 \
  --iid=0 --unequal=0 \
  --export_samples=100 --exp_rounds=30
```

---

## 六、输出文件说明

训练完成后会生成以下文件：

```
FedDiffuse-main/
├── results/
│   └── 20260416-203000_lora_celeba_R[10]_K[5]_r[8]g[16].../
│       ├── lora_..._R[0].pth          # 第 0 轮 checkpoint
│       ├── lora_..._R[5].pth          # 第 5 轮 checkpoint  
│       ├── lora_..._R[9].pth          # 最终 checkpoint
│       └── lora_....csv                # 训练 loss + 通信统计
│
├── flora_model_{dataset}.pth              # ⭐ 最终模型 (项目根目录，如 flora_model_celeba.pth)
│
├── exports/
│   ├── lora_.../                        # 最终采样结果
│   │   ├── image1000.png ~ image1016.png
│   └── lora_..._R[5]/                   # 中间轮次采样 (如果 exp_rounds>0)
│       └── ...
```

CSV 文件格式：
```
round,loss,upload_kb,download_kb
0,0.234512,45.2,38.7
1,0.198234,44.8,38.5
...
runtime,3600.5
```

---

## 七、显存参考表

基于 RTX 4090 D (24GB)，以 CelebA 64x64 为例：

| batch_size | 模型 | 预估显存占用 | 备注 |
|-----------|------|-------------|------|
| 32 | Unet(64)+LoRA(r=8) | ~6 GB | 安全 |
| 64 | Unet(64)+LoRA(r=8) | ~10 GB | 推荐 |
| 96 | Unet(64)+LoRA(r=8) | ~14 GB | 可行 |
| 128 | Unet(64)+LoRA(r=8) | ~18 GB | 注意OOM |
| 32 | Unet(64)+LoRA(r=16) | ~7 GB | 高 rank 略增 |

LoRA 只增加不到 1% 的参数量和少量显存开销。

---

## 八、评估指标

### 8.1 FID Score (Fréchet Inception Distance)

衡量生成图像与真实图像的分布距离，越低越好：

```bash
# 已有工具可直接使用
python calculate_fid_scores.py --path_real=data/celeba \
  --path_fake=exports/lora_.../
```

| FID 范围 | 含义 |
|----------|------|
| < 10 | 极好（接近真实分布） |
| 10 - 25 | 很好 |
| 25 - 50 | 一般 |
| > 50 | 需要改进 |

### 8.2 通信效率统计

本方案自动记录并输出到 CSV：

- **Upload bytes**: 所有客户端上传 LoRA 因子的总字节数
- **Download bytes**: 服务器下发因子的总字节数
- **压缩比**: vs FedAvg 全参数更新的压缩倍数

典型值（CelebA 64x64, rank=8, 5 clients）:
- FedAvg 全量: ~200 MB/轮
- LoRA (rank=8): ~40 KB/轮 (**~5000x 压缩**)

---

## 九、常见问题排查

### Q1: 报错 `ModuleNotFoundError: No module named 'lora'`
确保在 `src/` 目录下执行，或设置 PYTHONPATH：
```bash
cd src
set PYTHONPATH=%cd%;%PYTHONPATH%
python lora_federator.py ...
```

### Q2: 报错 CUDA out of memory
减小 `--local_bs`：
```bash
--local_bs=32   # 从 64 减到 32
```

### Q3: 生成的图像是噪声
检查以下参数是否正确：
- `--image_size=64` （CelebA 必须是 64）
- `--num_channels=3` （CelebA 是 RGB）
- `--load_model=model_celeba.pth` （确保加载了预训练模型）

### Q4: Loss 不下降
- 检查 `--lr`（推荐 1e-4）
- 增加 `--local_ep`
- 确认 `--load_model` 加载了预训练权重

### Q5: 如何只对 Attention 层插 LoRA？（当前默认行为）
代码默认只在 `Attention` 和 `LinearAttention` 的 Q/K/V/out 投影上注入 LoRA。
这是 P0 优先级的层，对生成质量影响最大。

如需扩展到 time_mlp 的 Linear 层，可修改 `target_layers='all'`（需改代码中的调用参数）。

---

## 十、算法与原论文的对应关系

| 你的 MD 文档中的步骤 | 代码实现位置 |
|--------------------|-------------|
| ① 异构 rank 客户端本地训练 ΔW = BA | `lora_client.py`: `LoRAClient.train_local()` |
| ② 计算 rank 校正聚合权重 α | `hetero_aggregation.py`: `compute_aggregation_weights()` |
| ③ 恢复真实更新并在更新空间聚合 | `hetero_aggregation.py`: `recover_true_updates()` + `aggregate_true_updates()` |
| ④ RSVD 低秩重投影 | `hetero_aggregation.py`: `reproject_to_low_rank()` / `randomized_svd()` |
| ⑤ Procrustes 对齐抑制分解漂移 | `hetero_aggregation.py`: `procrustes_alignment_per_layer()` |
| ⑥ 嵌套子空间前缀分发 | `hetero_aggregation.py`: `prefix_slice_distribution()` |
| LoRA 注入到 U-Net Attention | `lora.py`: `inject_lora_into_unet()` + `LoRAInjectedAttention()` |
| to_qkv 拆分为独立 Q/K/V | `lora.py`: `LoRAInjectedAttention.__init__()` |

---

## 十一、下一步建议

1. **先跑通 Phase 1**：用 FMNIST + 同构 LoRA (rank=4), 5轮训练, 确认整个流水线正常
2. **切换到 CelebA**：加载 model_celeba.pth, rank=8, 10轮训练
3. **开启异构**：设置 `--lora_ranks="4,8,16,8,4"`, 观察不同 rank 客户端的收敛情况
4. **对比实验**：记录 FID / loss 曲线 / 通信量, 与原版 FedAvg 基线对比
5. **消融实验**：依次关闭 Procrustes / RSVD / rank 校正, 分析各组件的贡献度
