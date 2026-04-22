# FedDiffuse 实验运行指南

本文档提供 FedDiffuse 项目的完整实验命令，包括不同数据集、训练方法、模型加载和评估方式。

---

## 目录

1. [快速开始](#1-快速开始)
2. [数据集选择](#2-数据集选择)
3. [训练方法](#3-训练方法)
4. [预设命令 (Presets)](#4-预设命令-presets)
5. [自定义参数详解](#5-自定义参数详解)
6. [推理与评估](#6-推理与评估)
7. [LoRA 参数调优](#7-lora-参数调优)
8. [常见问题](#8-常见问题)

---

## 1. 快速开始

### 环境激活
```powershell
conda activate dfg_env
```

### 基本运行格式
```powershell
python main.py --method <方法> --dataset <数据集> [其他参数...]
```

---

## 2. 数据集选择

### 支持的数据集

| 数据集 | 图像尺寸 | 通道数 | 类别数 | 条件生成 | 预训练模型 |
|--------|---------|--------|--------|---------|-----------|
| **fmnist** | 28×28 | 1 (灰度) | 10 | ✅ 需要 (`conditional=1`) | `model_fmnist.pth` |
| **celeba** | 64×64 | 3 (RGB) | 16 | ❌ 不需要 (`conditional=0`) | `model_celeba.pth` |

### 切换数据集示例

```powershell
# FMNIST 数据集
python main.py --method lora --dataset fmnist

# CelebA 数据集
python main.py --method lora --dataset celeba
```

> **注意**: 切换数据集时，`image_size`、`num_channels`、`num_classes`、`conditional` 会自动匹配。

### 自定义数据路径
```powershell
python main.py --dataset fmnist --data_root "D:\data\fashion-mnist-master"
python main.py --dataset celeba --data_root "D:\data\CelebA"
```

---

## 3. 训练方法

### 3.1 LoRA 联邦微调 (推荐)
只训练 ~5-20% 的参数（低秩适配器），通信效率高。

```powershell
# 基本命令 - FMNIST
python main.py --method lora --dataset fmnist --train 1 --rounds 30

# 基本命令 - CelebA
python main.py --method lora --dataset celeba --train 1 --rounds 30
```

**LoRA 训练流程：**
```
预训练基础模型 → 注入 LoRA 适配器 → 联邦学习微调 → 保存 flora_model_*.pth
```

### 3.2 FedAvg 全参数训练
训练 100% 参数，原论文的方法。

```powershell
# FMNIST FedAvg
python main.py --method fedavg --dataset fmnist --train 1 --rounds 30

# CelebA FedAvg
python main.py --method fedavg --dataset celeba --train 1 --rounds 30
```

**FedAvg 特有参数：**
```powershell
--train_mode full    # 完整训练模式 (full / usplit / udec / ulatdec)
--round_offset 0     # 轮次编号偏移 (断点续训用)
```

---

## 4. 预设命令 (Presets)

使用 `--preset` 快速切换实验配置：

### 4.1 LoRA 实验

```powershell
# CelebA LoRA 快速实验 (R=10, ~2.5小时)
python main.py --preset celeba_lora_quick

# CelebA LoRA 对齐论文 (R=30, E=5, ~7.5小时)
python main.py --preset celeba_lora_paper_align

# CelebA FedAvg Baseline (R=30, ~37小时)
python main.py --preset celeba_fedavg_baseline
```

### 4.2 推理/评估

```powershell
# FMNIST 模型推理: 生成图片 + FID 评估
python main.py --preset infer_fmnist

# CelebA 模型推理: 生成图片 + FID 评估
python main.py --preset infer_celeba
```

### 所有可用预设列表

| Preset 名称 | 说明 |
|------------|------|
| `celeba_lora_quick` | CelebA LoRA 快速实验 (R=10) |
| `celeba_lora_paper_align` | CelebA LoRA 对齐论文 (R=30, E=5) |
| `celeba_fedavg_baseline` | CelebA FedAvg 原论文对齐 |
| `infer_fmnist` | FMNIST 推理评估 |
| `infer_celeba` | CelebA 推理评估 |

---

## 5. 自定义参数详解

### 5.1 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | `lora` | 训练方法: `lora` 或 `fedavg` |
| `--dataset` | str | `fmnist` | 数据集: `fmnist` 或 `celeba` |
| `--train` | int | `1` | 1=训练, 0=仅推理 |
| `--load_model` | str | `` | 加载的模型文件名 |

### 5.2 联邦学习参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rounds` | int | `30` | 全局通信轮数 (R) |
| `--num_users` | int | `5` | 客户端数量 (K) |
| `--frac` | float | `1.0` | 每轮参与比例 |
| `--local_ep` | int | `5` | 每客户端本地训练 epoch 数 (E) |
| `--local_bs` | int | `64` | 本地批次大小 (B) |
| `--iid` | int | `1` | 1=IID分布, 0=Non-IID分布 |
| `--unequal` | int | `0` | 0=等量数据, 1=不等量分配 |

### 5.3 扩散模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--time_steps` | float | `1000` | DDPM 扩散步数 (T) |
| `--conditional` | int | `-1` | 条件生成 (-1=自动匹配) |
| `--lr` | float | `1e-4` | 学习率 (**不要超过 1e-4**) |
| `--optimizer` | str | `adam` | 优化器 |

### 5.4 LoRA 专用参数 (仅 method=lora 时生效)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lora_rank` | int | `8` | 客户端 LoRA rank (r) |
| `--global_lora_rank` | int | `16` | 服务端全局 rank (g) |
| `--lora_ranks` | str | `` | 各客户端逗号分隔的 rank，如 `"4,8,16,8,4"` |
| `--lora_alpha` | float | `1.0` | LoRA 缩放因子 α |
| `--lora_dropout` | float | `0.0` | LoRA dropout |

**LoRA 可训练参数占比参考：**

| Rank 配置 | 占比 | 适用场景 |
|----------|------|---------|
| r=8 / g=16 | ~5% | 极致省通信 |
| r=16 / g=32 | ~10% | 平衡选择 |
| r=32 / g=64 | ~20% | **推荐** |
| r=64 / g=128 | ~40% | 追求效果 |

### 5.5 导出/评估参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--export_samples` | int | `0` | 生成图片数量 (0=不导出) |
| `--export_dataset` | int | `0` | 导出真实样本数 (FID参照) |
| `--show_samples` | int | `0` | 是否展示 GUI (0=仅保存文件) |
| `--exp_rounds` | int | `0` | 中间轮次导出间隔 |

---

## 6. 推理与评估

### 6.1 基础评估命令

```powershell
# 生成 5000 张图片并计算 FID (推荐，与论文一致)
python main.py --method lora --dataset fmnist \
  --load_model "flora_model_fmnist_R[30]_K[5]_E[5].pth" \
  --train 0 \
  --export_samples 5000 --export_dataset 5000
```

### 6.2 评估不同的模型文件

#### LoRA 微调后的模型
```powershell
python main.py --method lora --dataset fmnist \
  --load_model "flora_model_fmnist_R[30]_K[5]_E[5].pth" \
  --train 0 --conditional 1 \
  --export_samples 5000 --export_dataset 5000
```

#### 原项目预训练模型 (条件版)
```powershell
python main.py --method lora --dataset fmnist \
  --load_model "model_fmnist.pth" \
  --train 0 --conditional 1 \
  --export_samples 5000 --export_dataset 5000
```

#### 原项目预训练模型 (无条件版)
```powershell
python main.py --method lora --dataset fmnist \
  --load_model "model_fmnist_unconditional.pth" \
  --train 0 --conditional 0 \    # 无条件必须设为0!
  --export_samples 5000 --export_dataset 5000
```

#### CelebA 模型
```powershell
python main.py --method lora --dataset celeba \
  --load_model "flora_model_celeba_R[10]_K[5]_E[3].pth" \
  --train 0 --conditional 0 \
  --export_samples 5000 --export_dataset 5000
```

### 6.3 评估结果输出

运行后会自动：
1. **生成图片**: 保存到 `exports/` 目录
2. **计算 FID**: 输出到终端
3. **保存日志**: 保存到 `results/eval_logs/` 目录

日志命名规则：
```
lora_<dataset>_R[<R>]_K[<K>]_r[<r>g<g>]_E[<E>]_B[<B>]_T[<T>]_I[<iid>,<unequal>]_<时间戳>.log
```

示例：
```
results/eval_logs/lora_fmnist_R[30]_K[5]_r[32g32]_E[5]_B[64]_T[1000]_I[1,0]_20260422-092200.log
```

---

## 7. LoRA 参数调优

### 7.1 在已有模型上继续训练

```powershell
python main.py --method lora --dataset fmnist \
  --load_model "flora_model_fmnist_R[30]_K[5]_E[5].pth" \
  --train 1 --rounds 10 \
  --lora_rank 32 --global_lora_rank 64   # 可增大 rank
```

> **注意**: 代码会自动检测是否为 LoRA checkpoint 并正确处理。

### 7.2 异构客户端 Rank 配置

让不同客户端使用不同的 LoRA rank（异构联邦学习）：

```powershell
python main.py --method lora --dataset fmnist \
  --train 1 --rounds 30 \
  --lora_ranks "4,8,16,8,4"
```
这表示 5 个客户端分别使用 rank = [4, 8, 16, 8, 4]。

### 7.3 Non-IID 数据分布

```powershell
# Non-IID 分布 (Dirichlet α=0.1)
python main.py --method lora --dataset fmnist --iid 0 --train 1 --rounds 30

# Non-IID + 不等量数据
python main.py --method lora --dataset fmnist --iid 0 --unequal 1 --train 1 --rounds 30
```

---

## 8. 常见问题

### Q1: 为什么加载的是 model_fmnist.pth 而不是我的 flora 模型？

**原因**: `build_args_from_config()` 中 `auto_select_model()` 会覆盖 preset 设置。

**解决方案**: 已修复此 bug。如果仍有问题，确保在 CLI 中显式指定 `--load_model`：

```powershell
python main.py --preset infer_fmnist --load_model "你的模型.pth"
```

### Q2: FID 结果波动大怎么办？

- 使用 **5000 张** 图片评测（而非 1000 张）
- 多次取平均或固定随机种子
- 确保 `--export_samples` 和 `--export_dataset` 数量一致

### Q3: Conditional vs Unconditional 如何选择？

| 数据集 | conditional | 原因 |
|--------|-------------|------|
| fmnist | **1** | 有 10 个类别，需要标签引导 |
| celeba | **0** | 人脸生成不需要类别条件 |

**错误设置会导致 FID 极差！**

### Q4: 终端卡住不动？

`--show_samples 1` 会打开 GUI 窗口阻塞程序。使用 `--show_samples 0` 并查看 `exports/` 目录中的图片。

### Q5: 如何查看已保存的结果？

```powershell
# 查看所有评估日志
ls results/eval_logs/

# 查看生成的图片
ls exports/
```

### Q6: 不同模型的对比汇总

| 模型文件 | 类型 | 条件 | 典型 FID |
|---------|------|------|---------|
| `model_fmnist.pth` | UnetConditional (预训练) | conditional=1 | ~100+ (模糊) |
| `model_fmnist_unconditional.pth` | Unet (预训练) | conditional=0 | ~100+ (模糊) |
| `flora_model_fmnist_R[*].pth` | LoRA 微调后 | conditional=1 | ~58-92 (取决于训练) |
| `model_celeba.pth` | Unet (预训练) | conditional=0 | ~47 (较好) |
| `flora_model_celeba_R[*].pth` | LoRA 微调后 | conditional=0 | ~35-47 |

---

## 附录: 常用完整命令速查

```powershell
# ===== FMNIST LoRA 完整实验 =====
python main.py --method lora --dataset fmnist --train 1 --rounds 30 \
  --num_users 5 --local_ep 5 --local_bs 64 --lr 1e-4 \
  --lora_rank 32 --global_lora_rank 32 --iid 1

# ===== FMNIST LoRA 评估 =====
python main.py --method lora --dataset fmnist --train 0 \
  --load_model "flora_model_fmnist_R[30]_K[5]_E[5].pth" \
  --export_samples 5000 --export_dataset 5000

# ===== FMNIST FedAvg 完整实验 =====
python main.py --method fedavg --dataset fmnist --train 1 --rounds 30 \
  --num_users 5 --local_ep 5 --local_bs 64 --lr 1e-4 --iid 1

# ===== CelebA LoRA 快速实验 =====
python main.py --method lora --dataset celeba --train 1 --rounds 10 \
  --num_users 5 --local_ep 3 --local_bs 128 --lr 1e-4

# ===== CelebA LoRA 评估 =====
python main.py --method lora --dataset celeba --train 0 \
  --load_model "flora_model_celeba_R[10]_K[5]_E[3].pth" \
  --export_samples 5000 --export_dataset 5000

# ===== 异构 LoRA (不同客户端不同 rank) =====
python main.py --method lora --dataset fmnist --train 1 --rounds 30 \
  --lora_ranks "4,8,16,8,4"

# ===== Non-IID 实验 =====
python main.py --method lora --dataset fmnist --train 1 --rounds 30 --iid 0
```

---

*文档更新时间: 2026-04-22*
