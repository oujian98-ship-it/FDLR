#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import pickle
import sys
import re
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from fid_score import calculate_fid
from improved_precision_recall import calculate_precision_recall
from metrics_inception_score import calculate_inception_score


def create_training_config(rounds=10, num_users=5, frac=1, local_ep=3, local_bs=128, train_mode='full', train=1,
                           load_model='', time_steps=1000, conditional=0, lr=1e-4, momentum=0.5, optimizer='adam',
                           dataset='fmnist', image_size=28, num_channels=1, iid=1, unequal=0, export_samples=0,
                           export_dataset=0, show_samples=0, num_classes=10, exp_rounds=0, round_offset=0):
    return Namespace(rounds=rounds, num_users=num_users, frac=frac, local_ep=local_ep, local_bs=local_bs,
                     train_mode=train_mode, train=train, load_model=load_model, time_steps=time_steps,
                     conditional=conditional, lr=lr, momentum=momentum, optimizer=optimizer, dataset=dataset,
                     image_size=image_size, num_channels=num_channels, iid=iid, unequal=unequal,
                     export_samples=export_samples, export_dataset=export_dataset, show_samples=show_samples,
                     num_classes=num_classes, exp_rounds=exp_rounds, round_offset=round_offset)


def create_fid_config(batch_size=50, num_workers=0, device='cuda', dims=2048, save_stats=False, paths=('', '')):
    return Namespace(batch_size=batch_size, num_workers=num_workers, device=device, dims=dims, save_stats=save_stats,
                     paths=paths)


def create_precision_recall_config(path_real='', path_fake='', batch_size=50, k=3, num_samples=5000,
                                   fname_precalc='', toy=False):
    return Namespace(path_real=path_real, path_fake=path_fake, batch_size=batch_size, k=k, num_samples=num_samples,
                     fname_precalc=fname_precalc, toy=toy)


def log_timestamp():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def build_experiment_log_name(method, args, model_name=''):
    """Return method_dataset_R[x]_K[x]_E[x], preferring explicit tags in model paths."""
    dataset = getattr(args, 'dataset', '')
    train_mode = getattr(args, 'train_mode', 'full')
    source = ' '.join(str(x) for x in (
        model_name,
        getattr(args, 'load_model', ''),
        getattr(args, 'resume_model', ''),
    ) if x)

    def _find_tag(tag, default):
        match = re.search(rf'{tag}\[([^\]]+)\]', source)
        return match.group(1) if match else default

    rounds = _find_tag('R', getattr(args, 'rounds', ''))
    num_users = _find_tag('K', getattr(args, 'num_users', ''))
    local_ep = _find_tag('E', getattr(args, 'local_ep', ''))
    if method == 'fedavg':
        for mode in ('full', 'usplit', 'udec', 'ulatdec'):
            if re.search(rf'(^|_){mode}(_|$)', source):
                train_mode = mode
                break
        return f'{method}_{train_mode}_{dataset}_R[{rounds}]_K[{num_users}]_E[{local_ep}]'
    return f'{method}_{dataset}_R[{rounds}]_K[{num_users}]_E[{local_ep}]'


def experiment_log_dir(parent_path, log_kind, method, args, model_name=''):
    """Build logs/<log_kind>/<method_dataset_R[x]_K[x]_E[x]>."""
    folder = Path(parent_path) / 'logs' / log_kind / build_experiment_log_name(method, args, model_name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def perform_evaluation(real_path, fake_path, dataset_to_export=None, num_samples=100,
                       eval_log_dir=None, config_tag='', batch_size=256,
                       compute_is=True, compute_pr=True, experiment_meta=None):
    """Run FID + IS + Precision/Recall evaluation and optionally log results."""
    import datetime

    if dataset_to_export is not None:
        export_dataset(real_path, dataset_to_export, num_samples, train=False)

    fid_config = create_fid_config(paths=(real_path, fake_path), batch_size=batch_size)
    _fid_result = calculate_fid(fid_config)
    _pr_result = None
    if compute_pr:
        precision_recall_config = create_precision_recall_config(path_real=real_path, path_fake=fake_path,
                                                                 num_samples=num_samples,
                                                                 batch_size=batch_size)
        _pr_result = calculate_precision_recall(precision_recall_config)
    else:
        print('\nPrecision/Recall: skipped (compute_pr=False)')
    _is_result = None
    if compute_is:
        print('\nComputing Inception Score...')
        _is_result = calculate_inception_score(
            image_folder=fake_path,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        print(f'Inception Score: {_is_result[0]} +/- {_is_result[1]}')
    else:
        print('\nInception Score: skipped (compute_is=False)')

    print('\nEvaluation summary:')
    print(f'  FID       : {_fid_result}')
    if _is_result is not None:
        print(f'  IS        : {_is_result[0]} +/- {_is_result[1]}')
    else:
        print('  IS        : skipped')
    if _pr_result is not None:
        print(f'  Precision : {_pr_result[0]}')
        print(f'  Recall    : {_pr_result[1]}')
    else:
        print('  Precision : skipped')
        print('  Recall    : skipped')

    # ---- Save evaluation log ----
    if eval_log_dir:
        eval_log_dir = Path(eval_log_dir)
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        ts = log_timestamp()
        if config_tag and config_tag.endswith('.log'):
            log_name = config_tag
        else:
            log_name = f'{config_tag}_{ts}.log' if config_tag else f'eval_{ts}.log'
        log_path = eval_log_dir / log_name
        append_log = log_path.exists()
        with open(log_path, 'a' if append_log else 'w', encoding='utf-8') as f:
            if append_log:
                f.write('\n\n')
            f.write(f'Evaluation Log  {ts}\n')
            f.write(f'{"="*60}\n\n')
            f.write(f'Tag         : {config_tag}\n')
            if experiment_meta:
                for key, value in experiment_meta.items():
                    f.write(f'{key:<12}: {value}\n')
            f.write(f'Real path   : {real_path}\n')
            f.write(f'Fake path   : {fake_path}\n')
            f.write(f'Num samples : {num_samples}\n\n')
            f.write(f'Batch size  : {batch_size}\n\n')
            f.write(f'FID Score   : {_fid_result}\n')
            f.write(f'IS Score    : {_is_result}\n')
            f.write(f'Precision/Recall: {_pr_result if _pr_result is not None else "skipped"}\n')
        print(f'\n[Evaluation log saved] {log_path}')

    return {
        'fid': _fid_result,
        'inception_score': _is_result,
        'precision_recall': _pr_result,
    }


def record_training_time(parent_path, method, model_name, args, training_time_sec,
                         result_folder=None, model_path=None, write_summary_log=True,
                         timestamp=None, extra_stats=None):
    """Write a timestamped training-only runtime log for an experiment."""
    parent_path = Path(parent_path)
    file_ts = timestamp or log_timestamp()
    display_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    detail_path = None
    if write_summary_log:
        detail_dir = experiment_log_dir(parent_path, 'train_eval_logs', method, args, model_name)
        detail_path = detail_dir / f'{file_ts}.log'
        with open(detail_path, 'w', encoding='utf-8') as f:
            f.write(f'Training Log  {file_ts}\n')
            f.write('=' * 60 + '\n\n')
            f.write(f'timestamp: {display_ts}\n')
            f.write(f'method: {method}\n')
            f.write(f'model_name: {model_name}\n')
            f.write(f'training_time_sec: {float(training_time_sec):.6f}\n')
            f.write(f'training_time_min: {float(training_time_sec) / 60.0:.6f}\n')
            if args is not None:
                f.write('\n[Training config]\n')
                for key in sorted(vars(args).keys()):
                    value = getattr(args, key)
                    f.write(f'{key}: {value}\n')
            if extra_stats:
                f.write('\n[Final stats]\n')
                for key, value in extra_stats.items():
                    f.write(f'{key}: {value}\n')
            if result_folder is not None:
                f.write('\n[Artifacts]\n')
                f.write(f'result_folder: {result_folder}\n')
            if model_path is not None:
                f.write(f'model_path: {model_path}\n')

    if detail_path is not None:
        print(f'Training time recorded: {float(training_time_sec):.2f}s -> {detail_path}')
    return detail_path


class LabeledCelebA(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.labeled_dataset = create_labeled_celeba(base_dataset)

    def __getitem__(self, index):
        base_index = self.labeled_dataset[index][0]
        image, _ = self.base_dataset[base_index]
        label = self.labeled_dataset[index][1]
        return image, label

    def __len__(self):
        return len(self.labeled_dataset)

    def get_labels(self):
        return np.array([item[1] for item in self.labeled_dataset])


class DecoderTrainingTask:
    def __init__(self, mid):
        self.mid = mid


class SplitTrainingTask:
    def __init__(self, up, down, mid):
        self.up = up
        self.down = down
        self.mid = mid


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def transform_mnist(image):
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)])
    return transform(image.convert("L"))


def transform_cifar10_train(image):
    data_range = globals().get('_DATA_RANGE', 'minus1_1')
    steps = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if data_range != '0_1':
        steps.append(transforms.Lambda(lambda t: (t * 2) - 1))
    transform = transforms.Compose(steps)
    return transform(image.convert("RGB"))


def transform_cifar10_eval(image):
    data_range = globals().get('_DATA_RANGE', 'minus1_1')
    steps = [transforms.ToTensor()]
    if data_range != '0_1':
        steps.append(transforms.Lambda(lambda t: (t * 2) - 1))
    transform = transforms.Compose(steps)
    return transform(image.convert("RGB"))


def save_image_01(tensor, path, data_range='auto'):
    """
    Save image tensor safely for FID/IS evaluation.

    Training data and diffusion samples may be in [-1, 1].
    FID/IS image files should be saved in [0, 1].
    """
    x = tensor.detach().cpu()
    if data_range == 'minus1_1' or (data_range == 'auto' and x.min() < 0):
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    torchvision.utils.save_image(x, path)


def clear_export_images(folder):
    """Remove stale generated images so repeated evaluations stay comparable."""
    image_exts = {'.png', '.jpg', '.jpeg'}
    folder = Path(folder)
    if not folder.exists():
        return 0
    removed = 0
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in image_exts:
            item.unlink()
            removed += 1
    return removed


def transform_celeba(image):
    transform = transforms.Compose(
        [transforms.CenterCrop(178),  # crop the center region of the image
         transforms.Resize(64),  # resize the image to 32x32 pixels
         transforms.ToTensor()])
    return transform(image)


def transform_celeba_greyscale(image):
    transform = transforms.Compose(
        [transforms.CenterCrop(178),  # crop the center region of the image
         transforms.Resize(64),  # resize the image to 64x64 pixels
         transforms.Grayscale(),
         transforms.ToTensor()])
    return transform(image)


def transform_celebhq(image):
    transform = transforms.Compose(
        [transforms.ToTensor(),  # img_to_latents
         ])
    return transform(image)


def export_dataset(folder, dataset_name, num_images=sys.maxsize, train=True, offset=0, data_range='auto'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    dataset = get_dataset_from_name(dataset_name, train=train)

    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        image_name = f'image_{offset + i}.png'
        image_path = os.path.join(folder, image_name)
        save_image_01(image, image_path, data_range=data_range)


class CustomCelebA(Dataset):
    """Custom CelebA dataset that reads directly from local files, bypassing torchvision's validation."""

    attr_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                  'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blurry',
                  'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                  'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                  'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                  'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                  'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                  'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        img_dir = os.path.join(root, 'img_align_celeba')

        # Load attribute file: line1=count, line2=header, rest=data
        self.attr = {}
        self.attr_names = []
        with open(os.path.join(root, 'list_attr_celeba.txt'), 'r') as f:
            f.readline()  # skip count line
            self.attr_names = f.readline().strip().split()
            for line in f:
                parts = line.strip().split()
                if len(parts) < len(self.attr_names) + 1:
                    continue
                name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                # Convert -1 to 0 for binary attributes
                attrs = [max(0, x) for x in attrs]
                self.attr[name] = attrs

        # Load partition file to get train/test/val split
        self.filenames = []
        with open(os.path.join(root, 'list_eval_partition.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name, part_id = parts[0], int(parts[1])
                # Ensure name has .jpg extension (partition file may omit it)
                if not name.endswith('.jpg'):
                    name += '.jpg'
                # 0=training, 1=validation, 2=test
                if (split == 'train' and part_id == 0) or \
                   (split == 'test' and part_id == 2):
                    if os.path.exists(os.path.join(img_dir, name)):
                        self.filenames.append(name)

        self.img_dir = img_dir
        # Build attribute matrix for compatibility with create_labeled_celeba
        _attr_list = [self.attr[name] for name in self.filenames]
        if _attr_list:
            self.attr = np.array(_attr_list)
        else:
            self.attr = np.array([])
        print(f"CustomCelebA loaded: {split}={len(self.filenames)} images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        img_path = os.path.join(self.img_dir, name)
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, index


class FedPhDCelebA4(Dataset):
    """
    FedPhD-style CelebA labels:
        0: young male
        1: old male
        2: young female
        3: old female
    """

    classes = ['young_male', 'old_male', 'young_female', 'old_female']

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.indices = []
        self.labels = []

        male_idx = base_dataset.attr_names.index('Male')
        young_idx = base_dataset.attr_names.index('Young')

        for i in range(len(base_dataset)):
            male = int(base_dataset.attr[i, male_idx])
            young = int(base_dataset.attr[i, young_idx])

            if male == 1 and young == 1:
                label = 0
            elif male == 1 and young == 0:
                label = 1
            elif male == 0 and young == 1:
                label = 2
            else:
                label = 3

            self.indices.append(i)
            self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image, _ = self.base_dataset[base_index]
        label = int(self.labels[index])
        return image, label

    def __len__(self):
        return len(self.indices)

    def get_labels(self):
        return self.labels


def get_dataset_from_name(name, train=True, labeled=True):
    name = name.lower()
    if name == 'fmnist':
        data_dir = str(Path(__file__).parent.parent / 'data/fmnist/')
        return datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform_mnist)
    elif name in ('cifar', 'cifar10'):
        data_root = globals().get('_CUSTOM_DATA_ROOT',
                                  str(Path(__file__).parent.parent / 'data/cifar10/'))
        download = bool(globals().get('_DOWNLOAD_DATASET', False))
        transform = transform_cifar10_train if train else transform_cifar10_eval
        return datasets.CIFAR10(data_root, train=train, download=download, transform=transform)
    elif name == 'celeba':
        # Use custom data root if provided (set by lora_federator.py --data_root)
        data_root = globals().get('_CUSTOM_DATA_ROOT',
                                    str(Path(__file__).parent.parent / 'data/celeba/'))
        dataset = CustomCelebA(data_root, split="train" if train else "test", transform=transform_celeba)
        partition = globals().get('_PARTITION_RULE', '')
        if partition == 'fedphd-celeba4':
            dataset = FedPhDCelebA4(dataset)
        elif labeled:
            dataset = LabeledCelebA(dataset)
        return dataset
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_celeb_hq():
    return torchvision.datasets.ImageFolder(
        root=str(Path(__file__).parent.parent / 'data/celeba_hq_256/'),
        transform=transform_celebhq,
    )


def partition_cifar10_fedphd_two_classes(dataset, num_users, seed=2023):
    """FedPhD-style CIFAR10 split: each client owns exactly two classes."""
    rng = np.random.default_rng(seed)
    y_train = np.array(dataset.targets)

    if num_users < 5:
        raise ValueError('fedphd-cifar2 requires num_users >= 5 so all CIFAR10 classes are assigned.')

    if num_users % 5 != 0:
        print(
            f"[Warning] FedPhD CIFAR2 split is cleanest when num_users is a multiple of 5. "
            f"Got num_users={num_users}."
        )

    base_pairs = [
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
    ]
    client_classes = {cid: base_pairs[cid % len(base_pairs)] for cid in range(num_users)}
    class_to_clients = {k: [] for k in range(10)}
    for cid, cls_list in client_classes.items():
        for c in cls_list:
            class_to_clients[c].append(cid)

    net_dataidx_map = {cid: [] for cid in range(num_users)}
    for c in range(10):
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)
        clients_for_c = class_to_clients[c]
        splits = np.array_split(idx_c, len(clients_for_c))
        for cid, split in zip(clients_for_c, splits):
            net_dataidx_map[cid].extend(split.tolist())

    for cid in net_dataidx_map:
        rng.shuffle(net_dataidx_map[cid])
        net_dataidx_map[cid] = np.array(net_dataidx_map[cid], dtype=np.int64)

    return net_dataidx_map


def partition_celeba_fedphd_one_class(dataset, num_users, seed=2023):
    """FedPhD-style CelebA split: each client owns exactly one of four labels."""
    rng = np.random.default_rng(seed)
    y_train = dataset.get_labels()
    if num_users < 4:
        raise ValueError('fedphd-celeba4 requires num_users >= 4 so all CelebA4 labels are assigned.')
    class_to_clients = {k: [] for k in range(4)}
    for cid in range(num_users):
        class_to_clients[cid % 4].append(cid)

    net_dataidx_map = {cid: [] for cid in range(num_users)}
    for c in range(4):
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)
        clients_for_c = class_to_clients[c]
        splits = np.array_split(idx_c, len(clients_for_c))
        for cid, split in zip(clients_for_c, splits):
            net_dataidx_map[cid].extend(split.tolist())

    for cid in net_dataidx_map:
        rng.shuffle(net_dataidx_map[cid])
        net_dataidx_map[cid] = np.array(net_dataidx_map[cid], dtype=np.int64)

    return net_dataidx_map


def assert_fedphd_cifar2_split(y_train, net_dataidx_map):
    for cid, idxs in net_dataidx_map.items():
        labels = np.unique(y_train[idxs])
        if len(labels) != 2:
            raise RuntimeError(
                f"FedPhD CIFAR2 split failed: client {cid} has labels {labels}, "
                f"expected exactly 2 classes."
            )
    print("[OK] FedPhD CIFAR2 split verified: each client has exactly 2 classes.")


def assert_fedphd_celeba4_split(y_train, net_dataidx_map):
    for cid, idxs in net_dataidx_map.items():
        labels = np.unique(y_train[idxs])
        if len(labels) != 1:
            raise RuntimeError(
                f"FedPhD CelebA4 split failed: client {cid} has labels {labels}, "
                f"expected exactly 1 class."
            )
    print("[OK] FedPhD CelebA4 split verified: each client has exactly 1 class.")


# https://github.com/Xtra-Computing/NIID-Bench/blob/5371adbff98156793a413c7658923673b4aef7d7/utils.py#L40
def get_partitioned_dataset(args, seed=2023, beta=0.5):
    seed = int(getattr(args, 'seed', seed))
    partition = getattr(args, 'partition', '')
    globals()['_PARTITION_RULE'] = partition
    dataset = get_dataset_from_name(args.dataset)

    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_name = args.dataset.lower()
    y_train = np.array(dataset.targets) if dataset_name in ("fmnist", "cifar", "cifar10") else dataset.get_labels()

    if dataset_name in ('cifar', 'cifar10') and partition == 'fedphd-cifar2':
        net_dataidx_map = partition_cifar10_fedphd_two_classes(
            dataset,
            num_users=args.num_users,
            seed=seed,
        )
        assert_fedphd_cifar2_split(y_train, net_dataidx_map)
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return dataset, net_dataidx_map, traindata_cls_counts

    if dataset_name == 'celeba' and partition == 'fedphd-celeba4':
        net_dataidx_map = partition_celeba_fedphd_one_class(
            dataset,
            num_users=args.num_users,
            seed=seed,
        )
        assert_fedphd_celeba4_split(y_train, net_dataidx_map)
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return dataset, net_dataidx_map, traindata_cls_counts

    distribution = "noniid-label"
    if args.iid and args.unequal:
        distribution = "iid-quantity"
    elif args.iid:
        distribution = "iid-homo"

    if distribution == "iid-homo":
        idxs = np.random.permutation(len(dataset))
        batch_idxs = np.array_split(idxs, args.num_users)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_users)}

    elif distribution == "noniid-label":
        min_size = 0
        min_require_size = 10
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.num_users)]
            for k in range(args.num_classes):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, args.num_users))
                proportions = np.array(
                    [p * (len(idx_j) < len(dataset) / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(args.num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif distribution == "iid-quantity":
        idxs = np.random.permutation(len(dataset))
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, args.num_users))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.num_users)}

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return dataset, net_dataidx_map, traindata_cls_counts


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts


def is_down_parameter(name):
    return 'init_' in name or 'time_' in name or 'downs' in name or 'label_emb' in name


def is_mid_parameter(name):
    return 'mid_' in name


def is_up_parameter(name):
    return 'ups' in name or 'final_' in name


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : Unet')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_update_parameters(update):
    if update is None:
        return 0
    return sum(u[1].numel() for u in update.items())


def load_pickle_stats(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def export_samples(diffuser, folder, conditional, model, time_steps, image_size, channels, num_classes=10,
                   num_images=1000, use_ddim=False, ddim_steps=100, batch_size=256, data_range='auto'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        removed = clear_export_images(folder)
        if removed:
            print(f'Cleared {removed} stale images from {folder}')

    labels = []
    if conditional:
        while len(labels) < num_images:
            # choose labels at random
            labels.append(np.random.randint(num_classes))

    num_to_go = num_images
    img_num = 1000
    label_offset = 0  # track label position across batches

    while num_to_go > 0:
        cur_batch_size = min(batch_size, num_to_go)
        # FIX: pass labels to sampler when conditional generation is used
        sample_labels = None
        if conditional and labels:
            _device = next(model.parameters()).device
            sample_labels = torch.tensor(labels[label_offset:label_offset + cur_batch_size], device=_device)

        images = diffuser.sample(model, time_steps, image_size=image_size, batch_size=cur_batch_size,
                                  channels=channels, labels=sample_labels,
                                  use_ddim=use_ddim, ddim_steps=ddim_steps)
        for i in range(len(images[-1])):
            image_name = f'image_{img_num}_label{labels[label_offset + i]}.png' if conditional else f'image{img_num}.png'
            img_num += 1
            image_path = os.path.join(folder, image_name)
            save_image_01(images[-1][i], image_path, data_range=data_range)
        if conditional:
            label_offset += cur_batch_size
        num_to_go -= cur_batch_size


def show_samples(diffuser, conditional, model, dataset, time_steps, image_size, channels,
                 use_ddim=False, ddim_steps=100):
    # sample one image for each class, and fill randomly
    num_samples = 64

    labels = []
    if conditional:
        labels = [i for i in range(len(dataset.classes))]
        while len(labels) < num_samples:
            labels.append(np.random.randint(len(dataset.classes)))
        _device = next(model.parameters()).device
        samples = diffuser.sample(model, time_steps, image_size=image_size, batch_size=num_samples,
                                  channels=channels,
                                  labels=torch.tensor(labels, device=_device), all_steps=True,
                                  use_ddim=use_ddim, ddim_steps=ddim_steps)
    else:
        samples = diffuser.sample(model, time_steps, image_size=image_size, batch_size=num_samples,
                                  channels=channels, all_steps=True,
                                  use_ddim=use_ddim, ddim_steps=ddim_steps)

    # show a random one
    random_index = 4
    plt.imshow(samples[-1][random_index].cpu().numpy().reshape(image_size, image_size, channels), cmap="gray")

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        if conditional:
            plt.title(f'label {dataset.classes[labels[i - 1]]}')
        plt.axis("off")
        plt.imshow(samples[-1][i - 1].cpu().numpy().reshape(image_size, image_size, channels), cmap="gray")
    plt.show()

    random_index = 6

    fig = plt.figure()
    ims = []
    for i in range(time_steps):
        im = plt.imshow(samples[i][random_index].cpu().numpy().reshape(image_size, image_size, channels), cmap="gray",
                        animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False, repeat_delay=1000)
    animate.save('diffusion2.gif')
    plt.show()


celeba_labels = {
    # Young males
    "Male_Young_Black_Hair": 0,
    "Male_Young_Brown_Hair": 1,
    "Male_Young_Blond_Hair": 2,
    "Male_Young_Gray_Hair": 3,
    # Old males
    "Male_Old_Black_Hair": 4,
    "Male_Old_Brown_Hair": 5,
    "Male_Old_Blond_Hair": 6,
    "Male_Old_Gray_Hair": 7,
    # Young females
    "Female_Young_Black_Hair": 8,
    "Female_Young_Brown_Hair": 9,
    "Female_Young_Blond_Hair": 10,
    "Female_Young_Gray_Hair": 11,
    # Old females
    "Female_Old_Black_Hair": 12,
    "Female_Old_Brown_Hair": 13,
    "Female_Old_Blond_Hair": 14,
    "Female_Old_Gray_Hair": 15,
}


def create_labeled_celeba(dataset):
    # 160.000+ unlabelled training images, only 100.000+ properly annotated
    # assign labels based on combinations of gender, age and hair color (16 classes)
    labeled_dataset = []
    for i in range(len(dataset)):
        # Get the attribute labels for gender, age, and hair color
        gender_label = dataset.attr[i, dataset.attr_names.index('Male')]
        age_label = dataset.attr[i, dataset.attr_names.index('Young')]
        black_hair_label = dataset.attr[i, dataset.attr_names.index('Black_Hair')]
        brown_hair_label = dataset.attr[i, dataset.attr_names.index('Brown_Hair')]
        blond_hair_label = dataset.attr[i, dataset.attr_names.index('Blond_Hair')]
        gray_hair_label = dataset.attr[i, dataset.attr_names.index('Gray_Hair')]

        category_string = 'Male' if gender_label == 1 else 'Female'
        if age_label == 1:
            category_string += '_Young'
        else:
            category_string += '_Old'
        if black_hair_label == 1:
            category_string += '_Black_Hair'
            labeled_dataset.append((i, celeba_labels[category_string]))
        elif brown_hair_label == 1:
            category_string += '_Brown_Hair'
            labeled_dataset.append((i, celeba_labels[category_string]))
        elif blond_hair_label == 1:
            category_string += '_Blond_Hair'
            labeled_dataset.append((i, celeba_labels[category_string]))
        elif gray_hair_label == 1:
            category_string += '_Gray_Hair'
            labeled_dataset.append((i, celeba_labels[category_string]))
    return labeled_dataset
