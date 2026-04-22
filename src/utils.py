#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import pickle
import sys
from argparse import Namespace
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


def perform_evaluation(real_path, fake_path, dataset_to_export=None, num_samples=100,
                       eval_log_dir=None, config_tag=''):
    """Run FID + Precision/Recall evaluation and optionally log results."""
    import datetime

    if dataset_to_export is not None:
        export_dataset(real_path, dataset_to_export, num_samples, train=False)

    fid_config = create_fid_config(paths=(real_path, fake_path))
    precision_recall_config = create_precision_recall_config(path_real=real_path, path_fake=fake_path,
                                                             num_samples=num_samples)

    # Capture FID result
    _fid_result = calculate_fid(fid_config)
    _pr_result = calculate_precision_recall(precision_recall_config)

    # ---- Save evaluation log ----
    if eval_log_dir:
        eval_log_dir = Path(eval_log_dir)
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_name = f'{config_tag}_{ts}.log' if config_tag else f'eval_{ts}.log'
        log_path = eval_log_dir / log_name
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'Evaluation Log  {ts}\n')
            f.write(f'{"="*60}\n\n')
            f.write(f'Tag         : {config_tag}\n')
            f.write(f'Real path   : {real_path}\n')
            f.write(f'Fake path   : {fake_path}\n')
            f.write(f'Num samples : {num_samples}\n\n')
            f.write(f'FID Score   : {_fid_result}\n')
            f.write(f'Precision/Recall: {_pr_result}\n')
        print(f'\n[Evaluation log saved] {log_path}')


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


def export_dataset(folder, dataset_name, num_images=sys.maxsize, train=True, offset=0):
    if not os.path.exists(folder):
        os.makedirs(folder)

    dataset = get_dataset_from_name(dataset_name, train=train)

    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        image_name = f'image_{offset + i}.png'
        image_path = os.path.join(folder, image_name)
        torchvision.utils.save_image(image, image_path)


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


def get_dataset_from_name(name, train=True, labeled=True):
    if name == 'fmnist':
        data_dir = str(Path(__file__).parent.parent / 'data/fmnist/')
        return datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform_mnist)
    elif name == 'celeba':
        # Use custom data root if provided (set by lora_federator.py --data_root)
        data_root = globals().get('_CUSTOM_DATA_ROOT',
                                    str(Path(__file__).parent.parent / 'data/celeba/'))
        dataset = CustomCelebA(data_root, split="train" if train else "test", transform=transform_celeba)
        if labeled:
            dataset = LabeledCelebA(dataset)
        return dataset


def create_celeb_hq():
    return torchvision.datasets.ImageFolder(
        root=str(Path(__file__).parent.parent / 'data/celeba_hq_256/'),
        transform=transform_celebhq,
    )


# https://github.com/Xtra-Computing/NIID-Bench/blob/5371adbff98156793a413c7658923673b4aef7d7/utils.py#L40
def get_partitioned_dataset(args, seed=2023, beta=0.5):
    dataset = get_dataset_from_name(args.dataset)
    distribution = "noniid-label"
    if args.iid and args.unequal:
        distribution = "iid-quantity"
    elif args.iid:
        distribution = "iid-homo"

    np.random.seed(seed)
    torch.manual_seed(seed)
    y_train = np.array(dataset.targets) if args.dataset != "celeba" else dataset.get_labels()

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
    return 'init_' in name or 'time_' in name or 'downs' in name


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
                   num_images=1000, use_ddim=False, ddim_steps=100):
    if not os.path.exists(folder):
        os.makedirs(folder)

    labels = []
    if conditional:
        while len(labels) < num_images:
            # choose labels at random
            labels.append(np.random.randint(num_classes))

    num_to_go = num_images
    img_num = 1000
    label_offset = 0  # track label position across batches

    while num_to_go > 0:
        batch_size = min(500, num_to_go)
        # FIX: pass labels to sampler when conditional generation is used
        sample_labels = None
        if conditional and labels:
            _device = next(model.parameters()).device
            sample_labels = torch.tensor(labels[label_offset:label_offset + batch_size], device=_device)
            label_offset += batch_size

        images = diffuser.sample(model, time_steps, image_size=image_size, batch_size=batch_size,
                                  channels=channels, labels=sample_labels,
                                  use_ddim=use_ddim, ddim_steps=ddim_steps)
        for i in range(len(images[-1])):
            image_name = f'image_{img_num}_label{labels[i]}.png' if conditional else f'image{img_num}.png'
            img_num += 1
            image_path = os.path.join(folder, image_name)
            torchvision.utils.save_image(images[-1][i], image_path)
        num_to_go -= batch_size


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