#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--train_mode', type=str, default='udec', help="training mode, can be (full, usplit, udec or ulatdec)")

    # diffusion arguments
    parser.add_argument('--train', type=int, default=1,
                        help='Retrain the model instead of loading the checkpointed version')
    parser.add_argument('--load_model', type=str, default='',
                        help='The path of the model to load')
    parser.add_argument('--time_steps', type=float, default=1000, help='Number of timestamps (default: 1000)')
    parser.add_argument('--conditional', type=int, default=0, help='Whether the model is class labeled (1) or not (0)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
    parser.add_argument('--model_dim', type=int, default=0,
                        help='base channel dimension of U-Net; if 0, fallback to image_size')
    parser.add_argument('--dim_mults', type=str, default='1,2,4',
                        help='comma-separated U-Net dim multipliers')

    # data arguments
    parser.add_argument('--dataset', type=str, default='fmnist', help="name of dataset")
    parser.add_argument('--data_root', type=str, default='',
                        help='custom dataset root path, e.g. D:\\data\\cifar-10-python')
    parser.add_argument('--download_dataset', type=int, default=0,
                        help='whether torchvision should download the dataset')
    parser.add_argument('--partition', type=str, default='',
                        help='partition rule: fedphd-cifar2, fedphd-celeba4, or empty')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--image_size', type=int, default=28, help="size of image")
    parser.add_argument('--num_channels', type=int, default=1, help="the number of channels in the image")
    parser.add_argument('--iid', type=int, default=1,
                        help='whether to use i.i.d data (1) or not (0).')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='the number of classes in the dataset. default 10')
    parser.add_argument('--round_offset', type=int, default=0, help='offset for round numbering')


    # export/sampling arguments
    parser.add_argument('--export_samples', type=int, default=0,
                        help="the number of generated samples to export (0 by default)")
    parser.add_argument('--export_dataset', type=int, default=0,
                        help="the number of training samples to export (0 by default)")
    parser.add_argument('--show_samples', type=int, default=0, help="whether to show some test samples (1) or not (0)")
    parser.add_argument('--exp_rounds', type=int, default=0, help="whether to export intermediate samples (1) or not (0)")
    parser.add_argument('--eval_num_samples', type=int, default=30000,
                        help='number of generated samples for FedPhD-aligned evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='batch size for FedPhD-aligned generation/evaluation')
    parser.add_argument('--central_agg_interval', type=int, default=5,
                        help='central aggregation interval for FedPhD-style communication reporting')
    parser.add_argument('--compute_is', type=int, default=1,
                        help='whether to compute Inception Score')
    parser.add_argument('--eval_real_split', type=str, default='train', choices=['train', 'test'],
                        help='which real split to export for FID/IS reference')
    parser.add_argument('--data_range', type=str, default='minus1_1', choices=['minus1_1', '0_1'],
                        help='training/sample tensor range')
    
    # DDIM acceleration
    parser.add_argument('--use_ddim', type=int, default=1, help="use DDIM sampler (1) or DDPM (0)")
    parser.add_argument('--ddim_steps', type=int, default=100, help="number of DDIM sampling steps")

    args = parser.parse_args()
    return args
