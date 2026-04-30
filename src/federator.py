#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import csv
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Create a dummy diff_model module to allow loading legacy .pth files that were pickled with it
# Map legacy class names to current implementations in unet.py
import types, importlib
from unet import (SinusoidalPositionEmbeddings, Residual, Upsample, Downsample,
                  Block, ResnetBlock, ConvNextBlock, Attention, LinearAttention,
                  PreNorm, Unet, UnetConditional)

_legacy_module = types.ModuleType('diff_model')
_legacy_module.SinusoidalPositionEmbeddings = SinusoidalPositionEmbeddings
_legacy_module.Residual = Residual
_legacy_module.Upsample = Upsample
_legacy_module.Downsample = Downsample
_legacy_module.Block = Block
_legacy_module.ResnetBlock = ResnetBlock
_legacy_module.ConvNextBlock = ConvNextBlock
_legacy_module.Attention = Attention
_legacy_module.LinearAttention = LinearAttention
_legacy_module.PreNorm = PreNorm
_legacy_module.Unet = Unet
_legacy_module.UnetConditional = UnetConditional

if 'diff_model' not in sys.modules:
    sys.modules['diff_model'] = _legacy_module

from aggregation import fed_avg_weights
from client import Client, SplitTrainingTask
from diffuser import Diffuser
from options import args_parser
from unet import UnetConditional, Unet
from utils import exp_details, DecoderTrainingTask, is_up_parameter, \
    is_mid_parameter, export_dataset, count_model_parameters, count_update_parameters, get_partitioned_dataset, \
    export_samples, show_samples, perform_evaluation


def parse_dim_mults(dim_mults_str: str):
    if not dim_mults_str:
        return (1, 2, 4)
    return tuple(int(x.strip()) for x in dim_mults_str.split(',') if x.strip())


def create_training_tasks(num_clients, train_mode):
    tasks = None
    if train_mode == 'full':
        return tasks
    elif train_mode == 'udec':
        return [DecoderTrainingTask(False) for _ in range(num_clients)]
    elif train_mode == 'ulatdec':
        return [DecoderTrainingTask(True) for _ in range(num_clients)]
    elif train_mode == 'usplit':
        tasks = []
        if num_clients == 1:
            # training up, down and middle task by itself
            tasks.append(SplitTrainingTask(up=True, down=True, mid=True))
        else:
            for _ in range(int(num_clients / 2)):
                # randomly assign training the mid task to the up or down task
                rand = random.randint(0, 1)
                tasks.append(SplitTrainingTask(up=True, down=False, mid=rand == 0))
                tasks.append(SplitTrainingTask(up=False, down=True, mid=rand == 1))
            if num_clients % 2 == 1:
                # randomly assign the up or down task in addition to the middle task
                rand = random.randint(0, 1)
                # add the task for the last client
                tasks.append(SplitTrainingTask(up=rand == 0, down=rand == 1, mid=True))
    return tasks


def create_model_update(global_weights, train_mode):
    if train_mode == 'ulatdec' or train_mode == 'udec':
        update = OrderedDict()
        for name, param in global_weights.items():
            if is_up_parameter(name) or is_mid_parameter(name) and train_mode == 'ulatdec':
                update[name] = param
        return update
    else:
        return global_weights


def main(args):
    start_time = time.time()
    parent_path = Path(__file__).parent.parent

    exp_details(args)

    # ---- Set custom data root if provided ----
    data_root = getattr(args, 'data_root', '')
    download_dataset = bool(getattr(args, 'download_dataset', 0))
    import utils as _utils
    if data_root:
        _utils._CUSTOM_DATA_ROOT = data_root
        print(f'Using custom data root: {data_root}')
    _utils._DOWNLOAD_DATASET = download_dataset
    _utils._PARTITION_RULE = getattr(args, 'partition', '')

    # load dataset and user groups
    train_dataset, client_groups, data_stats = get_partitioned_dataset(args)
    print(data_stats)

    # intialize the diffuser
    diffuser = Diffuser(args.time_steps)

    image_size = args.image_size
    is_conditional = args.conditional == 1
    channels = args.num_channels
    model_dim = int(getattr(args, 'model_dim', image_size))
    if model_dim <= 0:
        model_dim = image_size
    dim_mults = parse_dim_mults(getattr(args, 'dim_mults', '1,2,4'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    # check if its further training of an existing model
    existing_model_path = parent_path / args.load_model
    found_existing_model = False
    if os.path.isfile(existing_model_path):
        found_existing_model = True

    model_name = f'model_{args.dataset}_' \
                 f'R[{args.rounds + args.round_offset}]_' \
                 f'K[{args.num_users}]_' \
                 f'C[{args.frac}]_' \
                 f'E[{args.local_ep}]_' \
                 f'B[{args.local_bs}]_' \
                 f'T[{args.time_steps}]_' \
                 f'I[{args.iid},{args.unequal}]_' \
                 f'Y[{args.conditional}]_' \
                 f'{args.train_mode}'
    if args.train == 0 and args.load_model:
        model_name = f'infer_{Path(args.load_model).stem}'

    # Build model structure first (always needed for training)
    global_model = UnetConditional(
        dim=model_dim,
        channels=channels,
        dim_mults=dim_mults,
        num_classes=args.num_classes,
    ) if is_conditional else Unet(
        dim=model_dim,
        channels=channels,
        dim_mults=dim_mults,
    )
    print(f'[Model] image_size={image_size}, model_dim={model_dim}, dim_mults={dim_mults}, conditional={is_conditional}')
    global_model.to(device)

    if found_existing_model:
        # Load pre-trained weights
        print(f'Loading model from {existing_model_path}')
        checkpoint = torch.load(existing_model_path, map_location=torch.device(device), weights_only=False)
        if isinstance(checkpoint, dict):
            # state_dict format - load into existing model
            global_model.load_state_dict(checkpoint)
        elif hasattr(checkpoint, 'state_dict'):
            # Full model object - use it directly if it's a compatible nn.Module
            # Try loading its state_dict first; if shapes mismatch, use the loaded model itself
            try:
                global_model.load_state_dict(checkpoint.state_dict())
                print('Loaded pre-trained weights into new model')
            except RuntimeError:
                print(f'Architecture mismatch - using loaded model directly')
                checkpoint.to(device)
                global_model = checkpoint
        else:
            raise ValueError(f'Unsupported model format in {existing_model_path}')
    elif args.train == 0 and not found_existing_model:
        print(f'Warning: --train=0 but no model found at {existing_model_path}, using random weights')
        print('Use --load_model=<path> to specify a pre-trained model file')

    num_params = count_model_parameters(global_model)
    print(f"{'Creating' if args.train == 1 else 'Using'} model with {num_params} "
          f"trainable parameters")

    # Training if flag was set
    if args.train == 1:
        # Keep track of training loss and number of shared parameters every round
        train_loss = []
        num_params_shared = []

        # Create the model and result folder
        model_folder = parent_path / 'models/'
        result_folder = parent_path / f'results/{time.strftime("%Y%m%d-%H%M%S")}_{model_name}/'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # TODO: remove, only for first task, establishing number of rounds
        checkpoint_path = result_folder / f'{model_name}_temp_initial.pth'
        torch.save(global_model, checkpoint_path)
        print(f"Created initial model checkpoint {checkpoint_path}")

        # Copy weights
        global_weights = global_model.state_dict()
        client_models = {}

        save_local_models = args.train_mode == 'udec' or args.train_mode == 'ulatdec'

        # in case of continued training and local models, fill dict with existing local models and apply global update
        if save_local_models and found_existing_model and args.train == 1:
            model_update = create_model_update(global_weights, args.train_mode)
            for idx in range(args.num_users):
                existing_client_path = parent_path / args.load_model.replace(".pth", "_client[{}].pth".format(idx))
                if not os.path.isfile(existing_client_path):
                    print(f'Cannot load existing client model {existing_client_path}')
                else:
                    client_model = torch.load(existing_client_path, map_location=torch.device(device))
                    client_models[idx] = Client(args=args,
                                                dataset=train_dataset,
                                                initial_model=client_model,
                                                indices=client_groups[idx],
                                                time_steps=args.time_steps,
                                                diffuser=diffuser)
                    client_task = DecoderTrainingTask(args.train_mode == 'ulatdec')
                    client_models[idx].apply_global_update(model_update, client_task)

        model_update = None
        for round in tqdm(range(args.rounds)):
            local_weights, local_losses = [], []
            print(f"\n-------------------Global Training Round[{round + args.round_offset}]---------")

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            user_indices = np.random.choice(range(args.num_users), m, replace=False)
            client_tasks = create_training_tasks(m, args.train_mode)

            params_shared = 0
            for i, idx in enumerate(user_indices):
                print(f"\n-------------------------------")
                print(f"Updating weights for client {idx}")

                local_model = client_models[idx] if idx in client_models.keys() else Client(args=args,
                                                                                            dataset=train_dataset,
                                                                                            initial_model=copy.deepcopy(
                                                                                                global_model),
                                                                                            indices=client_groups[idx],
                                                                                            time_steps=args.time_steps,
                                                                                            diffuser=diffuser)
                if idx not in client_models.keys():
                    # Count number of initial parameters being sent
                    params_shared += num_params
                    client_models[idx] = local_model

                # Count number of parameters sent in global model update
                params_shared += count_update_parameters(model_update)
                w, loss = local_model.update_diff_weights(
                    model_update=model_update,
                    training_task=client_tasks[i] if args.train_mode != 'full' else None)
                local_weights.append(copy.deepcopy(w))

                # Count number of parameters sent back from local update
                params_shared += count_update_parameters(w)
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            dataset_lengths = [len(client_groups[i]) for i in user_indices]
            global_weights = fed_avg_weights(global_weights, local_weights, dataset_lengths, client_tasks)

            # update global weights
            global_model.load_state_dict(global_weights)

            # prepare the update for the clients
            model_update = create_model_update(global_weights, args.train_mode)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            num_params_shared.append(params_shared)

            print(f"\n-------------------------------")
            print(f"Avg training loss over clients: {loss_avg}")
            print(f"Number of parameters shared this round: {params_shared}")
            print(f"Cumulative number of parameters shared: {sum(num_params_shared)}")
            print('Run Time: {0:0.4f} seconds'.format(time.time() - start_time))
            print("------------------------------------------------------------\n")

            if save_local_models and round < args.rounds - 1:
                # Create checkpoint of all local models every global round
                for client in client_models.keys():
                    checkpoint_path = result_folder / f'{model_name}_client[{client}]_temp_R[{round + args.round_offset}].pth'
                    torch.save(client_models[client].local_model, checkpoint_path)
                    print(f"Created model checkpoint {checkpoint_path}")

            elif round < args.rounds - 1:
                # Create checkpoint of model every global round
                checkpoint_path = result_folder / f'{model_name}_temp_R[{round + args.round_offset}].pth'
                torch.save(global_model, checkpoint_path)
                print(f"Created model checkpoint {checkpoint_path}")

                # save output images of every odd round starting with the fifth
                if args.exp_rounds > 0 and round + args.round_offset >= 5 and ((round + args.round_offset) % 2) == 1:
                    export_samples(diffuser, parent_path / f'exports/{model_name}_temp_R[{round + args.round_offset}]',
                                   is_conditional, global_model, args.time_steps, image_size, channels,
                                   args.num_classes, args.exp_rounds,
                                   use_ddim=bool(getattr(args, 'use_ddim', 1)),
                                   ddim_steps=getattr(args, 'ddim_steps', 100),
                                   batch_size=getattr(args, 'eval_batch_size', 256))

        # Export the averaged loss data and number of parameter updates shared
        file_name = result_folder / f'{model_name}.csv'
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the headers (optional)
            writer.writerow(['loss', 'params_shared'])
            # Write the values for each round
            for value1, value2 in zip(train_loss, num_params_shared):
                writer.writerow([value1, value2])
            writer.writerow(["runtime", time.time() - start_time])

        if save_local_models:
            # In case of udec or ulatdec training, export all client models separately
            for client in client_models.keys():
                client_model_path = result_folder / f'{model_name}_client[{client}].pth'
                torch.save(client_models[client].local_model, client_model_path)
                client_model_path = model_folder / f'{model_name}_client[{client}].pth'
                torch.save(client_models[client].local_model, client_model_path)
                print(f'Exported final client model {client_model_path}')
        else:
            # Export the final model in the result and model folders
            final_model_path = result_folder / f'{model_name}.pth'
            torch.save(global_model, final_model_path)
            final_model_path = model_folder / f'{model_name}.pth'
            torch.save(global_model, final_model_path)
            # Also save a simplified copy to project root with hyperparam tags
            root_model_name = (f'fedavg_model_{args.dataset}'
                               f'_R[{args.rounds + args.round_offset}]'
                               f'_K[{args.num_users}]_E[{args.local_ep}].pth')
            torch.save(global_model, parent_path / root_model_name)
            print(f'Exported final model {final_model_path}')
            print(f'Also saved to project root: {root_model_name}')

    if args.export_dataset > 0:
        real_split = getattr(args, 'eval_real_split', 'train')
        real_train = real_split == 'train'
        real_dir = parent_path / f'exports/{args.dataset}/dataset_{real_split}'
        export_dataset(real_dir, args.dataset, args.export_dataset,
                       train=real_train,
                       data_range=getattr(args, 'data_range', 'minus1_1'))
        print(f'Exported {args.export_dataset} real {real_split} samples to {real_dir}')
    if args.export_samples > 0:
        fake_dir = parent_path / f'exports/{model_name}'
        export_samples(diffuser, fake_dir, is_conditional, global_model,
                       int(args.time_steps),
                       image_size, channels, args.num_classes, args.export_samples,
                       use_ddim=bool(getattr(args, 'use_ddim', 1)),
                       ddim_steps=getattr(args, 'ddim_steps', 100),
                       batch_size=getattr(args, 'eval_batch_size', 256),
                       data_range=getattr(args, 'data_range', 'minus1_1'))
        print(f'Exported {args.export_samples} generated samples to {fake_dir}')
    if args.export_samples > 0 and args.export_dataset > 0:
        real_split = getattr(args, 'eval_real_split', 'train')
        real_dir = parent_path / f'exports/{args.dataset}/dataset_{real_split}'
        fake_dir = parent_path / f'exports/{model_name}'
        print(f'\n{"="*50}')
        print('Computing FID, IS & Precision/Recall...')
        print(f'  Real (reference): {real_dir}')
        print(f'  Fake (generated): {fake_dir}')
        print(f'{"="*50}')

        import datetime as _dt
        _ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        if args.train == 0 and getattr(args, 'load_model', ''):
            _model_stem = Path(args.load_model).stem
            _tag = f'eval_fedavg_{args.dataset}_{_model_stem}_{_ts}.log'
        else:
            _tag = (f'eval_fedavg_{args.dataset}_R[{args.rounds}]_K[{args.num_users}]'
                    f'_E[{args.local_ep}]_B[{args.local_bs}]_{_ts}.log')
        _eval_log_dir = parent_path / 'results' / 'eval_logs'
        _experiment_meta = {
            'Method': 'fedavg',
            'Load model': getattr(args, 'load_model', '') or '(none)',
            'Data dist.': 'IID' if getattr(args, 'iid', 0) else 'Non-IID',
            'Partition': getattr(args, 'partition', '') or '(default)',
            'Client mode': 'Homogeneous',
        }
        perform_evaluation(real_path=str(real_dir), fake_path=str(fake_dir),
                           num_samples=min(
                               int(getattr(args, 'eval_num_samples', 30000)),
                               args.export_samples,
                               args.export_dataset,
                           ),
                           eval_log_dir=str(_eval_log_dir), config_tag=_tag,
                           batch_size=getattr(args, 'eval_batch_size', 256),
                           compute_is=bool(getattr(args, 'compute_is', 1)),
                           experiment_meta=_experiment_meta)
    if args.show_samples:
        show_samples(diffuser, is_conditional, global_model, train_dataset, args.time_steps, image_size,
                     channels, use_ddim=bool(getattr(args, 'use_ddim', 1)),
                     ddim_steps=getattr(args, 'ddim_steps', 100))


if __name__ == '__main__':
    args = args_parser()
    main(args)
