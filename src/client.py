#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils import SplitTrainingTask, is_up_parameter, is_mid_parameter, is_down_parameter, DecoderTrainingTask


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.dataset[self.indices[item]]
        return image, torch.tensor(label)


class Client(object):

    def __init__(self, args, dataset, initial_model, indices, time_steps, diffuser):
        self.args = args
        self.indices = list(indices)
        self.local_model = initial_model
        self.train_loader = DataLoader(DatasetSplit(dataset, indices), batch_size=self.args.local_bs,
                                       shuffle=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_steps = time_steps
        self.diffuser = diffuser

    def freeze_weights(self, training_task):
        if training_task is None:
            for _, param in self.local_model.named_parameters():
                param.requires_grad = True
            return

        if isinstance(training_task, DecoderTrainingTask):
            for name, param in self.local_model.named_parameters():
                param.requires_grad = (
                    is_up_parameter(name)
                    or (training_task.mid and is_mid_parameter(name))
                )
            return

        if isinstance(training_task, SplitTrainingTask):
            # USplit reduces communication by reporting only the assigned
            # network part. The local optimization itself remains full-model
            # so encoder/bottleneck/decoder can co-adapt during each client
            # update, matching the paper's "reporting task" description.
            for _, param in self.local_model.named_parameters():
                param.requires_grad = True

    def select_return_params(self, training_task):
        if training_task is None:
            return self.local_model.state_dict()
        return_params = OrderedDict()
        return_up = return_mid = return_down = False
        if isinstance(training_task, DecoderTrainingTask):
            return_up = True
            return_mid = training_task.mid
        elif isinstance(training_task, SplitTrainingTask):
            return_up = training_task.up
            return_mid = training_task.mid
            return_down = training_task.down
        for name, param in self.local_model.named_parameters():
            if is_up_parameter(name) and return_up:
                return_params[name] = param
            elif is_mid_parameter(name) and return_mid:
                return_params[name] = param
            elif is_down_parameter(name) and return_down:
                return_params[name] = param
        return return_params

    def apply_global_update(self, model_update, training_task):
        if model_update is None:
            return
        apply_up = apply_mid = apply_down = True
        if isinstance(training_task, DecoderTrainingTask):
            apply_up = True
            apply_mid = training_task.mid
            apply_down = False
        # elif isinstance(training_task, SplitTrainingTask):
        #     apply_up = training_task.up
        #     apply_mid = training_task.mid
        #     apply_down = training_task.down

        updated_weights = self.local_model.state_dict()
        for name, param in model_update.items():
            if is_up_parameter(name) and apply_up:
                updated_weights[name] = param
            elif is_mid_parameter(name) and apply_mid:
                updated_weights[name] = param
            elif is_down_parameter(name) and apply_down:
                updated_weights[name] = param

        self.local_model.load_state_dict(updated_weights)

    def update_diff_weights(self, model_update, training_task):
        # Apply the global model update from the server
        self.apply_global_update(model_update, training_task)
        self.freeze_weights(training_task)

        # Set mode to train model
        self.local_model.train()

        epoch_loss = []
        optimizer = None
        self.local_model.state_dict()

        # Set optimizer for the local updates
        non_frozen_parameters = [p for p in self.local_model.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(non_frozen_parameters, lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(non_frozen_parameters, lr=self.args.lr)

        for iter in range(self.args.local_ep):
            print(f"running local epoch {iter}")
            batch_loss = []
            for batch_index, (images, labels) in enumerate(self.train_loader):
                batch_size = images.shape[0]
                batch, batch_labels = images.to(self.device), labels.to(self.device)

                self.local_model.zero_grad()
                # Algorithm 1 line 3: sample t uniformly for every example in the batch
                t = torch.randint(0, self.time_steps, (batch_size,), device=self.device).long()

                loss = self.diffuser.p_losses(self.local_model, batch, t, loss_type="huber", labels=batch_labels)

                if batch_index % 100 == 0:
                    print("Loss:", loss.item())

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.select_return_params(training_task), sum(epoch_loss) / len(epoch_loss)
