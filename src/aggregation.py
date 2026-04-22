import copy

import torch

from utils import DecoderTrainingTask, is_up_parameter, \
    is_mid_parameter, is_down_parameter, SplitTrainingTask


def average_weights(w):
    """
    Returns the simple average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def fed_avg_weights(global_weights, weight_updates, dataset_lengths, client_tasks):
    """
    Returns the average of the weights, weighted by dataset size.
    """
    keys = global_weights.keys()
    total_size = sum(dataset_lengths)
    if client_tasks is None:
        for key in keys:
            global_weights[key] = sum(torch.mul(copy.deepcopy(weight_updates[i][key]),
                                                float(dataset_lengths[i]) / total_size) for i in
                                      range(len(weight_updates)))

    elif isinstance(client_tasks[0], DecoderTrainingTask):
        for key in keys:
            if is_up_parameter(key) or (is_mid_parameter(key) and client_tasks[0].mid):
                global_weights[key] = sum(torch.mul(copy.deepcopy(weight_updates[i][key]),
                                                    float(dataset_lengths[i]) / total_size) for i in
                                          range(len(weight_updates)))

    elif isinstance(client_tasks[0], SplitTrainingTask):
        down_updates = [i for i in range(len(weight_updates)) if client_tasks[i].down]
        mid_updates = [i for i in range(len(weight_updates)) if client_tasks[i].mid]
        up_updates = [i for i in range(len(weight_updates)) if client_tasks[i].up]
        total_down_size = sum(dataset_lengths[j] for j in down_updates)
        total_mid_size = sum(dataset_lengths[j] for j in mid_updates)
        total_up_size = sum(dataset_lengths[j] for j in up_updates)

        for key in keys:
            if is_down_parameter(key):
                global_weights[key] = sum(torch.mul(copy.deepcopy(weight_updates[i][key]),
                                                    float(dataset_lengths[i]) / total_down_size) for i in down_updates)
            elif is_up_parameter(key):
                global_weights[key] = sum(torch.mul(copy.deepcopy(weight_updates[i][key]),
                                                    float(dataset_lengths[i]) / total_up_size) for i in up_updates)
            else:
                global_weights[key] = sum(torch.mul(copy.deepcopy(weight_updates[i][key]),
                                                    float(dataset_lengths[i]) / total_mid_size) for i in mid_updates)
    return global_weights
