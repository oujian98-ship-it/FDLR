from pathlib import Path

from calculate_fid_scores import calculate_fids
from federator import main
from utils import create_training_config


def get_model_paths(config):
    model_paths = []
    model_path = Path(__file__).parent.parent / f'models/model_{config.dataset}_' \
                                                f'R[{config.rounds + config.round_offset}]_' \
                                                f'K[{config.num_users}]_' \
                                                f'C[{config.frac}]_' \
                                                f'E[{config.local_ep}]_' \
                                                f'B[{config.local_bs}]_' \
                                                f'T[{config.time_steps}]_' \
                                                f'I[{config.iid},{config.unequal}]_' \
                                                f'Y[{config.conditional}]_' \
                                                f'{config.train_mode}.pth'
    if config.train_mode == 'udec' or config.train_mode == 'ulatdec':
        for i in range(config.num_users):
            client_path = model_path.with_name(model_path.stem + f'_client[{i}]' + model_path.suffix)
            model_paths.append(client_path)
    else:
        model_paths.append(model_path)
    return model_paths


if __name__ == '__main__':
    num_times = 4

    for i in range(num_times):
        configs = []

        # K=1
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=1, local_ep=1))

        # E=1
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=2, local_ep=1))
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=5, local_ep=1))

        # K=10
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=10, local_ep=1))
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=10, local_ep=2))
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=10, local_ep=3))
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=10, local_ep=5))
        configs.append(create_training_config(rounds=15, dataset='fmnist', train_mode='full', num_users=10, local_ep=8))


        #configs.append(create_training_config(rounds=30, dataset='celeba', train_mode='full', num_users=5, local_ep=5, local_bs=64, num_classes=16, image_size=64, num_channels=3, exp_rounds=50))

        # execute the training
        fid_paths = []
        for config in configs:
            fid_paths += get_model_paths(config)
            main(config)

        # calculate the fid scores for this training round
        calculate_fids(fid_paths)
