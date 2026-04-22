import csv
import os
import shutil
import time
from pathlib import Path

import torch

from diffuser import Diffuser
from fid_score import calculate_fid
from utils import create_fid_config, export_samples


def calculate_fids(paths):
    # Set parameters
    time_steps = 1000
    image_size = 28
    num_channels = 1
    num_classes = 10
    num_samples = 1000
    # use only models of the same dataset
    dataset = 'fmnist'

    parent_path = Path(__file__).parent.parent

    ref_dir_or_npz = parent_path / 'evaluation/celeba_5000.npz'  # f'evaluation/{dataset}_{num_samples}'  # pass npz file here in case of precompute  #

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(ref_dir_or_npz) and not os.path.isfile(ref_dir_or_npz):
        print(f'could not find reference directory or npz file {ref_dir_or_npz}')

    data = []
    for path in paths:
        exact_path = parent_path / path
        model_name = filename = os.path.splitext(os.path.basename(path))[0]

        if os.path.isfile(exact_path):
            # load model from checkpoint
            model = torch.load(exact_path, map_location=torch.device(device))
            model.to(device)

            # # samples directory
            sample_dir = parent_path / f'evaluation/{model_name}'
            if os.path.exists(sample_dir):
                # clean the folder if it already existed
                shutil.rmtree(sample_dir)
                os.makedirs(sample_dir)

            # initialize a diffuser time_steps
            diffuser = Diffuser(time_steps)

            # export samples with the model
            export_samples(diffuser, sample_dir, False, model, time_steps,
                           image_size, num_channels, num_classes, num_samples)

            # calculate the FID score 5nd add it to the data
            fid_config = create_fid_config(paths=(str(ref_dir_or_npz), str(sample_dir)))
            fid = calculate_fid(fid_config)
            model_fid = {"model_name": model_name, "fid": fid}
            print(model_fid)
            data.append(model_fid)
        else:
            print(f'failed to locate model {path}')

    # export the csv file
    csv_file = parent_path / f'evaluation/{time.strftime("%Y%m%d-%H%M%S")}_{dataset}.csv'

    # Generate CSV file
    with open(csv_file, 'w', newline='') as file:
        field_names = ['model_name', 'fid']
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file '{csv_file}' has been generated.")


if __name__ == '__main__':

    paths_backup = [
        # 'models/model_celeba_R[30]_K[5]_C[1]_E[5]_B[64]_T[1000]_I[1,0]_Y[0]_full.pth'


        # number of epochs
        # 'models/model_fmnist_R[15]_K[1]_C[1]_E[2]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[1]_C[1]_E[3]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[1]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[1]_C[1]_E[8]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        #
        # 'models/model_fmnist_R[15]_K[2]_C[1]_E[2]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[2]_C[1]_E[3]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[2]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[2]_C[1]_E[8]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        #
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[2]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[3]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[8]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',

        # quantity-skewed IID
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_usplit.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[0].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[1].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[2].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[3].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[4].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[0].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[1].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[2].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[3].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[4].pth',

        # label-skewed non-IID
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_usplit.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[0].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[1].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[2].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[3].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[4].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[0].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[1].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[2].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[3].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[4].pth',

        # IID
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_usplit.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[0].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[1].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[2].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[3].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[4].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[0].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[1].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[2].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[3].pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[4].pth',

        #  Number of local epochs
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[2]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[3]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[8]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',

        # Number of clients
        # 'models/model_fmnist_R[15]_K[2]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[4]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[5]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[8]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[10]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[15]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
        # 'models/model_fmnist_R[15]_K[20]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',

        # Number of rounds
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_initial.pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[0].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[1].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[4].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[9].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[14].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[19].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[24].pth',
        # 'results/20230527-140611_num_rounds/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
    ]
    calculate_fids(paths_backup)
