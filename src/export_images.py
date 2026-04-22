import os
import shutil
from pathlib import Path

import torch
from diffuser import Diffuser
from utils import export_samples

parent_path = Path(__file__).parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"

paths = [
    'models/model_celeba_R[30]_K[5]_C[1]_E[5]_B[64]_T[1000]_I[1,0]_Y[0]_full.pth'
    # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.pth',
    # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_usplit.pth',
    # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[1].pth',
    # 'models/model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[1].pth',
    # 'models/model_fmnist_R[30]_K[1]_C[1]_E[1]_B[128]_T[1000]_I[1,0]_Y[0]_full_temp_R[14].pth',
]

for path in paths:
    exact_path = parent_path / path
    model_name = filename = os.path.splitext(os.path.basename(path))[0]

    if os.path.isfile(exact_path):
        # load model from checkpoint
        model = torch.load(exact_path, map_location=torch.device(device))
        model.to(device)

        # samples directory
        sample_dir = parent_path / f'evaluation/{model_name}'
        if os.path.exists(sample_dir):
            # clean the folder if it already existed
            shutil.rmtree(sample_dir)
            os.makedirs(sample_dir)

        # initialize a diffuser time_steps
        diffuser = Diffuser(1000)

        # export samples with the model
        export_samples(diffuser, sample_dir, False, model, 1000, 64, 3, 16, 500)
