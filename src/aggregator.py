import statistics

data = [
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[0]", 41.66394746413192),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[1]", 34.20272685740571),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[2]", 48.84017110818155),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[3]", 60.65573280582822),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec_client[4]", 54.144948892799846),

    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[0]", 231.63902521047677),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[1]", 364.33636722316635),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[2]", 22.680830579977624),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[3]", 28.79261944089808),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_udec_client[4]", 64.02889961314304),

    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[0]", 122.36497425280083),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[1]", 63.04083178179843),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[2]", 50.59632261550189),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[3]", 41.09396862798391),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_udec_client[4]", 73.7093154285813),

    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[0]", 40.7078051345768),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[1]", 32.83616000347257),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[2]", 43.07079513305297),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[3]", 59.128507094400874),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec_client[4]", 70.20264095432358),

    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[0]", 112.9843364584004),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[1]", 383.88299657639),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[2]", 28.943038095157704),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[3]", 30.160291124652645),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[1,1]_Y[0]_ulatdec_client[4]", 47.774178203636836),

    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[0]", 135.2978404383578),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[1]", 62.04571247563268),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[2]", 43.9353882254494),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[3]", 39.988724416734726),
    ("model_fmnist_R[15]_K[5]_C[1]_E[5]_B[128]_T[1000]_I[0,0]_Y[0]_ulatdec_client[4]", 71.27263310127154),
]

def calculate_mean_std(entries):
    means = []
    stds = []
    for i in range(0, len(entries), 5):
        batch = entries[i:i+5]
        values = [entry[1] for entry in batch]
        mean = sum(values) / len(values)
        std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        means.append(mean)
        stds.append(std)
    return means, stds

means, stds = calculate_mean_std(data)

for i in range(len(means)):
    print(f"Mean {i+1}: {means[i]}")
    print(f"Std {i+1}: {stds[i]}")
    print()