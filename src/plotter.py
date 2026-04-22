import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

epoch_data = pd.read_csv('../csvs/20230527-153653_fmnist_num_rounds.csv')
client_epoch_data = pd.read_csv('../csvs/results_client_epochs.csv')

params_full = pd.read_csv('../csvs/model_fmnist_R[15]_K[10]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_full.csv')
params_usplit = pd.read_csv('../csvs/model_fmnist_R[15]_K[10]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_usplit.csv')
params_udec = pd.read_csv('../csvs/model_fmnist_R[15]_K[10]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_udec.csv')
params_ulatdec = pd.read_csv('../csvs/model_fmnist_R[15]_K[10]_C[1]_E[5]_B[128]_T[1000]_I[1,0]_Y[0]_ulatdec.csv')


def plot_skew_dict():
    data_l = {
        4: {0: 866, 1: 325, 2: 816, 3: 87, 4: 2466, 5: 5, 6: 373, 7: 844, 8: 3707, 9: 2957},
        3: {0: 2, 1: 3979, 2: 1959, 3: 149, 4: 0, 5: 2265, 6: 5484, 7: 0, 8: 0, 9: 0},
        2: {0: 243, 1: 121, 2: 2622, 3: 4584, 4: 1010, 5: 2481, 6: 3, 7: 1293, 8: 0, 9: 0},
        1: {0: 1, 1: 259, 2: 225, 3: 244, 4: 2378, 5: 72, 6: 44, 7: 919, 8: 728, 9: 3043},
        0: {0: 4888, 1: 1316, 2: 378, 3: 936, 4: 146, 5: 1177, 6: 96, 7: 2944, 8: 1565, 9: 0}
    }

    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data_l).T
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt="d", cbar=True, vmin=0, vmax=5500)
    plt.xlabel("Label")
    plt.ylabel("Client")

    # Export the plot as PDF
    plt.savefig('data_dist_l.pdf', format="pdf")

    # Display the plot
    plt.show()
    plt.close()


def plot_skew_dicts():
    data = {0: {0: 1141, 1: 1219, 2: 1258, 3: 1135, 4: 1195, 5: 1193, 6: 1236, 7: 1261, 8: 1148, 9: 1214},
            1: {0: 1228, 1: 1182, 2: 1183, 3: 1199, 4: 1213, 5: 1261, 6: 1177, 7: 1190, 8: 1225, 9: 1142},
            2: {0: 1198, 1: 1182, 2: 1220, 3: 1180, 4: 1213, 5: 1160, 6: 1232, 7: 1204, 8: 1219, 9: 1192},
            3: {0: 1234, 1: 1208, 2: 1168, 3: 1230, 4: 1174, 5: 1190, 6: 1165, 7: 1187, 8: 1200, 9: 1244},
            4: {0: 1199, 1: 1209, 2: 1171, 3: 1256, 4: 1205, 5: 1196, 6: 1190, 7: 1158, 8: 1208, 9: 1208}}

    data_l = {0: {0: 4888, 1: 1316, 2: 378, 3: 936, 4: 146, 5: 1177, 6: 96, 7: 2944, 8: 1565, 9: 0},
              1: {0: 1, 1: 259, 2: 225, 3: 244, 4: 2378, 5: 72, 6: 44, 7: 919, 8: 728, 9: 3043},
              2: {0: 243, 1: 121, 2: 2622, 3: 4584, 4: 1010, 5: 2481, 6: 3, 7: 1293, 8: 0, 9: 0},
              3: {0: 2, 1: 3979, 2: 1959, 3: 149, 4: 0, 5: 2265, 6: 5484, 7: 0, 8: 0, 9: 0},
              4: {0: 866, 1: 325, 2: 816, 3: 87, 4: 2466, 5: 5, 6: 373, 7: 844, 8: 3707, 9: 2957}}

    data_q = {0: {0: 52, 1: 58, 2: 59, 3: 41, 4: 45, 5: 54, 6: 50, 7: 59, 8: 51, 9: 60},
              1: {0: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 3, 6: 1, 7: 1, 8: 3, 9: 1},
              2: {0: 4605, 1: 4598, 2: 4624, 3: 4558, 4: 4612, 5: 4610, 6: 4644, 7: 4648, 8: 4600, 9: 4588},
              3: {0: 1191, 1: 1177, 2: 1169, 3: 1244, 4: 1206, 5: 1182, 6: 1165, 7: 1163, 8: 1216, 9: 1185},
              4: {0: 152, 1: 166, 2: 147, 3: 155, 4: 136, 5: 151, 6: 141, 7: 129, 8: 130, 9: 166}}

    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)
    df_l = pd.DataFrame(data_l)
    df_q = pd.DataFrame(data_q)

    # Create a grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Plot heatmap 1 on the first subplot
    axes[0].set_ylabel("Label")
    heatmap = sns.heatmap(df, annot=True, cmap="viridis", fmt="d", ax=axes[0], cbar=False, vmin=0, vmax=5500)
    axes[0].set_xlabel("Client")
    axes[0].set_title("IID")

    # Plot heatmap 2 on the first subplot
    sns.heatmap(df_l, annot=True, cmap="viridis", fmt="d", ax=axes[1], cbar=False, vmin=0, vmax=5500)
    axes[1].set_ylabel("Label")
    axes[1].set_xlabel("Client")
    axes[1].set_title("l-skew")

    # Plot heatmap 3 on the second subplot
    sns.heatmap(df_q, annot=True, cmap="viridis", fmt="d", ax=axes[2], vmin=0, vmax=5500)
    axes[2].set_ylabel("Label")
    axes[2].set_xlabel("Client")
    axes[2].set_title("q-skew")

    sns.set_theme(style="darkgrid")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Export the plot as PDF
    plt.savefig('data_dists.pdf', format="pdf")

    # Display the plot
    plt.show()
    plt.close()


def plot_number_of_params():
    num_rounds = 15
    x = np.arange(1, num_rounds + 1).reshape(-1, 1)
    y_full = params_full['params_shared'].cumsum()
    print(f'full: {y_full}')
    y_usplit = params_usplit['params_shared'].cumsum()
    print(f'usplit: {y_usplit}')
    y_udec = params_udec['params_shared'].cumsum()
    print(f'udec: {y_udec}')
    y_ulatdec = params_ulatdec['params_shared'].cumsum()
    print(f'ulatdec: {y_ulatdec}')

    sns.set_theme(style="darkgrid")
    sns.lineplot(x=x.flatten(), y=y_full, marker='o', label="Full")
    sns.lineplot(x=x.flatten(), y=y_usplit, marker='o', label="USplit")
    sns.lineplot(x=x.flatten(), y=y_udec, marker='o', label="UDec")
    sns.lineplot(x=x.flatten(), y=y_ulatdec, marker='o', label="ULatDec")

    plt.xlabel('Global Training Round')
    plt.ylabel('Cumulative Number of Communicated Parameters')

    # Set x-axis ticks for every value
    plt.xticks(x.flatten())

    # Add legend
    plt.legend()

    # Export the plot as PDF
    plt.savefig('num_params.pdf', format="pdf")

    # Display the plot
    plt.show()
    plt.close()


def plot_client_epochs():
    # Split the DataFrame into groups of 3
    groups = [client_epoch_data.iloc[i:i + 3] for i in range(0, len(client_epoch_data), 3)]

    epochs = {
        0: "1",
        1: "2",
        2: "3",
        3: "5",
        4: "8",
    }

    sns.set_theme(style="darkgrid")

    # Create a new line plot for each group
    for i, group in enumerate(groups):
        # Extract x and y values from the data
        sns.lineplot(data=group, x='model_name', y='mean', marker='o',
                     label=f'E = {epochs[i]}')
        plt.fill_between(group['model_name'], group['mean'] - group['std'], group['mean'] + group['std'], alpha=0.2)

    # Plot a horizontal line artefact threshold
    artefact_threshold_mu = 72.24
    plt.axhline(y=artefact_threshold_mu, color='grey', linestyle='--',
                label="artifact threshold")

    non_fed_mu = 42.53
    plt.axhline(y=non_fed_mu, color='black', linestyle='--',
                label="non-fed baseline")

    plt.xlabel('Number of Clients')
    plt.ylabel('FID score')

    # Add legend
    plt.legend()

    # Export the plot as PDF
    plt.savefig('client_epochs.pdf', format="pdf")

    # Display the plot
    plt.show()
    plt.close()


def plot_number_of_epochs():
    # Read the CSV file

    # Extract x and y values from the data
    x = epoch_data['model_name']
    y = epoch_data['fid']

    # Reshape the x values to a 2D array
    x = np.array(x).reshape(-1, 1)

    plt.plot(x, y, '-o', label="FID score trail")

    # Plot a horizontal line at the image fidelity boundary
    plt.axhline(y=epoch_data.loc[epoch_data['model_name'] == 10, 'fid'].values[0], color='grey', linestyle='--',
                label="image fidelity boundary")

    plt.xticks(epoch_data['model_name'])
    plt.xlabel('Number of Training Epochs')
    plt.ylabel('FID score')
    # plt.title('FID scores for different number of Epochs on Non-Federated Baseline')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()


plot_number_of_epochs()
plot_client_epochs()
plot_number_of_params()
plot_skew_dict()
plot_skew_dicts()
