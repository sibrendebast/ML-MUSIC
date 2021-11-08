import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import neural_nets as nn
import scipy.io as sio

import os
num_gpus = 1
total_gpus = 8
start_gpu = 2
cuda = ""
for i in range(num_gpus):
    cuda += str((start_gpu + i) % total_gpus) + ","
print("Adding visible CUDA devices:", cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

model_names = ['hybrid']
train_sizes = [100, 500, 1000, 10000, 50000]
movement = ['Back', 'Middle', 'Front', 'Left', 'Center', 'Right', 'Reference']


nb_runs = np.load('results/number_of_runs.npy')
print('Number of runs done:', nb_runs)
results = np.zeros((nb_runs, len(model_names), len(train_sizes), 4))
for i in range(nb_runs):
    results[i] = np.load(f'results/sample_efficiency_results_{i}.npy')

me = results[:, :, :, 0]
best_results = me.min(axis=0)
best_results_arg = me.argmin(axis=0)

print(best_results)
print(best_results_arg.shape)
print(best_results_arg)




plt.figure()
for i in range(best_results.shape[0]):  # model_names
    plt.plot(best_results[i, :], '-o', label=model_names[i])
plt.legend()
plt.ylabel('Mean error [mm]')
plt.xlabel('Number of training samples')
plt.xticks(np.arange(len(train_sizes)), train_sizes)
plt.ylim(0, 1200)
plt.grid()
plt.savefig("results/sample_efficiency.png", bbox_inches='tight', pad_inches=0)


data_folder = '/mnt/myhome/data/measurements_june/'
scenario = 'DIS_lab_LoS'
dataset = 'nomadic/'
movement = ['Back', 'Middle', 'Front', 'Left', 'Center', 'Right', 'Reference']

antenna_positions = np.load(f'{data_folder}{dataset}{scenario}/antenna_positions.npy')
user_positions = np.load(f'{data_folder}ultra_dense/{scenario}/user_positions.npy')
samples_folder = f'{data_folder}{dataset}{scenario}/samples/'

num_samples = 252004
num_antennas = 64
num_subcarriers = 100
num_subarrays = 8
num_samples_nomadic = 240
num_scenarios = 7
num_users = 4

IDs = np.arange(num_samples)
np.random.seed(64)
np.random.shuffle(IDs)

train_sizes = [100, 500, 1000, 10000, 50000]


pos_model = load_model("./ete_noisy_pos_model.h5", custom_objects={"tf": tf, "dist": nn.dist})
# make the models to test

samples = np.zeros((num_scenarios, num_users, num_samples_nomadic, num_antennas, num_subcarriers, 2))
for scenario in range(num_scenarios):
    for user in range(num_users):
        for sample in range(num_samples_nomadic):
            id = scenario * 10000 + user * 1000 + sample
            channel = np.load(samples_folder + "channel_measurement_" + str(id).zfill(6) + '.npy')
            samples[scenario, user, sample, :, :, 0] = channel.real
            samples[scenario, user, sample, :, :, 1] = channel.imag

results = np.zeros((len(train_sizes), num_scenarios, num_users, num_samples_nomadic, 2))
for j, train_size in enumerate(train_sizes):
    pos_model = load_model(f"./pos_model_{train_size}.h5", custom_objects={"tf": tf, "dist": nn.dist})
    print(model_names[i], train_size)
    models = []
    for subarray in range(num_subarrays):
        input = Input((8, num_subcarriers, 2))
        calibrated = nn.build_fully_connected(num_antenna=8)(input)
        musiced = nn.build_MUSIC_ULA(num_antenna=8)(calibrated)
        model = Model(inputs=input, outputs=musiced)
        models.append(model)
        models[subarray].load_weights(f'hybrid_model_{train_size}_{best_results_arg}_{subarray}.h5')
    for scenario in range(num_scenarios):
        for user in range(num_users):
            aoas = np.zeros((num_samples_nomadic, num_subarrays))
            for subarray in range(num_subarrays):
                aoas[:, subarray] = np.argmax(models[subarray].predict(samples[scenario, user, :, subarray*8:(subarray+1)*8], batch_size=16), axis=1)/180*3.14
            results[j, scenario, user] = pos_model.predict(aoas, batch_size=16)


np.save("results_nomadic_hybrid",  results)

nomadic_results = np.load('results_nomadic_hybrid.npy')


def dist(pos0, pos1):
    return np.sqrt((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)


user_ref_positions = np.zeros((4, 2))
for i in range(4):
    user_ref_positions[i] = user_positions[31500 + 63001*i, :2]

# for j, scen in enumerate(movement):
# plt.figure()
# for i in range(4):
#     plt.scatter(user_ref_positions[i, 0], user_ref_positions[i, 1])
# plt.savefig(f'plots/nomadic_check_label.png')

print('shape of results', nomadic_results.shape)


diff_results = np.zeros(nomadic_results.shape[:-1])
for model in range(len(models)):
    for train_size in range(len(train_sizes)):
        for scen in range(num_scenarios):
            for user in range(num_users):
                for sample in range(num_samples_nomadic):
                    diff_results[train_size, scen, user, sample] = dist(user_ref_positions[user], nomadic_results[train_size, scen, user, sample])


me = np.zeros((diff_results.shape[0]))
for i in range(diff_results.shape[0]):
    me[i] = np.mean(diff_results[i])

print(me)

plt.figure()
plt.plot(me, '-o')
plt.legend()
plt.ylabel('Mean error [mm]')
plt.xlabel('Number of training samples')
plt.xticks(np.arange(len(train_sizes)), train_sizes)
plt.ylim(0, 1300)
plt.savefig("plots/results/hybrid_nomadic_mean_error.png", bbox_inches='tight', pad_inches=0)

positions = np.zeros((len(train_sizes), len(model_names)))
for i in range(len(train_sizes)):
    for j in range(len(model_names)):
        positions[i, j] = i*len(model_names) - 1.2 + 0.8*j

colors = ['C0', 'C1', 'C2', 'C3']

plt.figure(figsize=(10, 4))
for i in range(len(train_sizes)):
    for j in range(len(model_names)):
        box = plt.boxplot(diff_results[j, i].flatten(), positions=[positions[i, j]], showfliers=False,
                          widths=[0.75])
        for item in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(box[item], color=colors[j])
        # plt.setp(box['boxes'], facecolor=colors[j])
tick = np.arange(0, len(train_sizes)*len(model_names), len(model_names))
plt.xticks(tick, train_sizes)
plt.savefig("plots/results/hybrid_nomadic_boxplot.png", bbox_inches='tight', pad_inches=0)


for j, train_size in enumerate(train_sizes):
    print(j)
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    fig.suptitle(f'Model: Hybrid; Number of training samples: {train_size}')
    for i in range(3, 7):
        # plt.figure()
        axs[i-3].plot(diff_results[j, i, 0], label='User 1')
        axs[i-3].plot(diff_results[j, i, 1], label='User 2')
        axs[i-3].plot(diff_results[j, i, 2], label='User 3')
        axs[i-3].plot(diff_results[j, i, 3], label='User 4')
        axs[i-3].set_title(movement[i])
        axs[i-3].set_ylim(0, 750)
        axs[i-3].set_xlim(0, 120)
        if i == 6:
            axs[i-3].legend()
    fig.savefig(f'plots/results/hybrid_nomadic_{train_size}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
