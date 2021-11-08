import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import neural_nets as nn
import scipy.io as sio
import data_generator

import os
num_gpus = 1
total_gpus = 8
start_gpu = 3
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
best_results_arg = me.argmin(axis=0)[0]

print(best_results)
print(best_results_arg.shape)
print(best_results_arg)


data_folder = './data/'
scenario = 'DIS_lab_LoS'
name = 'ultra_dense/'
dataset = f'{data_folder}{name}{scenario}/samples/'
labels = np.load(f'{data_folder}{name}{scenario}/user_positions.npy')
antenna_labels = np.load(f'{data_folder}{name}{scenario}/antenna_positions.npy')

num_samples = 252004
num_antennas = 64
num_subcarriers = 100
num_subarrays = 8


train_sizes = [100, 500, 1000, 10000, 50000]

IDs = np.arange(num_samples)
np.random.seed(64)
np.random.shuffle(IDs)

val_size = 1000
test_size = 20000
val_IDs = IDs[-test_size-val_size: -test_size]
test_IDs = IDs[-test_size:]
test_labels = labels[test_IDs][:, :2]


# make the models to test
test_angles = np.load(f'results/hybrid_DIS_angles_0_100.npy', allow_pickle=True)
test_angles = np.arctan2(test_labels[:, 1], test_labels[:, 0])/np.pi*180
print(test_angles.shape)

results = np.zeros((len(train_sizes), test_size, 2))
pos_errors = np.zeros((len(train_sizes), test_size))
aoas = np.zeros((len(train_sizes), test_size, num_subarrays))
aoa_errors = np.zeros((len(train_sizes), test_size, num_subarrays))
for j, train_size in enumerate(train_sizes):
    pos_model = load_model(f"./pos_model_{train_size}", custom_objects={"tf": tf, "dist": nn.dist})
    print('train size', train_size)
    models = []
    for subarray in range(num_subarrays):
        input = Input((8, num_subcarriers, 2))
        calibrated = nn.build_fully_connected(num_antenna=8)(input)
        musiced = nn.build_MUSIC_ULA(num_antenna=8)(calibrated)
        model = Model(inputs=input, outputs=musiced)
        models.append(model)
        models[subarray].load_weights(f'hybrid_model_{train_size}_{best_results_arg[j]}_{subarray}.h5')
    # aoas = np.zeros((num_samples_nomadic, num_subarrays))
    for subarray in range(num_subarrays):
        test_gen = data_generator.DataGenerator_DIS_subarray(test_IDs, labels, antenna_labels, dataset, shuffle=False,
                                                             subarray=subarray)
        aoas[j, :, subarray] = np.argmax(models[subarray].predict(test_gen), axis=1)
        aoa_errors[j, :, subarray] = np.abs(aoas[j, :, subarray] - test_angles)
    results[j] = pos_model.predict(aoas[j]/180*3.14, batch_size=16)
    pos_errors[j] = np.sqrt((results[j, :, 0] - test_labels[:, 0])**2 + (results[j, :, 1] - test_labels[:, 1])**2)

np.save('pos_error_ml_music', pos_errors)
np.save('aoa_error_ml_music', aoa_errors)


plt.figure()
for i, train_size in enumerate(train_sizes):
    data = np.array(pos_errors[i])
    data = np.sort(data)
    average = sum(data)/len(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    curve_x = [0]
    curve_x.extend(data)
    curve_x.extend([10000])
    curve_y = [0]
    curve_y.extend(p)
    curve_y.extend([1])
    plt.plot(curve_x, curve_y, label=f'{train_size} training samples')
font_size = 10
# plt.title("CDF of the AoA Error")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
plt.xticks(fontsize=font_size)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
plt.legend(loc='lower right')
plt.grid(linestyle=':', linewidth=1)
plt.axis([0, 500, -0.1, 1.1])
plt.savefig('plots/cdf_pos.eps',
            bbox_inches='tight', pad_inches=0)
plt.savefig('plots/cdf_pos.png',
            bbox_inches='tight', pad_inches=0)


aoa_errors = np.reshape(aoa_errors, (len(train_sizes), -1))
plt.figure()
for i, train_size in enumerate(train_sizes):
    data = np.array(aoa_errors[i])
    data = np.sort(data)
    average = sum(data)/len(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    curve_x = [0]
    curve_x.extend(data)
    curve_x.extend([10000])
    curve_y = [0]
    curve_y.extend(p)
    curve_y.extend([1])
    plt.plot(curve_x, curve_y, label=f'{train_size} training samples')
font_size = 10
# plt.title("CDF of the AoA Error")
plt.ylabel("F(X)")
plt.xlabel('AoA error [degree]')
plt.xticks(fontsize=font_size)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
plt.legend(loc='lower right')
plt.grid(linestyle=':', linewidth=1)
plt.axis([0, 500, -0.1, 1.1])
plt.savefig('plots/cdf_aoa.eps',
            bbox_inches='tight', pad_inches=0)
plt.savefig('plots/cdf_aoa.png',
            bbox_inches='tight', pad_inches=0)
