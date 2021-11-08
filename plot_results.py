import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import neural_nets as nn
# import scipy.io as sio

import os
num_gpus = 1
total_gpus = 8
start_gpu = 4
cuda = ""
for i in range(num_gpus):
    cuda += str((start_gpu + i) % total_gpus) + ","
print("Adding visible CUDA devices:", cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda


train_sizes = [100, 500, 1000, 10000, 50000]
num_subarrays = 8
test_size = 20000

errors = np.zeros((len(train_sizes), num_subarrays, test_size))
angles = np.zeros((len(train_sizes), num_subarrays, test_size))
labels = np.zeros((len(train_sizes), num_subarrays, test_size, 3))
for idx, train_size in enumerate(train_sizes):
    for subarray in range(num_subarrays):
        # results = np.load(f'results/hybrid_DIS_result_{subarray}_{train_size}.npy', allow_pickle=True)
        # angle = np.load(f'results/hybrid_DIS_angles_{subarray}_{train_size}.npy', allow_pickle=True)
        results = np.load(f'hybrid_DIS_result_{subarray}_{train_size}.npy', allow_pickle=True)
        angle = np.load(f'hybrid_DIS_angles_{subarray}_{train_size}.npy', allow_pickle=True)
        angles[idx, subarray] = angle
        # labels[idx, subarray] = np.load(f'results/hybrid_DIS_labels_{subarray}_{train_size}.npy', allow_pickle=True)
        labels[idx, subarray] = np.load(f'hybrid_DIS_labels_{subarray}_{train_size}.npy', allow_pickle=True)
        error = np.abs(results - angle)
        errors[idx, subarray] = error

for idx, train_size in enumerate(train_sizes):
    # train_IDs = IDs[:train_size]
    for subarray in range(num_subarrays):
        fig = plt.figure()
        plt.scatter(x=labels[idx, subarray, :, 0], y=labels[idx, subarray, :, 1], c=errors[idx, subarray], s=(72./fig.dpi)**2)
        plt.axis('equal')
        cbar = plt.colorbar()
        cbar.set_label("AoA error [degrees]")
        plt.savefig(f'plots/error_scatter_{train_size}_{subarray}.png',
                    bbox_inches='tight', pad_inches=0.1)
        plt.savefig(f'plots/error_scatter_{train_size}_{subarray}.eps',
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()

for idx, train_size in enumerate(train_sizes):
    # train_IDs = IDs[:train_size]
    # for subarray in range(num_subarrays):
    fig = plt.figure()
    plt.scatter(x=labels[idx, subarray, :, 0], y=labels[idx, subarray, :, 1], c=np.mean(errors[idx], axis=0), s=(72./fig.dpi)**2)
    plt.axis('equal')
    cbar = plt.colorbar()
    cbar.set_label("AoA error [degrees]")
    plt.savefig(f'plots/mean_angle_error_scatter_{train_size}.eps',
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'plots/mean_angle_error_scatter_{train_size}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()




errors = np.reshape(errors, (len(train_sizes), num_subarrays*test_size))
print(np.median(errors, axis=1))
print(np.mean(errors, axis=1))

plt.figure()
for i, train_size in enumerate(train_sizes):
    # print(i)
    data = np.array(errors[i])
    data = np.sort(data)
    average = sum(data)/len(data)
    # print(labels[i], average)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    # length = len(data)
    # nb_samples = 200
    # step = length / nb_samples
    # idx = [i*step for i in range(nb_samples-1)]
    # data = np.take(data, idx)
    curve_x = [0]
    curve_x.extend(data)
    curve_x.extend([60])
    curve_y = [0]
    curve_y.extend(p)
    curve_y.extend([1])
    # print(len(data))
    plt.plot(curve_x, curve_y, label=f'{train_size} training samples')
    print(1-np.argwhere(np.array(curve_x) > 3)[0][0]/len(curve_x))
    # print(len(sinrs[i]))
    # plt.hist(sinrs[i], density=True, cumulative=True,
    #          label=labels[i], histtype='step', bins=250)
# print("Histograms created")
font_size = 10
# plt.xlim(0, 60)
# plt.title("CDF of the AoA Error")
plt.ylabel("F(X)")
plt.xlabel('AoA error [degree]')
plt.xticks(fontsize=font_size)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
plt.legend(loc='lower right')
plt.grid(linestyle=':', linewidth=1)
plt.axis([0, 60, -0.1, 1.1])
# plt.show()
# print("Saving plots")
plt.savefig('plots/cdf_aoa.eps',
            bbox_inches='tight', pad_inches=0)
plt.savefig('plots/cdf_aoa.png',
            bbox_inches='tight', pad_inches=0)



# pos_model = load_model("./ete_noisy_pos_model.h5", custom_objects={"tf": tf, "dist": nn.dist})
# aoa_errors = np.load('results/aoa_error_ml_music.npy', allow_pickle=True)
# pos_errors = np.load('results/pos_error_ml_music.npy', allow_pickle=True)
pos_errors = np.load('pos_error_ml_music.npy', allow_pickle=True)
print(np.mean(pos_errors, axis=1))
print(np.median(pos_errors, axis=1))

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

for idx, train_size in enumerate(train_sizes):
    # train_IDs = IDs[:train_size]
    # for subarray in range(num_subarrays):
    fig = plt.figure()
    plt.scatter(x=labels[idx, subarray, :, 0], y=labels[idx, subarray, :, 1], c=pos_errors[idx],
                s=(72./fig.dpi)**2, vmin=0, vmax=80)
    plt.axis('equal')
    cbar = plt.colorbar()
    cbar.set_label("Positioning error [mm]")
    plt.savefig(f'plots/pos_error_scatter_{train_size}.eps',
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'plots/pos_error_scatter_{train_size}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
# aoa_errors = np.reshape(aoa_errors, (len(train_sizes), -1))
# plt.figure()
# for i, train_size in enumerate(train_sizes):
#     data = np.array(aoa_errors[i])
#     data = np.sort(data)
#     average = sum(data)/len(data)
#     p = 1. * np.arange(len(data)) / (len(data) - 1)
#     curve_x = [0]
#     curve_x.extend(data)
#     curve_x.extend([10000])
#     curve_y = [0]
#     curve_y.extend(p)
#     curve_y.extend([1])
#     plt.plot(curve_x, curve_y, label=f'{train_size} training samples')
# font_size = 10
# # plt.title("CDF of the AoA Error")
# plt.ylabel("F(X)")
# plt.xlabel('AoA error [degree]')
# plt.xticks(fontsize=font_size)
# plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
# plt.legend(loc='lower right')
# plt.grid(linestyle=':', linewidth=1)
# plt.axis([0, 60, -0.1, 1.1])
# plt.savefig('plots/cdf_aoa_2.eps',
#             bbox_inches='tight', pad_inches=0)
# plt.savefig('plots/cdf_aoa_2.png',
#             bbox_inches='tight', pad_inches=0)
