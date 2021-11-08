import numpy as np
import neural_nets as nn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import data_generator

# Define which GPU to use (when using multiple GPU workstation)
# import os
# num_gpus = 1
# total_gpus = 8
# start_gpu = 3
# cuda = ""
# for i in range(num_gpus):
#     cuda += str((start_gpu + i) % total_gpus) + ","
# print("Adding visible CUDA devices:", cuda)
# os.environ["CUDA_VISIBLE_DEVICES"] = cuda

# Core parameters of the experiment
num_samples = 252004
num_antennas = 8
num_sub = 100
num_subarrays = 8

# Def ine where the dataset can be found
data_folder = './data/'
scenario = 'DIS_lab_LoS'
name = 'ultra_dense/'
dataset = f'{data_folder}{name}{scenario}/samples/'
labels = np.load(f'{data_folder}{name}{scenario}/user_positions.npy')
antenna_labels = np.load(f'{data_folder}{name}{scenario}/antenna_positions.npy')


input = Input((num_antennas, num_sub, 2))
calibrated = nn.build_fully_connected(num_antenna=num_antennas)(input)
musiced = nn.build_MUSIC_ULA(num_antenna=num_antennas)(calibrated)
model = Model(inputs=input, outputs=musiced)

model.compile(optimizer='Adam', loss='mse')
model.summary()


IDs = np.array([x for x in range(num_samples)])
np.random.seed(64)
np.random.shuffle(IDs)
# val_size = 1000
train_sizes = [100, 500, 1000, 10000, 50000]
nb_epochs = [50, 25, 10, 10, 10]
# train_IDs = IDs[:train_size]
test_size = 20000
# val_IDs = IDs[-test_size-val_size: -test_size]
test_IDs = IDs[-test_size:]

for idx, train_size in enumerate(train_sizes):
    train_IDs = IDs[:train_size]
    for subarray in range(num_subarrays):
        print(f"SUBARRAY number {subarray + 1} of {num_subarrays}")
        training_gen = data_generator.DataGenerator_DIS_subarray(train_IDs, labels, antenna_labels, dataset,
                                                                 num_antennas=num_antennas, subarray=subarray)
        test_gen = data_generator.DataGenerator_DIS_subarray(test_IDs, labels, antenna_labels, dataset, shuffle=False,
                                                             num_antennas=num_antennas, subarray=subarray)

        print("TRAINING MUSIC MODEL...")
        model.fit(training_gen, epochs=nb_epochs[idx])

        # model.save_weights(f'hybrid_model_{subarray}.h5')

        print('EVALUATING PERFORMANCE...')
        result = model.predict(test_gen)
        music_result = np.argmax(result, axis=1)
        test_labels = labels[test_IDs][:len(music_result)]

        print("Getting labels...")
        test_angles = []
        for i in range(len(test_gen)):
            data = test_gen[i]
            label = data[1]
            test_angles.extend(np.argmax(label, axis=1))
        error = np.abs(music_result - test_angles)

        # save the results
        np.save(f'results/hybrid_DIS_result_{subarray}_{train_size}', music_result)
        np.save(f'results/hybrid_DIS_angles_{subarray}_{train_size}', test_angles)
        np.save(f'results/hybrid_DIS_labels_{subarray}_{train_size}', test_labels)

        # print the results for some feedback
        print(f'angle of arrival following MUSIC: {music_result}')
        print(f'angle of arrival error: {error}')
        print(f'angle of arrival mean error: {np.mean(error)} degrees')

        # Generate the dataset with training data and corresponing AoA label
        print("generating train data for pos model")
        training_gen = data_generator.DataGenerator_DIS_subarray(train_IDs, labels, antenna_labels, dataset,
                                                                 num_antennas=num_antennas, subarray=subarray, shuffle=False)
        train_aoa = model.predict(training_gen)
        train_aoa = np.argmax(train_aoa, axis=1)
        train_labels = labels[train_IDs][:len(train_aoa)]

        # Save the training set with estimated AoA label
        np.save(f'results/hybrid_DIS_aoa_train_{subarray}_{train_size}', train_aoa)
        np.save(f'results/hybrid_DIS_labels_train_{subarray}_{train_size}', train_labels)
