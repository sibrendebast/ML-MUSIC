import numpy as np
import data_generator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import neural_nets as nn
import tensorflow as tf

import os
num_gpus = 1
total_gpus = 8
start_gpu = 3
cuda = ""
for i in range(num_gpus):
    cuda += str((start_gpu + i) % total_gpus) + ","
print("Adding visible CUDA devices:", cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

data_folder = '/mnt/myhome/data/measurements_june/'
scenario = 'DIS_lab_LoS'
dataset = 'ultra_dense/'

antenna_positions = np.load(f'{data_folder}{dataset}{scenario}/antenna_positions.npy')
user_positions = np.load(f'{data_folder}{dataset}{scenario}/user_positions.npy')

num_samples = 252004
num_antennas = 64
num_subcarriers = 100
num_subarrays = 8
subarray_size = int(num_antennas/num_subarrays)

IDs = np.arange(num_samples)
np.random.seed(64)
np.random.shuffle(IDs)

val_size = 1000
test_size = 20000
val_IDs = IDs[-test_size-val_size: -test_size]
test_IDs = IDs[-test_size:]
test_labels = user_positions[test_IDs][:, :2]

samples_folder = f'{data_folder}{dataset}{scenario}/samples/'
# val_generator = data_generator.DataGenerator(val_IDs, user_positions, samples_folder)
# test_generator = data_generator.DataGenerator(test_IDs, user_positions, samples_folder, shuffle=False)

# set which size of train sets to use
train_sizes = [100, 500, 1000, 10000, 50000]

for size in train_sizes:
    np.save(f'train_IDs_{size}', IDs[:size])
np.save('test_IDs', test_IDs)

pos_model = load_model("./ete_noisy_pos_model.h5", custom_objects={"tf": tf, "dist": nn.dist})

try:
    nb_runs = np.load('results/number_of_runs.npy')
except Exception as e:
    print('No previous runs found')
    nb_runs = 0

while nb_runs < 100:
    input = Input((subarray_size, num_subcarriers, 2))
    calibrated = nn.build_fully_connected(num_antenna=subarray_size)(input)
    musiced = nn.build_MUSIC_ULA(num_antenna=subarray_size)(calibrated)
    model = Model(inputs=input, outputs=musiced)

    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    results = np.zeros((len(train_sizes), 4))
    for j, train_size in enumerate(train_sizes):
        train_IDs = IDs[:train_size]
        aoa = []
        for subarray in range(num_subarrays):
            print(f"SUBARRAY number {subarray + 1} of {num_subarrays}")
            training_gen = data_generator.DataGenerator_DIS_subarray(train_IDs, user_positions, antenna_positions, samples_folder,
                                                               subarray=subarray)
            test_gen = data_generator.DataGenerator_DIS_subarray(test_IDs, user_positions, antenna_positions, samples_folder, shuffle=False,
                                                          subarray=subarray)

            print("TRAINING MUSIC MODEL...")
            model.fit(training_gen, epochs=15)

            model.save_weights(f'hybrid_model_{train_size}_{nb_runs}_{subarray}.h5')

            print('EVALUATING PERFORMANCE...')
            result = model.predict(test_gen)
            music_result = np.argmax(result, axis=1)

            aoa.append(music_result)
        aoa = np.array(aoa).T/180*3.1415
        test_pos = pos_model.predict(aoa, batch_size=32)
        errors = np.sqrt(np.power(test_labels[:, 0] - test_pos[:, 0], 2) + np.power(test_labels[:, 1] - test_pos[:, 1], 2))
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)
        sorted_errors = np.sort(errors)
        percentile = sorted_errors[int(0.95*len(errors))]
        results[j, 0] = mean_error
        results[j, 1] = median_error
        results[j, 2] = max_error
        results[j, 3] = percentile
        print(results)

    np.save(f"results/sample_efficiency_results_{nb_runs}", results)
    np.save('results/number_of_runs', nb_runs)
    nb_runs += 1
    print(results)
