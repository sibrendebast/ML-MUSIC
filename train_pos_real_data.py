import numpy as np
import data_generator
import neural_nets as nn
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf

import os
num_gpus = 1
total_gpus = 8
start_gpu = 7
cuda = ""
for i in range(num_gpus):
    cuda += str((start_gpu + i) % total_gpus) + ","
print("Adding visible CUDA devices:", cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda

data_folder = '/mnt/myhome/data/measurements_june/'
# data_folder = '/home/sdebast/data/measurements_june/'
scenario = 'DIS_lab_LoS'
dataset = 'ultra_dense/'

train_sizes = [100, 500, 1000, 10000, 50000]
num_samples = 252004
num_arrays = 8
IDs = np.array([x for x in range(num_samples)])
np.random.seed(64)
np.random.shuffle(IDs)
test_size = 20000
test_IDs = IDs[-test_size:]

antenna_positions = np.load(f'{data_folder}{dataset}{scenario}/antenna_positions.npy')
user_positions = np.load(f'{data_folder}{dataset}{scenario}/user_positions.npy')
pos_model = load_model("./ete_noisy_pos_model.h5", custom_objects={"tf": tf, "dist": nn.dist})

for idx, train_size in enumerate(train_sizes):
    pos_model = load_model("./ete_noisy_pos_model.h5", custom_objects={"tf": tf, "dist": nn.dist})
    train_set = []
    train_labels = []
    for i in range(num_arrays):
        train_set.append(np.load(f'hybrid_DIS_aoa_train_{i}_{train_size}.npy'))
        train_labels.append(np.load(f'hybrid_DIS_labels_train_{i}_{train_size}.npy'))
    train_set = np.array(train_set).T/180*np.pi
    train_labels = np.array(train_labels, dtype=np.float32)[0, :, :2]

    print(train_set.shape)
    print(train_labels.shape)

    pos_model.fit(x=train_set, y=train_labels, epochs=100)

    pos_model.save(f'pos_model_{train_size}')


# pos_model = nn.build_aoa_to_position()
# pos_model.compile('Adam', nn.dist)
# train_noise = data_generator.DataGenerator_DIS_noisy_aoa(np.arange(252004), user_positions, antenna_positions)
# es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)
# mc = ModelCheckpoint(f'ete_noisy_pos_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
# pos_model.fit(train_noise, callbacks=[es, mc], epochs=1000)
