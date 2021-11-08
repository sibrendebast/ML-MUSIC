import numpy as np
import data_generator
import neural_nets as nn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
scenario = 'DIS_lab_LoS'
dataset = 'ultra_dense/'

antenna_positions = np.load(f'{data_folder}{dataset}{scenario}/antenna_positions.npy')
user_positions = np.load(f'{data_folder}{dataset}{scenario}/user_positions.npy')


pos_model = nn.build_aoa_to_position()
pos_model.compile('Adam', nn.dist)
train_noise = data_generator.DataGenerator_DIS_noisy_aoa(np.arange(252004), user_positions, antenna_positions)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)
mc = ModelCheckpoint(f'pos_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
pos_model.fit(train_noise, callbacks=[es, mc], epochs=1000)
