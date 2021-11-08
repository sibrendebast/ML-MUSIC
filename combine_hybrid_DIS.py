import numpy as np
import neural_nets as nn
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import data_generator


# import os
# num_gpus = 1
# total_gpus = 8
# start_gpu = 3
# cuda = ""
# for i in range(num_gpus):
#     cuda += str((start_gpu + i) % total_gpus) + ","
# print("Adding visible CUDA devices:", cuda)
# os.environ["CUDA_VISIBLE_DEVICES"] = cuda


num_samples = 252004
num_antennas = 8
num_sub = 100
num_subarrays = 8


IDs = np.array([x for x in range(num_samples)])
np.random.seed(64)
np.random.shuffle(IDs)
# val_size = 1000
train_size = 1000
train_IDs = IDs[:train_size]
test_size = 20000
# val_IDs = IDs[-test_size-val_size: -test_size]
test_IDs = IDs[-test_size:]

data_folder = './data/'
scenario = 'DIS_lab_LoS'
name = 'ultra_dense/'
dataset = f'{data_folder}{name}{scenario}/samples/'
labels = np.load(f'{data_folder}{name}{scenario}/user_positions.npy')
antenna_positions = np.load(f'{data_folder}{name}{scenario}/antenna_positions.npy')

test_labels = labels[test_IDs][:, :2]

results = []

for subarray in range(num_subarrays):
    result = np.load(f'hybrid_DIS_result_{subarray}.npy')/180*3.1415
    results.append(result)

results = np.array(results).T
print(results.shape)

# pretrain model
pretrain = 0
if pretrain:
    pos_model = nn.build_aoa_to_position()
    pos_model.compile('Adam', nn.dist)
    train_noise = data_generator.DataGenerator_DIS_noisy_aoa(np.arange(252004), labels, antenna_positions)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)
    mc = ModelCheckpoint(f'pos_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
    pos_model.fit(train_noise, callbacks=[es, mc], epochs=1000)

pos_model = load_model("./pos_model.h5", custom_objects={"tf": tf, "dist": nn.dist})
test_pos = pos_model.predict(results, batch_size=32)

errors = np.sqrt(np.power(test_labels[:, 0] - test_pos[:, 0], 2) + np.power(test_labels[:, 1] - test_pos[:, 1], 2))
mean_error = np.mean(errors)
median_error = np.median(errors)
max_error = np.max(errors)
sorted_errors = np.sort(errors)
percentile = sorted_errors[int(0.95*len(errors))]

print(f"The mean error of the model is {mean_error}")
print(f"The max error of the model is {max_error}")
print(f"The median error of the model is {median_error}")
print(f"The 95th percentile of the model is {percentile}")
