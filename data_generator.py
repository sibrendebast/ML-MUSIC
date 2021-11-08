import numpy as np
from tensorflow.keras.utils import Sequence
import util


class DataGenerator_DIS_subarray(Sequence):
    # 'Generates data for Keras'
    def __init__(self, list_IDs, labels, antenna_labels, data_path, batch_size=32, num_antennas=8,
                 num_subc=100, n_channels=2, shuffle=True, subarray=None):
        # 'Initialization'
        self.dim = (num_antennas, num_subc)
        if subarray is None:
            self.antennas = [x for x in range(8)]
            self.origin = (0, 0)
            self.direction = (antenna_labels[0] - antenna_labels[7])[:2]
            self.direction = self.direction/np.linalg.norm(self.direction)
        else:
            assert (subarray < 8 and subarray >= 0)
            self.antennas = [x for x in range(subarray*8, (subarray + 1)*8)]
            self.origin = np.mean(antenna_labels[self.antennas], axis=0)
            self.direction = (antenna_labels[subarray*8] - antenna_labels[subarray*8 + 7])[:2]
            self.direction = self.direction/np.linalg.norm(self.direction)
            # print('origin', self.origin)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data_path = data_path
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 181), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            sample = np.load(self.data_path + "channel_measurement_" + str(ID).zfill(6) + '.npy')
            # print(X[i, :, :, 0].shape)
            # print(sample.real.shape)
            X[i, :, :, 0] = sample.real[self.antennas, :]
            X[i, :, :, 1] = sample.imag[self.antennas, :]

            # Store class
            # label_degree = np.arctan2(self.labels[ID][1], self.labels[ID][0] - self.origin[0])/np.pi*180
            position = self.labels[ID]
            aoa = util.angle_array_point(self.direction, self.origin, position)
            # ohe[int(aoa/np.pi*180)] = 1
            label = np.zeros((181))
            label[int(aoa/np.pi*180)] = 1
            y[i] = label

        return X, y


class DataGenerator_DIS_noisy_aoa(Sequence):
    # 'Generates data for Keras'
    def __init__(self, list_IDs, labels, antenna_labels, batch_size=32, shuffle=True, subarray=None):
        # 'Initialization'
        self.num_subarrays = 8
        self.array_centers = np.zeros((self.num_subarrays, 2))
        self.array_directions = np.zeros((self.num_subarrays, 2))
        for i in range(self.num_subarrays):
            self.array_centers[i] = np.mean(antenna_labels[i*8:(i+1)*8], axis=0)[:2]
            self.array_directions[i] = (antenna_labels[i*8] - antenna_labels[i*8 + 7])[:2]
            self.array_directions[i] = self.array_directions[i]/np.linalg.norm(self.array_directions[i])
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = 2
        self.shuffle = shuffle
        self.max_blockages = 4
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 8))
        y = np.empty((self.batch_size, 2))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            position = self.labels[ID]
            for j in range(self.num_subarrays):
                X[i, j] = util.angle_array_point(self.array_directions[j], self.array_centers[j], position)
            y[i] = position[:2]

            # add noise to angles
            nb_blockages = np.random.exponential(1.2)
            nb_blockages = int(nb_blockages)
            if nb_blockages > self.max_blockages:
                nb_blockages = self.max_blockages
            blocked_antennas = np.arange(8)
            np.random.shuffle(blocked_antennas)
            blocked_antennas = blocked_antennas[:nb_blockages]
            for j in range(nb_blockages):
                X[i, blocked_antennas[j]] = X[i, blocked_antennas[j]] + np.random.normal(0, 0.7)

        return X, y
