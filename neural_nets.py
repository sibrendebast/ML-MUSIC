import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Dot, Permute
from scipy.constants import speed_of_light, pi


frequency = 2.61e9
wavelength = speed_of_light/frequency
antenna_spacing = wavelength/2


# metric function
# Computes the 2D distance between the given points
def dist(y_true, y_pred):
    return tf.reduce_mean((
        tf.sqrt(
            tf.square(tf.abs(y_pred[:, 0] - y_true[:, 0]))
            + tf.square(tf.abs(y_pred[:, 1] - y_true[:, 1]))
        )))


# Build a computegraph that calculates the MUSIC spectrum for a given signal.
# The spectrum spans 0 -> 180 degrees, with 1 degree intervals
def build_MUSIC_ULA(num_antenna=64, num_sub=100):

    def to_complex(H):
        H_complex = tf.complex(H[:, :, :, 0], H[:, :, :, 1])
        return H_complex

    def get_noise_subspace(S):
        e, v = tf.compat.v1.linalg.eigh(S)
        v = v[:, :, :num_antenna-2]
        v_h = Lambda(tf.math.conj)(Permute((2, 1))(v))
        v = Dot(axes=(2, 1))([v, v_h])
        return v

    def get_spectrum(noise):
        angles = tf.cast(tf.range(0, 180.5, 1), 'complex64')
        angles = tf.cos(tf.math.scalar_mul(pi / 180, angles))
        angles = tf.reshape(angles, (1, -1))
        geo_array = tf.cast(tf.math.scalar_mul(antenna_spacing, tf.reshape(tf.range(num_antenna, dtype='float64'), (-1, 1))), 'complex64')
        structure = tf.tensordot(geo_array, angles, axes=1)
        res = tf.math.scalar_mul(1j * 2 * pi / wavelength, structure)
        steering_vectors = tf.math.exp(res)
        res = tf.map_fn(lambda noise_b: tf.tensordot(tf.transpose(steering_vectors, conjugate=True), noise_b, 1), noise, dtype='complex64')
        res = tf.map_fn(lambda noise_b: tf.math.abs(tf.tensordot(noise_b, steering_vectors, 1)), res, dtype='float32')
        spectrum = tf.map_fn(lambda spec: tf.linalg.tensor_diag_part(spec), res, dtype='float32')
        spectrum = tf.map_fn(lambda spec: tf.math.scalar_mul(1/num_antenna, spec), spectrum, dtype='float32')
        return spectrum

    music_input = Input((num_antenna, num_sub, 2))
    complex_input = Lambda(to_complex)(music_input)
    H = Lambda(tf.math.conj)(Permute((2, 1))(complex_input))
    S = Dot(axes=(2, 1))([complex_input, H])
    noise = Lambda(get_noise_subspace)(S)
    spectrum = Lambda(get_spectrum)(noise)
    spectrum = Lambda(lambda x: 1 - x)(spectrum)

    model = Model(inputs=music_input, outputs=spectrum)
    return model


# Build the fully connected network to calibrate the signal for th egiven scenario
def build_fully_connected(num_antenna=8, num_sub=100):
    nn_input = Input((num_antenna, num_sub, 2))
    lay = Flatten()(nn_input)
    lay = Dense(num_antenna * num_sub * 2, activation="selu")(lay)
    lay = Reshape((num_antenna, num_sub, 2))(lay)

    model = Model(inputs=nn_input, outputs=lay)
    return model


# Build the fully connected network to combine the AoAs for the distributed antenna arrays to
# a 2D position
def build_aoa_to_position():
    nn_in = Input((8))

    lay = Dense(32, activation='selu')(nn_in)
    lay = Dense(32, activation='selu')(lay)
    lay = Dense(16, activation='selu')(lay)
    lay = Dense(8, activation='selu')(lay)
    lay = Dense(2, activation='linear')(lay)

    model = Model(inputs=nn_in, outputs=lay)
    model.summary()

    return model
