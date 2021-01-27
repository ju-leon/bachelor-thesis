import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from mendeleev import element

import sklearn
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, concatenate
from keras.utils import np_utils
from keras import Model
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from kerastuner.tuners import Hyperband
import kerastuner as kt

from feature_generation.definitions import Point, Atom
from feature_generation.create_slices import slice_to_contour, slice_to_map
from feature_generation.contour_descriptor import fourier_descriptor
from feature_generation.alignment import align_catalyst

import time
import csv


radii = dict()


def get_radius(atom):
    """
    Getting radii from mendeleev seems to be really slow. Buffer them to speed up exectuion
    """
    if atom in radii:
        return radii[atom]
    else:
        radius = element(atom).vdw_radius / 100
        radii[atom] = radius
        return radius


def read_from_file(file):
    atoms = []

    for lineidx, line in enumerate(open(file, "r")):
        if lineidx >= 2:
            elem = line.split()[0].capitalize()
            location = np.array([float(line.split()[1]), float(
                line.split()[2]), float(line.split()[3])])
            radius = get_radius(elem)
            atoms.append(Atom(elem, location, radius))

    return atoms


def generate_slices(atoms, layer_height, z_start, z_end, resolution, channels, bitmap):
    aligned_atoms = align_catalyst(atoms)
    if bitmap:
        slices = slice_to_map(aligned_atoms, layer_height,
                              z_start, z_end, resolution, channels)
    else:
        slices = slice_to_contour(aligned_atoms, layer_height,
                                  z_start, z_end, resolution, channels)
    return slices


def generate_fourier_descriptions(slices, order):
    """
    Generates an invariant feature vector from fourier coefficients
    """
    fourier = []
    for slice in slices:
        channels = []
        for channel in slice:
            channels.append(fourier_descriptor(channel, order))

        fourier.append(np.dstack(channels))

    fourier = np.array(fourier)
    fourier = fourier.reshape((fourier.shape[0], fourier.shape[2], fourier.shape[3]))

    reference_angle = 0
    for x in range(len(fourier)):
        if fourier[x][-1][0] > reference_angle:
            reference_angle = fourier[x][-1][0]
    
    for slice in fourier:
        for channel in slice[-2]:
            if channel != 0:
                channel = (channel - reference_angle) % (2 * np.pi) 

    return fourier


def num_element(atoms, element):
    x = 0
    for atom in atoms:
        if atom.element == element:
            x += 1

    return x


def get_model(hp):
    input_shape = (15, 40, 1)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    filter_size_limit = [4, 6, 4]
    conv_blocks = hp.Int('conv_blocks', 0, 2, default=1)
    for i in range(conv_blocks):
        filters = hp.Int('filters_' + str(i), 2, 12, step=2)

        filter_size = hp.Int('filter_size_' + str(i), 2,
                             filter_size_limit[i], step=1)

        x = tf.keras.layers.Conv2D(filters, [filter_size, 1])(x)

        dropout = hp.Float('conv_dropout_' + str(i), 0,
                           0.6, step=0.1, default=0.2)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Flatten()(x)

    if conv_blocks > 0:
        x = tf.keras.layers.BatchNormalization()(x)

    for i in range(hp.Int('hidden_layers', 1, 3, default=3)):
        size = hp.Int('hidden_size_' + str(i), 10, 300, step=40)
        reg = hp.Float('hidden_reg_' + str(i), 0,
                       0.06, step=0.01, default=0.02)
        dropout = hp.Float('hidden_dropout_' + str(i),
                           0, 0.5, step=0.1, default=0.2)

        x = tf.keras.layers.Dense(size, activation="relu",
                                  kernel_regularizer=regularizers.l2(reg))(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        norm = hp.Choice('hidden_batch_norm_' + str(i), values=[True, False])

        if norm:
            x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1, kernel_regularizer='l2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-6, 1e-4, sampling='log')),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def main():
    """
    Extracts all features from files in given directory. Saves extracted features as numpy array in out location.
    """
    parser = argparse.ArgumentParser(
        description='Generate rotationally invariant features from catalysts using fourier descriptors')
    parser.add_argument(
        'data_dir', help='Directory with xyz files for feature generation')

    args = parser.parse_args()

    barriers = dict()
    with open(args.data_dir + 'vaskas_features_properties_smiles_filenames.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            barriers[row[93]] = float(row[91])


    features_maps = []
    labels = []

    layer_height = 1

    for f in tqdm(os.listdir(args.data_dir + "/coordinates_molSimplify/")):
        if f.endswith(".xyz"):
            atoms = read_from_file(args.data_dir + "/coordinates_molSimplify/" + f)

            slices = generate_slices(atoms, layer_height,
                                     -10, 5, 0.1, ["X"], False)

            feature_map = generate_fourier_descriptions(slices, 10)
            features_maps.append(feature_map)
            
            labels.append(barriers[f[:-4]])
            

    features_maps = np.array(features_maps)
    # Scale coefficents
    fourierScaler = StandardScaler()
    fourierScaler.fit(features_maps.reshape(len(features_maps), -1))
    features_maps = fourierScaler.transform(features_maps.reshape(len(features_maps), -1))

    features_maps = features_maps.reshape(len(features_maps), -1, 40, 1)

    # Scale labels
    labels = np.array(labels)
    barrierScaler = StandardScaler()
    barrierScaler.fit(labels.reshape(-1, 1))
    labels = barrierScaler.transform(labels.reshape(-1, 1))

    # Reserve 10% as validation
    (features_maps, features_maps_test, labels, labels_test) = train_test_split(
        features_maps, labels, test_size=0.1, random_state=32)

    # Split the rest of the data
    (trainX, testX, trainY, testY) = train_test_split(
        features_maps, labels, test_size=0.2, random_state=32)

    np.save("fourier_features_train.npy", trainX)
    np.save("fourier_labels_train.npy", trainY)

    np.save("fourier_features_val.npy", testX)
    np.save("fourier_labels_val.npy", testY)

    np.save("fourier_features_test.npy", features_maps_test)
    np.save("fourier_labels_test.npy", labels_test)

    tuner = kt.Hyperband(
        get_model,
        objective='val_mean_squared_error',
        max_epochs=600,
        project_name="Hyperband_Fourier",
    )

    tuner.search(trainX, trainY,
                 validation_data=(testX, testY),
                 epochs=500,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)])

    tuner.results_summary()


if __name__ == "__main__":
    main()
