import dscribe
import numpy as np
import os
from tqdm import tqdm
from dscribe.descriptors import SOAP

from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.visualize import view
import math
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

from kerastuner.tuners import Hyperband
import kerastuner as kt

import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

import csv

from sklearn.preprocessing import StandardScaler

from soap_generation.alignment import align_elements
from soap_generation.augment import augment_elements


def read_data(data_dir):
    barriers = dict()

    with open(data_dir + 'vaskas_features_properties_smiles_filenames.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            barriers[row[93]] = float(row[91])

    labels = []
    elems = []
    for f in tqdm(os.listdir(data_dir + "coordinates_TS/")):
        if f.endswith(".xyz"):
            elems.append(read(data_dir + "coordinates_TS/" + f))
            labels.append(barriers[f[:-7]])

    labels = np.array(labels)

    return elems, labels


def save_loss(history, location):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(location)


def save_scatter(train_y_real, train_y_pred, test_y_real, test_y_pred, location):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_y_real, train_y_pred,
               marker="o", c="C1", label="Training")
    ax.scatter(test_y_real, test_y_pred, marker="o",
               c="C3", label="Validation")
    ax.set_aspect('equal')
    ax.set_xlabel("Calculated barrier [kcal/mol]")
    ax.set_ylabel("Predicted barrier [kcal/mol]")
    ax.legend(loc="upper left")
    plt.savefig(location)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.6
    epochs_drop = 80.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1+epoch)/epochs_drop))
    return lrate


def get_model(hp):
    input_shape = (12, 48, 1)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    filter_config = hp.Choice(
        'filter_config',
        values=[0, 1],
        default=0,
    )

    if filter_config == 0:
        filter_size_limit = [4, 6, 4]
    else:
        filter_size_limit = [4, 8, 2]

    for i in range(hp.Int('conv_blocks', 1, 3, default=1)):
        filters = hp.Int('filters_' + str(i), 1, 8, step=1)

        filter_size = hp.Int('filter_size_' + str(i), 1,
                             filter_size_limit[i], step=1)


        filter_quadratic = hp.Choice(
            'quadratic',
            values=[True, False],
            default=False,
        )

        if filter_quadratic:
            x = tf.keras.layers.Conv2D(filters, [filter_size, filter_size])(x)
        else:
            x = tf.keras.layers.Conv2D(filters, [filter_size, 1])(x)

        dropout = hp.Float('conv_dropout_' + str(i), 0,
                           0.6, step=0.1, default=0.2)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    for i in range(hp.Int('hidden_layers', 1, 6, default=3)):
        size = hp.Int('hidden_size_' + str(i), 10, 700, step=40)
        reg = hp.Float('hidden_reg_' + str(i), 0, 0.1, step=0.01, default=0.02)
        dropout = hp.Float('hidden_dropout_' + str(i),
                           0, 0.5, step=0.1, default=0.2)
        x = tf.keras.layers.Dense(size, activation="relu",
                                  kernel_regularizer=regularizers.l2(reg))(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1, kernel_regularizer='l2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def reg_stats(y_true, y_pred, scaler):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae


def main():
    parser = argparse.ArgumentParser(
        description='Generate rotationally invariant features from catalysts using fourier descriptors')
    parser.add_argument(
        'data_dir', help='Directory with xyz files for feature generation')

    parser.add_argument(
        'out_dir', help='Directory for storing generated features')

    parser.add_argument('--test_split', default=0.2,
                        help='Size of test fraction from training data', type=float)

    parser.add_argument('--augment_steps', default=20,
                        help='Number of augmentations around Z axis for every sample', type=int)

    parser.add_argument('--nmax', default=3,
                        help='Size of test fraction from training data', type=int)

    parser.add_argument('--lmax', default=3,
                        help='Size of test fraction from training data', type=int)

    parser.add_argument('--rcut', default=6.0,
                        help='Size of test fraction from training data', type=float)

    args = parser.parse_args()

    elems, labels = read_data(args.data_dir)

    number_samples = len(elems)

    elems = align_elements(elems)

    elems, labels = augment_elements(elems, labels, args.augment_steps)

    species = ["H", "C", "N", "O", "F", "P", "S", "Cl", "As", "Br", "I", "Ir"]
    rcut = args.rcut
    nmax = args.nmax
    lmax = args.lmax

    soap = dscribe.descriptors.SOAP(
        species=species,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        rbf="gto"
    )

    # Create soap coefficients
    atom_index = [[0]] * len(elems)
    features_soap = soap.create_coeffs(elems, positions=atom_index)

    # Scale coefficents
    soapScaler = StandardScaler()
    soapScaler.fit(features_soap)
    features_soap = soapScaler.transform(features_soap)

    # Scale labels
    labels = np.array(labels)
    barrierScaler = StandardScaler()
    barrierScaler.fit(labels.reshape(-1, 1))
    labels = barrierScaler.transform(labels.reshape(-1, 1))

    # Reshape so all auguemented data for every sample are either in training or in test data
    features_soap = features_soap.reshape(
        number_samples, args.augment_steps, -1)

    print(features_soap.shape)

    labels = labels.reshape(
        number_samples, args.augment_steps, -1)

    (trainX, testX, trainY, testY) = train_test_split(
        features_soap, labels, test_size=args.test_split, random_state=32)

    trainX = trainX.reshape(-1, 12, int(features_soap.shape[2] / 12), 1)
    testX = testX.reshape(-1, 12, int(features_soap.shape[2] / 12), 1)
    trainY = trainY.flatten()
    testY = testY.flatten()

    tuner = kt.Hyperband(
        get_model,
        objective='val_mean_squared_error',
        max_epochs=1000,
        hyperband_iterations=2)

    tuner.search(trainX, trainY,
                 validation_data=(testX, testY),
                 epochs=1000,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=20)])

    models = tuner.get_best_models(num_models=2)
    tuner.results_summary()

    for model, index in zip(models, len(models)):
        model.save("hypermodel_" + str(index) + ".h5")


if __name__ == "__main__":
    main()
