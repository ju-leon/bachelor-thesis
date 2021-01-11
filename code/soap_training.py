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


def get_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Dense(540, activation="relu",
                              kernel_regularizer=regularizers.l2(0.04))(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.BatchNormalization()(x)

    #x = tf.keras.layers.Dense(300, activation="relu", kernel_regularizer=regularizers.l2(0.02))(x)
    #x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(340, activation="relu",
                              kernel_regularizer=regularizers.l2(0.03))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(220, activation="relu",
                              kernel_regularizer=regularizers.l2(0.02))(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(100, activation="relu",
                              kernel_regularizer=regularizers.l2(0.02))(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=regularizers.l2(0.01))(x)
    #x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1, kernel_regularizer='l2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

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

    parser.add_argument('--augment_steps', default=5,
                        help='NUmber of augmentations around Z axis for every sample', type=int)

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

    labels = labels.reshape(
        number_samples, args.augment_steps, -1)

    (trainX, testX, trainY, testY) = train_test_split(
        features_soap, labels, test_size=args.test_split, random_state=32)

    trainX = trainX.reshape(-1, features_soap.shape[2])
    testX = testX.reshape(-1, features_soap.shape[2])
    trainY = trainY.flatten()
    testY = testY.flatten()

    model = get_model((trainX.shape[1],))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="mean_squared_error", optimizer=opt)

    # Train the model
    H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=800,
                  batch_size=60, verbose=2, callbacks=[tf.keras.callbacks.LearningRateScheduler(step_decay)])

    # Save loss of current model
    save_loss(H, args.out_dir + "loss__augment_steps=" + str(args.augment_steps) +
              "_l=" + str(lmax) + ",n=" + str(nmax) + ".pdf")

    # Scale back
    train_y_pred = barrierScaler.inverse_transform(model.predict(trainX))
    train_y_real = barrierScaler.inverse_transform(trainY)

    test_y_pred = barrierScaler.inverse_transform(model.predict(testX))
    test_y_real = barrierScaler.inverse_transform(testY)

    save_scatter(train_y_real, train_y_pred, test_y_real, test_y_pred, args.out_dir +
                 "scatter__" + str(args.augment_steps) +
                 "_l=" + str(lmax) + ",n=" + str(nmax) + ".pdf")

    # Save R2, MAE
    r2, mae = reg_stats(testY, model.predict(testX), barrierScaler)
    file = open(args.out_dir + "out.csv", "a")
    file.write(str(args.augment_steps))
    file.write(",")
    file.write(str(args.nmax))
    file.write(",")
    file.write(str(args.lmax))
    file.write(",")
    file.write(str(r2))
    file.write(",")
    file.write(str(mae))
    file.write("\n")
    file.close()


if __name__ == "__main__":
    main()
