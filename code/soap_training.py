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

from kerastuner.tuners import Hyperband
import kerastuner as kt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

import csv

from sklearn.preprocessing import StandardScaler

from soap_generation.alignment import align_elements
from soap_generation.augment import augment_elements
from soap_generation.generate import generate_features

import os.path
from os import path
import sys

input_shape = 0


def read_data(data_dir):
    barriers = dict()

    with open(data_dir + 'vaskas_features_properties_smiles_filenames.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            barriers[row[93]] = float(row[91])

    labels = []
    elems = []
    for f in tqdm(os.listdir(data_dir + "coordinates_molSimplify/")):
        if f.endswith(".xyz"):
            elems.append(read(data_dir + "coordinates_molSimplify/" + f))
            labels.append(barriers[f[:-4]])

    labels = np.array(labels)

    return elems, labels


def save_loss(history, location):
    fig = plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(location)


def save_scatter(train_y_real, train_y_pred, val_y_real, val_y_pred, test_y_real, test_y_pred, location):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_y_real, train_y_pred,
               marker="o", c="C1", label="Training")
    ax.scatter(val_y_real, val_y_pred, marker="o", c="C3", label="Validation")
    ax.scatter(test_y_real, test_y_pred, marker="o",
               c="C2", label="Testing")
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
    global input_shape
    # input_shape = (12, 48, 1)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    x = tf.keras.layers.Flatten()(x)

    for i in range(hp.Int('hidden_layers', 1, 6, default=3)):
        size = hp.Int('hidden_size_' + str(i), 10, 700, step=40)
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


def reg_stats(y_true, y_pred, scaler):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae


def interpolate(feature1, feature2, steps=10):
    interpolations = []
    for alpha in np.linspace(0, 1, steps):
        interpolations.append(
            (feature1 * alpha) + (feature2 * (1 - alpha)))

    return interpolations


def main():
    parser = argparse.ArgumentParser(
        description='Generate rotationally invariant features from catalysts using fourier descriptors')
    parser.add_argument(
        'data_dir', help='Directory with xyz files for feature generation')

    parser.add_argument(
        'out_dir', help='Directory for storing generated features')

    parser.add_argument('--augment_steps', default=20,
                        help='Number of augmentations around Z axis for every sample', type=int)

    parser.add_argument('--nmax', default=3,
                        help='Size of test fraction from training data', type=int)

    parser.add_argument('--lmax', default=3,
                        help='Size of test fraction from training data', type=int)

    parser.add_argument('--rcut', default=12,
                        help='Cutoff radius', type=int)

    parser.add_argument('--test_split', default=0.2, type=float)

    parser.add_argument('--batch_size', default=400,
                        help='Batch size', type=int)

    parser.add_argument('--add_interpolations', default=True, type=bool)

    parser.add_argument('--interpolation_steps', default=2, type=int)

    args = parser.parse_args()

    # Check if hyperparam optimization was run for given pair
    if not path.exists("Hyperband_FINAL_SNAP_" + str(args.nmax) + ":" + str(args.lmax) + ":0.2"):
        print("Skipping " + str(args.nmax) + ":" + str(args.lmax))
        # sys.exit()

    species = ["H", "C", "N", "O", "F", "P", "S", "Cl", "As", "Br", "I", "Ir"]

    trainX, trainY, testX, testY = generate_features(species,
                                                     split=args.test_split,
                                                     data_dir=args.data_dir,
                                                     augment_steps=args.augment_steps,
                                                     interpolate=True,
                                                     nmax=args.nmax,
                                                     lmax=args.lmax,
                                                     rcut=args.rcut,
                                                     interpolation_steps=args.interpolation_steps
                                                     )

    print("Data Length: " + str(len(trainX)))
    print("Label Length: " + str(len(trainY)))

    trainY = np.array(trainY)
    testY = np.array(testY)

    (testX, valX, testY, valY) = train_test_split(
        testX, testY, test_size=0.5, random_state=32)

    nmax = args.nmax
    lmax = args.lmax
    rcut = args.rcut

    # np.save("features_train_" + str(nmax) + ":" +
    #        str(lmax) + ":" + str(args.test_split) + ".npy", trainX)
    # np.save("labels_train_" + str(nmax) + ":" + str(lmax) + ":" +
    #        str(args.test_split) + ".npy", trainY)

    np.save("features_val_" + str(nmax) + ":" +
            str(lmax) + ":" + str(args.test_split) + ".npy", valX)
    np.save("labels_val_" + str(nmax) + ":" + str(lmax) +
            ":" + str(args.test_split) + ".npy", valX)

    np.save("features_test_" + str(nmax) + ":" +
            str(lmax) + ":" + str(args.test_split) + ".npy", testX)
    np.save("labels_test_" + str(nmax) + ":" +
            str(lmax) + ":" + str(args.test_split) + ".npy", testY)

    # trainX = trainX.reshape(-1, 12, int(trainX.shape[-1] / 12), 1)
    testX = testX.reshape(-1, 12, int(testX.shape[-1] / 12), 1)
    valX = valX.reshape(-1, 12, int(valX.shape[-1] / 12), 1)
    trainY = trainY.reshape(-1, 1)
    testY = testY.flatten(-1, 1)
    valY = valY.flatten(-1, 1)

    file_identifier = "__augment_steps=" + str(args.augment_steps) + "_l=" + str(
        lmax) + "_n=" + str(nmax) + "_split=" + str(args.test_split) + "_rcut=" + str(rcut) + "_batch" + str(args.batch_size)

    global input_shape
    input_shape = trainX[0].shape

    print(input_shape)
    print(trainX[-1].shape)
    print(testX[0].shape)
    print(testX[-1].shape)

    print(len(trainX))
    print(len(trainY))

    print(len(testX))
    print(len(testY))

    tuner = kt.Hyperband(
        get_model,
        objective='val_mean_squared_error',
        max_epochs=1200,
        project_name="Hyperband_FINAL_SNAP_" +
        str(args.nmax) + ":" + str(args.lmax) + ":0.2"
    )

    best_hp = tuner.get_best_hyperparameters(3)[0]

    model = get_model(best_hp)

    opt = tf.keras.optimizers.Adam(learning_rate=tuner.get_best_hyperparameters(3)[
        0]["learning_rate"])
    model.compile(loss="mean_squared_error", optimizer=opt)

    # Train the model
    H = model.fit(
        x=[trainX],
        y=trainY,
        validation_data=([valX], valY),
        epochs=20000,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=200)]
    )

    # Save loss of current model
    save_loss(H, args.out_dir + "loss" + file_identifier + ".png")

    # Scale back
    train_y_pred = barrierScaler.inverse_transform(
        model.predict(trainX))
    train_y_real = barrierScaler.inverse_transform(trainY)

    val_y_pred = barrierScaler.inverse_transform(model.predict(valX))
    val_y_real = barrierScaler.inverse_transform(valY)

    test_y_pred = barrierScaler.inverse_transform(model.predict(testX))
    test_y_real = barrierScaler.inverse_transform(testY)

    save_scatter(train_y_real, train_y_pred, val_y_real, val_y_pred,
                 test_y_real, test_y_pred, args.out_dir + "scatter" + file_identifier + ".png")

    # Save R2, MAE
    r2, mae = reg_stats(testY, model.predict(testX), barrierScaler)

    file = open(args.out_dir + "out.csv", "a")
    file.write(str(args.augment_steps))
    file.write(",")
    file.write(str(args.batch_size))
    file.write(",")
    file.write(str(args.test_split))
    file.write(",")
    file.write(str(args.nmax))
    file.write(",")
    file.write(str(args.lmax))
    file.write(",")
    file.write(str(rcut))
    file.write(",")
    file.write(str(r2))
    file.write(",")
    file.write(str(mae))
    file.write("\n")
    file.close()

    model.save(args.out_dir + "model" + file_identifier + ".h5")


if __name__ == "__main__":
    main()
