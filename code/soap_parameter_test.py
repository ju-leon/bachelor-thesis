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


def reg_stats(y_true, y_pred, scaler):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae


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
    ax.scatter(train_y_real, train_y_pred, marker="o", c="C1", label="Training")
    ax.scatter(test_y_real, test_y_pred, marker="o", c="C3", label="Validation")
    ax.set_aspect('equal')
    ax.set_xlabel("Calculated barrier [kcal/mol]")
    ax.set_ylabel("Predicted barrier [kcal/mol]")
    ax.legend(loc="upper left")
    plt.savefig(location)

def get_model():
    model_full = Sequential()
    model_full.add(keras.layers.Flatten())

    model_full.add(Dense(256, activation="relu", kernel_regularizer='l2'))
    model_full.add(keras.layers.Dropout(0.5))

    model_full.add(Dense(128, activation="relu", kernel_regularizer='l2'))
    model_full.add(keras.layers.Dropout(0.3))

    model_full.add(Dense(64, activation="relu", kernel_regularizer='l2'))
    model_full.add(keras.layers.BatchNormalization())

    model_full.add(Dense(1, kernel_regularizer='l2'))

    return model_full


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.6
    epochs_drop = 80.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1+epoch)/epochs_drop))
    return lrate


def main():
    parser = argparse.ArgumentParser(
        description='Generate rotationally invariant features from catalysts using fourier descriptors')
    parser.add_argument(
        'data_dir', help='Directory with xyz files for feature generation')

    parser.add_argument(
        'out_dir', help='Directory for storing generated features')

    parser.add_argument('--test_split', default=0.2,
                        help='Size of test fraction from training data', type=float)


    args = parser.parse_args()

    names = []
    elems = []
    for f in tqdm(os.listdir(args.data_dir + "coordinates_TS/")):
        if f.endswith(".xyz"):
            elems.append(read(args.data_dir + "coordinates_TS/" + f))
            names.append(f)

    species = ["H", "C", "O", "N", "Ir", "As", "S", "P", "Br", "Cl", "F", "I"]
    rcut = 8.0
    nmax = 3
    lmax = 1

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
    )

    atom_index = [[0]] * len(elems)
    features_cnn = soap.create(elems, positions=atom_index)

    # Read barriers from CSV
    barriers = dict()
    with open(args.data_dir + 'vaskas_features_properties_smiles_filenames.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            # images.append(row[0])
            # elos.append(row[1])
            barriers[row[93]] = float(row[91])

    # Get labels for each molecule
    labels = []
    for x in range(len(features_cnn)):
        labels.append(barriers[names[x][:-7]])

    labels = np.array(labels)

    print(features_cnn.shape)
    print(labels.shape)

    # Scale Input
    inputScaler = StandardScaler()
    inputScaler.fit(features_cnn)
    features_cnn = inputScaler.transform(features_cnn)

    # Scale Output
    barrierScaler = StandardScaler()
    barrierScaler.fit(labels.reshape(-1, 1))
    labels = barrierScaler.transform(labels.reshape(-1, 1))

    # Train Test split
    (trainX_cnn, testX_cnn, trainY, testY) = train_test_split(
        features_cnn, labels, test_size=args.test_split, random_state=32)

    # Train model
    model = get_model()
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="mean_squared_error", optimizer=opt)

    H = model.fit(x=trainX_cnn, y=trainY, validation_data=(testX_cnn, testY),
                       epochs=1000, batch_size=64, callbacks=[tf.keras.callbacks.LearningRateScheduler(step_decay)])


    save_loss(H, args.out_dir + "loss_" + str(args.test_split) + ".pdf")

    # Save R2, MAE
    r2, mae = reg_stats(trainY, model.predict(trainX_cnn), barrierScaler)
    file = open(args.out_dir  + "out.csv","a")
    file.write(str(args.test_split))
    file.write(",")
    file.write(str(r2))
    file.write(",")
    file.write(str(mae))
    file.write("\n")
    file.close()
    
    # Scale back
    train_y_pred = barrierScaler.inverse_transform(model.predict(trainX_cnn))
    train_y_real = barrierScaler.inverse_transform(trainY)

    test_y_pred = barrierScaler.inverse_transform(model.predict(testX_cnn))
    test_y_real = barrierScaler.inverse_transform(testY)

    save_scatter(train_y_real, train_y_pred, test_y_real, test_y_pred, args.out_dir + "scatter_" + str(args.test_split) + ".pdf")


if __name__ == "__main__":
    main()
