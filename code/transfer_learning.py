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

from kerastuner.tuners import Hyperband
import kerastuner as kt

import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import csv

from sklearn.preprocessing import StandardScaler

from soap_generation.alignment import align_elements
from soap_generation.augment import augment_elements

import pickle


def read_data(data_dir, dataset):
    properties = dict()

    with open(data_dir + dataset + '.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            properties[row[0]] = row  # float(row[22])

    labels = []
    elems = []
    for f in tqdm(os.listdir(data_dir + "combined/")):
        if f.endswith(".xyz"):
            if f in properties.keys():
                elems.append(read(data_dir + "combined/" + f))
                labels.append(properties[f])

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


def save_scatter(train_y_real, train_y_pred, val_y_real, val_y_pred,
                 test_y_real, test_y_pred, location):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_y_real,
               train_y_pred,
               marker="o",
               c="C1",
               label="Training")
    ax.scatter(val_y_real, val_y_pred, marker="o", c="C3", label="Validation")
    ax.scatter(test_y_real, test_y_pred, marker="o", c="C2", label="Test")

    ax.set_aspect('equal')
    ax.set_xlabel("Calculated transfer property")
    ax.set_ylabel("Predicted transfer property")
    ax.legend(loc="upper left")
    plt.savefig(location)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.6
    epochs_drop = 80.0
    lrate = initial_lrate * math.pow(drop, math.floor(
        (1 + epoch) / epochs_drop))
    return lrate


def get_model(hp):
    global input_shape

    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    x = tf.keras.layers.Flatten()(x)

    for i in range(hp.Int('hidden_layers', 1, 6, default=3)):
        size = hp.Int('hidden_size_' + str(i), 10, 700, step=40)
        reg = hp.Float('hidden_reg_' + str(i),
                       0,
                       0.06,
                       step=0.01,
                       default=0.02)
        dropout = hp.Float('hidden_dropout_' + str(i),
                           0,
                           0.5,
                           step=0.1,
                           default=0.2)

        x = tf.keras.layers.Dense(
            size,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(reg))(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        norm = hp.Choice('hidden_batch_norm_' + str(i), values=[True, False])

        if norm:
            x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(1, kernel_regularizer='l2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(
        hp.Float('learning_rate', 1e-6, 1e-4, sampling='log')),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def replace_last_layer(model, size):
    output = model.layers[-2].output
    output = tf.keras.layers.Dense(size)(output)

    model2 = tf.keras.Model(inputs=model.input, outputs=output)

    return model2


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
        description=
        'Generate rotationally invariant features from catalysts using fourier descriptors'
    )
    parser.add_argument('data_dir',
                        help='Directory with xyz files for feature generation')

    parser.add_argument('out_dir',
                        help='Directory for storing generated features')

    parser.add_argument('--test_split',
                        default=0.2,
                        help='Size of test fraction from training data',
                        type=float)

    parser.add_argument(
        '--augment_steps',
        default=30,
        help='Number of augmentations around Z axis for every sample',
        type=int)

    parser.add_argument('--nmax',
                        default=3,
                        help='Size of test fraction from training data',
                        type=int)

    parser.add_argument('--lmax',
                        default=3,
                        help='Size of test fraction from training data',
                        type=int)

    parser.add_argument('--rcut',
                        default=12.0,
                        help='Size of test fraction from training data',
                        type=float)

    parser.add_argument('--dataset',
                        default='combined',
                        help='Which dataset to train on',
                        type=str)

    parser.add_argument('--ligand_test', default='none', type=str)

    args = parser.parse_args()

    print(args.ligand_test)

    elems, labels = read_data(args.data_dir, args.dataset)

    elems = align_elements(elems)

    number_samples = len(elems)
    elems, labels = augment_elements(elems, labels, args.augment_steps)

    species = ["H", "C", "N", "O", "F", "P", "S", "Cl", "As", "Br", "I", "Ir"]
    rcut = args.rcut
    nmax = args.nmax
    lmax = args.lmax

    soap = dscribe.descriptors.SOAP(species=species,
                                    periodic=False,
                                    rcut=rcut,
                                    nmax=nmax,
                                    lmax=lmax,
                                    rbf="gto")

    # Create soap coefficients
    atom_index = [[0]] * len(elems)
    features_soap = soap.create_coeffs(elems, positions=atom_index)

    # Scale labels
    labels = np.array(labels)

    # Reshape so all augumented data for every sample are either in training or in test data
    features_soap = features_soap.reshape(number_samples, args.augment_steps,
                                          -1)

    labels = labels.reshape(number_samples, args.augment_steps, -1)

    if args.ligand_test != 'none':
        testX = []
        trainX = []
        testY = []
        trainY = []

        for x in range(len(labels)):
            if args.ligand_test in labels[x, 0, 2:6]:
                testX.append(features_soap[x])
                testY.append(labels[x])
            else:
                trainX.append(features_soap[x])
                trainY.append(labels[x])

        testX = np.array(testX)
        testY = np.array(testY)
        trainX = np.array(trainX)
        trainY = np.array(trainY)
    else:
        (trainX, testX, trainY,
         testY) = train_test_split(features_soap,
                                   labels,
                                   test_size=args.test_split,
                                   random_state=32)

    print("Len test: ", len(testX))

    (testX, valX, testY, valY) = train_test_split(testX,
                                                  testY,
                                                  test_size=0.5,
                                                  random_state=32)

    trainX = trainX.reshape(-1, 12, int(features_soap.shape[2] / 12), 1)
    valX = valX.reshape(-1, 12, int(features_soap.shape[2] / 12), 1)
    testX = testX.reshape(-1, 12, int(features_soap.shape[2] / 12), 1)

    trainY = trainY.reshape(-1, labels.shape[-1])
    valY = valY.reshape(-1, labels.shape[-1])
    testY = testY.reshape(-1, labels.shape[-1])

    directory = os.path.dirname(args.out_dir)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    """
    np.save(
        args.out_dir + "features_pre_train_" + str(nmax) + ":" + str(lmax) +
        ":" + str(args.test_split) + ".npy", trainX)
    np.save(
        args.out_dir + "labels_pre_train_" + str(nmax) + ":" + str(lmax) +
        ":" + str(args.test_split) + ".npy", trainY)

    np.save(
        args.out_dir + "features_pre_val_" + str(nmax) + ":" + str(lmax) +
        ":" + str(args.test_split) + ".npy", valX)
    np.save(
        args.out_dir + "labels_pre_val_" + str(nmax) + ":" + str(lmax) + ":" +
        str(args.test_split) + ".npy", valX)

    np.save(
        args.out_dir + "features_pre_test_" + str(nmax) + ":" + str(lmax) +
        ":" + str(args.test_split) + ".npy", testX)
    np.save(
        args.out_dir + "labels_pre_test_" + str(nmax) + ":" + str(lmax) + ":" +
        str(args.test_split) + ".npy", testY)
    """

    # Select multiple properties to train on
    trained_rows = list(range(6, 21))
    trainY = trainY[:, trained_rows].astype(float)
    valY = valY[:, trained_rows].astype(float)
    testY = testY[:, trained_rows].astype(float)

    barrierScaler = StandardScaler()
    barrierScaler.fit(trainY)
    trainY = barrierScaler.transform(trainY)
    valY = barrierScaler.transform(valY)
    testY = barrierScaler.transform(testY)

    global input_shape
    input_shape = trainX[0].shape

    tuner = kt.Hyperband(get_model,
                         objective='val_mean_squared_error',
                         max_epochs=1200,
                         project_name="Hyperband_FINAL_SNAP_" + str(nmax) +
                         ":" + str(lmax) + ":" + str(args.test_split))

    best_hp = tuner.get_best_hyperparameters(3)[0]

    model = get_model(best_hp)

    # Save the untrained model to compare later
    model.save(args.out_dir + "model_pretrain_no_transfer.h5")

    # Remove the last layer to allow to lear on transfer learning properties
    model = replace_last_layer(model, len(trained_rows))

    opt = tf.keras.optimizers.Adam(
        learning_rate=tuner.get_best_hyperparameters(3)[0]["learning_rate"])
    model.compile(loss="mean_squared_error", optimizer=opt)

    # Train the model
    H = model.fit(x=trainX,
                  y=trainY,
                  validation_data=(valX, valY),
                  epochs=300,
                  batch_size=256,
                  verbose=2,
                  callbacks=[
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=200)
                  ])

    # Save loss of current model
    save_loss(H, args.out_dir + "loss_pretrain_" + args.dataset + ".png")

    model.save(args.out_dir + "model_pretrain_" + args.dataset + ".h5")

    # These metrics are irrelevant for the transfer model
    """
    # Save R2, MAE
    r2, mae = reg_stats(testY, model.predict(testX), barrierScaler)
    file = open(args.out_dir + "out_pretrain.csv", "a")
    file.write(str(args.test_split))
    file.write(",")
    file.write(str(args.nmax))
    file.write(",")
    file.write(str(args.lmax))
    file.write(",")
    file.write(str(args.rcut))
    file.write(",")
    file.write(str(r2))
    file.write(",")
    file.write(str(mae))
    file.write("\n")
    file.close()

    # Scale back
    train_y_pred = barrierScaler.inverse_transform(model.predict(trainX))
    train_y_real = barrierScaler.inverse_transform(trainY)

    val_y_pred = barrierScaler.inverse_transform(model.predict(valX))
    val_y_real = barrierScaler.inverse_transform(valY)

    test_y_pred = barrierScaler.inverse_transform(model.predict(testX))
    test_y_real = barrierScaler.inverse_transform(testY)

    save_scatter(
        train_y_real, train_y_pred, val_y_real, val_y_pred, test_y_real,
        test_y_pred, args.out_dir + "scatter_pretrain_l=" + str(lmax) + ",n=" +
        str(nmax) + ".pdf")
    """


if __name__ == "__main__":
    main()
