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

import matplotlib.pyplot as plt

radii = dict()
layers = 1

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

def reg_stats(y_true, y_pred, scaler):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    return r2, mae

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
    fourier = fourier.reshape(
        (fourier.shape[0], fourier.shape[2], fourier.shape[3]))

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
    input_shape = (layers, 40, 1)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    x = tf.keras.layers.Flatten()(x)

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

    parser.add_argument(
        'out_dir', help='Directory for output')

    args = parser.parse_args()

    barriers = dict()
    auto = dict()
    with open(args.data_dir + 'vaskas_features_properties_smiles_filenames.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            barriers[row[93]] = float(row[91])
            auto[row[93]] = row[0:30]



    layer_heights = [0.5]
    for layer_height in layer_heights:
        features_maps = []
        labels = []
        autocor = []
        for f in tqdm(os.listdir(args.data_dir + "/coordinates_molSimplify/")):
            if f.endswith(".xyz"):
                atoms = read_from_file(
                    args.data_dir + "/coordinates_molSimplify/" + f)

                slices = generate_slices(atoms, layer_height,
                                         -10, 5, 0.1, ["X"], False)

                feature_map = generate_fourier_descriptions(slices, 10)
                features_maps.append(feature_map)

                labels.append(barriers[f[:-4]])
                autocor.append(auto[f[:-4]])

        labels = np.array(labels).reshape(-1)

        features_maps = np.array(features_maps)

        autocor = np.array(autocor).astype(np.float)

        print(autocor)

        np.save("fourier_features_input.npy", features_maps)
        np.save("fourier_features_autocor.npy", autocor)
        np.save("fourier_features_labels.npy", labels)

        exit()

        # Scale coefficents
        fourierScaler = StandardScaler()
        fourierScaler.fit(features_maps.reshape(len(features_maps), -1))
        features_maps = fourierScaler.transform(
            features_maps.reshape(len(features_maps), -1))

        features_maps = features_maps.reshape(len(features_maps), -1, 40, 1)

        global layers
        layers = features_maps.shape[1]

        # Scale labels
        labels = np.array(labels)
        barrierScaler = StandardScaler()
        barrierScaler.fit(labels.reshape(-1, 1))
        labels = barrierScaler.transform(labels.reshape(-1, 1))

        # Reserve 10% as validation
        (features_maps, testX, labels, testY) = train_test_split(
            features_maps, labels, test_size=0.1, random_state=32)

        tuner = kt.Hyperband(
            get_model,
            objective='val_mean_squared_error',
            max_epochs=600,
            project_name="Hyperband_Fourier_" + str(layer_height),
        )

        for split in [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            # Split the rest of the data
            (trainX, valX, trainY, valY) = train_test_split(
                features_maps, labels, test_size=split, random_state=32)

            np.save("fourier_features_train_" + str(layer_height) + ".npy", trainX)
            np.save("fourier_labels_train_" + str(layer_height) + ".npy", trainY)

            np.save("fourier_features_val_" + str(layer_height) + ".npy", valX)
            np.save("fourier_labels_val_" + str(layer_height) + ".npy", valX)

            np.save("fourier_features_test_" +
                    str(layer_height) + ".npy", testX)
            np.save("fourier_labels_test_" +
                    str(layer_height) + ".npy", testY)



            best_hp = tuner.get_best_hyperparameters(3)[0]

            model = get_model(best_hp)

            opt = tf.keras.optimizers.Adam(learning_rate=tuner.get_best_hyperparameters(3)[
                                        0]["learning_rate"])
            model.compile(loss="mean_squared_error", optimizer=opt)

            # Train the model
            H = model.fit(
                x=trainX,
                y=trainY,
                validation_data=(valX, valY),
                epochs=1000,
                batch_size=args.batch_size,
                verbose=2,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=200)]
            )


            # Scale back
            train_y_pred = barrierScaler.inverse_transform(
                model.predict(trainX))
            train_y_real = barrierScaler.inverse_transform(trainY)

            val_y_pred = barrierScaler.inverse_transform(model.predict(valX))
            val_y_real = barrierScaler.inverse_transform(valY)

            test_y_pred = barrierScaler.inverse_transform(model.predict(testX))
            test_y_real = barrierScaler.inverse_transform(testY)

            save_scatter(train_y_real, train_y_pred, val_y_real, val_y_pred,
                        test_y_real, test_y_pred, args.out_dir + "scatter_fourier" + str(split) + ".png")

            # Save R2, MAE
            r2, mae = reg_stats(testY, model.predict(testX), barrierScaler)

            file = open(args.out_dir + "out_fourier.csv", "a")
            file.write(str(split))
            file.write(",")
            file.write(str(r2))
            file.write(",")
            file.write(str(mae))
            file.write("\n")
            file.close()




        


if __name__ == "__main__":
    main()
