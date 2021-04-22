from kerastuner.tuners import Hyperband
import kerastuner as kt

import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import imshow
import math
import sklearn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from collections import namedtuple
from ase.io import read
from ase.build import molecule
from ase import Atoms, Atom
from ase.visualize import view
from ase.geometry.analysis import Analysis
import plotly.graph_objects as go
from ase.data import vdw_radii
from ase.data.colors import cpk_colors, jmol_colors

from voxel.generator import VoxelGenerator
import csv

from kerastuner.tuners import Hyperband
import kerastuner as kt

from soap_generation.alignment import align_elements
from soap_generation.augment import augment_elements

from voxel.generator import VoxelGenerator

def get_model(hp):
    input_shape = trainX[0].shape
    inputs = tf.keras.Input(shape=input_shape)
    
    x = inputs
    
    for i in range(hp.Int('conv_layer', 1, 4, default=3)):
        kernel = hp.Int('kernel_size_' + str(i), 3, 50)
        filters = hp.Int('num_filter_' + str(i), 1, 32)
        
        x = tf.keras.layers.Conv3D(filters, kernel, activation='relu', padding='same', kernel_initializer='he_uniform')(x)
    
        pooling = hp.Choice('pooling_' + str(i), values=[True, False])
        
        if pooling:
            pool = hp.Int('pooling_size_' + str(i), 2, 3)
            x = tf.keras.layers.MaxPooling3D(pool_size=(pool, pool, pool))(x)
    
    
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




'''
LOAD THE DATA
'''
barriers = dict()
with open('../data/vaskas_features_properties_smiles_filenames.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        #images.append(row[0])
        #elos.append(row[1])
        barriers[row[93]] = float(row[91])


labels = []
elems = []
for f in tqdm(os.listdir("../data/coordinates_molSimplify/")):
    if f.endswith(".xyz"):
        elems.append(read("../data/coordinates_molSimplify/" + f))
        labels.append(barriers[f[:-4]])

labels = np.array(labels)
number_samples = len(labels)



'''
ALIGN AND AUGMENT
'''
elems = align_elements(elems)
elems, labels = augment_elements(elems, labels, 30)


'''
VOXELISE
'''
species = ["H","C","N","O","F","P","S","Cl","As","Br","I","Ir"]
voxel_gen = VoxelGenerator(species, scale=8, resolution=20)
elems_voxel = voxel_gen.generate_voxel(elems)


'''
TRAIN, TEST, VAL SPLIT
'''
labels = np.array(labels).reshape(-1,1)

(trainX, testX, trainY, testY) = train_test_split(
        elems_voxel, labels, test_size=0.2, random_state=32)

(testX, valX, testY, valY) = train_test_split(
        testX, testY, test_size=0.5, random_state=32)



tuner = kt.Hyperband(
    get_model,
    objective='val_mean_squared_error',
    max_epochs=1200,
    project_name="Hyperband_VOXEL_100",
    directory="/pfs/work7/workspace/scratch/utpqw-data-0/hyperband_voxel/"
)

trainX = np.array(trainX)
trainY = np.array(trainY)
valX = np.array(valX)
valY = np.array(valY)
testX = np.array(testX)
testY = np.array(testY)

np.save("/pfs/work7/workspace/scratch/utpqw-data-0/dump/trainX_voxel.npy", trainX)
np.save("/pfs/work7/workspace/scratch/utpqw-data-0/dump/trainY_voxel.npy", trainY)

np.save("/pfs/work7/workspace/scratch/utpqw-data-0/dump/testX_voxel.npy", testX)
np.save("/pfs/work7/workspace/scratch/utpqw-data-0/dump/testY_voxel.npy", testY)

np.save("/pfs/work7/workspace/scratch/utpqw-data-0/dump/trainX_voxel.npy", trainX)
np.save("/pfs/work7/workspace/scratch/utpqw-data-0/dump/trainY_voxel.npy", trainY)


tuner.search(trainX, trainY,
             validation_data=(valX, valY),
             epochs=1500)

tuner.results_summary()