from soap_generation.neighbor import get_interpolations
import csv
import numpy as np
from tqdm import tqdm

import dscribe
from dscribe.descriptors import SOAP

import os

from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.visualize import view

from soap_generation.alignment import align_elements
from soap_generation.augment import augment_elements

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pickle import dump


def read_data(data_dir):
    barriers = dict()

    with open(data_dir + 'vaskas_features_properties_smiles_filenames.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            barriers[row[93]] = float(row[91])

    labels = []
    elems = []
    names = []
    for f in tqdm(os.listdir(data_dir + "coordinates_molSimplify/")):
        if f.endswith(".xyz"):
            elems.append(read(data_dir + "coordinates_molSimplify/" + f))
            labels.append(barriers[f[:-4]])
            names.append(f)

    labels = np.array(labels)

    return elems, labels, names


def generate_features(species, data_dir, split=0.2, nmax=8, lmax=4, rcut=12, augment_steps=30, interpolate=False, interpolation_steps=10, sidegroup_validation="none", file_identifier="save_", out_dir=""):
    # Load data
    elems, labels, names = read_data(data_dir)
    number_samples = len(elems)

    # Align elements
    elems = align_elements(elems)

    elems, labels, names = augment_elements(
        elems, labels, augment_steps, names=names)

    # Setting up the SOAP descriptor
    soap = dscribe.descriptors.SOAP(
        species=species,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        rbf="gto"
    )

    atom_index = [[0]] * len(elems)
    features_soap = soap.create_coeffs(elems, positions=atom_index)

    soapScaler = StandardScaler()
    soapScaler.fit(features_soap)
    features_soap = soapScaler.transform(features_soap)

    dump(soapScaler, open(out_dir + 'featureScaler.pkl', 'wb'))

    labels = np.array(labels)
    barrierScaler = StandardScaler()
    barrierScaler.fit(labels.reshape(-1, 1))
    labels = barrierScaler.transform(labels.reshape(-1, 1))

    dump(barrierScaler, open(out_dir + 'barrierScaler.pkl', 'wb'))

    features_soap = features_soap.reshape(
        number_samples, augment_steps, -1)

    labels = labels.reshape(
        number_samples, augment_steps, -1)

    names = np.array(names)
    names = names.reshape(
        number_samples, augment_steps, -1)

    if sidegroup_validation != 'none':
        print("Using ligand as validation: " + sidegroup_validation)
        trainX = []
        trainY = []
        trainNames = []
        testX = []
        testY = []
        testNames = []

        features_soap = features_soap.reshape(
            number_samples * augment_steps, -1)

        labels = labels.reshape(
            number_samples * augment_steps, -1)

        names = names.reshape(
            number_samples * augment_steps, -1)

        for feature, label, name in zip(features_soap, labels, names):
            if sidegroup_validation in name[0]:
                testX += [feature]
                testY += [label]
                testNames += [name]
            else:
                trainX += [feature]
                trainY += [label]
                trainNames += [name]

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
    else:
        (trainX, testX, trainY, testY) = train_test_split(
            features_soap, list(zip(labels, names)), test_size=0.2, random_state=32)
        trainY, trainNames = list(zip(*trainY))
        testY, testNames = list(zip(*testY))

    trainX = np.array(trainX).reshape(-1, trainX.shape[-1])
    trainY = np.array(trainY).flatten()
    trainNames = np.array(trainNames).flatten()
    testNames = np.array(testNames).flatten()

    if interpolate:
        features_interpolates, labels = get_interpolations(
            data_dir, trainX, trainY, trainNames, interpolation_steps=interpolation_steps)
        return features_interpolates, labels, trainNames, testX, testY, testNames
    else:
        return features_soap, labels, trainNames, testX, testY, testNames
