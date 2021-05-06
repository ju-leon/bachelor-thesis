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


def generate_features(species, data_dir, nmax=8, lmax=4, rcut=12, augment_steps=30, interpolate=False, interpolation_steps=10):
    # Load data
    elems, labels, names = read_data(data_dir)

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

    if interpolate:
        features_interpolates, labels = get_interpolations(
            data_dir, features_soap, labels, names, interpolation_steps=interpolation_steps)
        return features_interpolates, labels
    else:
        return features_soap, labels
