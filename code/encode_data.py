import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from mendeleev import element

from feature_generation.definitions import Point, Atom
from feature_generation.create_slices import slice_catalyst
from feature_generation.contour_descriptor import fourier_descriptor
from feature_generation.alignment import align_catalyst


def read_from_file(file):
    atoms = []
    for lineidx, line in enumerate(open(file, "r")):
        if lineidx >= 2:
            elem = line.split()[0].capitalize()
            location = np.array([float(line.split()[1]), float(
                line.split()[2]), float(line.split()[3])])

            atoms.append(Atom(elem, location, element(
                elem).atomic_radius_rahm / 100))
    return atoms


def generate_slices(atoms, layer_height, z_start, z_end, resolution):
    aligned_atoms = align_catalyst(atoms)
    slices = slice_catalyst(aligned_atoms, layer_height, z_start, z_end, resolution)
    return slices


def generate_fourier_descriptions(slices, order):
    fourier = []
    for slice in slices:
        fourier.append(fourier_descriptor(slice, order))

    return np.array(fourier)


def main():
    """
    Extracts all features from files in given directory. Saves extracted features as numpy array in out location.
    """
    parser = argparse.ArgumentParser(
        description='Generate rotationally invariant features from catalysts using fourier descriptors')
    parser.add_argument(
        'data_dir', help='Directory with xyz files for feature generation')
    parser.add_argument(
        'out_dir', help='Directory for storing generated features')

    parser.add_argument('--layer_height', default=0.5,
                        help='Height of each slice through the atom in Angstrom', type=float)
    parser.add_argument('--z_start', default=-5,
                        help='Start of the slices relative to metal center in Angstrom')
    parser.add_argument(
        '--z_end', default=5, help='End of the slices relative to metal center in Angstrom')
    parser.add_argument('--order', default=10,
                        help='Order of the fourier descriptor', type=int)
    parser.add_argument('--contour_res', default=0.01,
                        help='Resolution of the contour. Smaller numer is higher resolution', type=float)

    args, other_args = parser.parse_known_args()

    for f in tqdm(os.listdir(args.data_dir)):
        if f.endswith(".xyz"):
            atoms = read_from_file(args.data_dir + f)
            slices = generate_slices(atoms, args.layer_height,
                                     args.z_start, args.z_end, args.contour_res)

            fourier = generate_fourier_descriptions(slices, args.order)
            np.save(args.out_dir + f.replace(".xyz", ".npy"), fourier)


if __name__ == "__main__":
    main()
