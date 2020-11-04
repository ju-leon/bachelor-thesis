import os
import sys
import argparse
import numpy as np
from alignment import align_catalyst
from definitions import Point, Atom
from mendeleev import element
from create_slices import slice_catalyst
from tqdm import tqdm

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


def generate_features(atoms, layer_height, z_start, z_end):
    aligned_atoms = align_catalyst(atoms)
    return slice_catalyst(aligned_atoms, layer_height, z_start, z_end)


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
    parser.add_argument('--z_start', default=-5)
    parser.add_argument('--z_end', default=5)

    args, other_args = parser.parse_known_args()

    for f in tqdm(os.listdir(args.data_dir)):
        if f.endswith(".xyz"):
            atoms = read_from_file(args.data_dir + f)
            features = generate_features(atoms, args.layer_height,
                                         args.z_start, args.z_end)
            
            np.save(args.out_dir + f.replace(".xyz", "npy"), features)


if __name__ == "__main__":
    main()
