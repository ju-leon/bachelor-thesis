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

import time


radii = dict()


def get_radius(atom):
    """
    Getting radii from mendeleev seems to be reallly slow. Buffer them to speed up exectuion
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


def generate_slices(atoms, layer_height, z_start, z_end, resolution, channels):
    aligned_atoms = align_catalyst(atoms)
    slices = slice_catalyst(aligned_atoms, layer_height,
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

    return np.array(fourier)


def num_element(atoms, element):
    x = 0
    for atom in atoms:
        if atom.element == element:
            x += 1

    return x

def generate_feature_vector(atoms):
    """
    Generates feature vector that holds aditional inforamtion about the molecule.
    This feature vector does not contain information abpout the shape of the molecule but rather features independent of the molecules shape.
    """
    features = []
    
    # add metal atom number
    features.append(element(atoms[0].element).atomic_number)
    features.append(len(atoms))
    features.append(num_element(atoms, "H"))
    features.append(num_element(atoms, "C"))
    features.append(num_element(atoms, "O"))
    features.append(num_element(atoms, "As"))
    features.append(num_element(atoms, "N"))
    features.append(num_element(atoms, "F"))
    features.append(num_element(atoms, "S"))
    features.append(num_element(atoms, "Cl"))
    features.append(num_element(atoms, "Br"))

    return np.array(features)


def roling_average_time(prefix=''):
    e_time = time.time()
    if not hasattr(roling_average_time, 's_time'):
        roling_average_time.s_time = e_time
        roling_average_time.average = 0
        roling_average_time.iterations = 0
    else:
        roling_average_time.average += e_time - roling_average_time.s_time
        roling_average_time.iterations += 1


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

    parser.add_argument('--layer_height', default=1,
                        help='Height of each slice through the atom in Angstrom', type=float)
    parser.add_argument('--z_start', default=-10,
                        help='Start of the slices relative to metal center in Angstrom')
    parser.add_argument(
        '--z_end', default=5, help='End of the slices relative to metal center in Angstrom')
    parser.add_argument('--order', default=10,
                        help='Order of the fourier descriptor', type=int)
    parser.add_argument('--contour_res', default=0.1,
                        help='Resolution of the contour. Smaller number is higher resolution', type=float)

    parser.add_argument('--channels', default="X",
                        help='Channels of the feature vector. X=All Atoms, Atom Letter for specific atom. As String, e.g. XHC')

    parser.add_argument('--combine_files', default=False,
                        help='Combine all feature vectors into a single file. This improves reading speed on some systems.', action='store_true')

    parser.add_argument('--track_time', default=False,
                        help='Prints the average time to encode each molecule after process is finished.', action='store_true')

    args, other_args = parser.parse_known_args()

    os.makedirs(args.out_dir, exist_ok=True)

    track_time = args.track_time
    combine_files = args.combine_files
    if combine_files:
        features = []
        labels = []

    if track_time:
        roling_average_time()
    
    for f in tqdm(os.listdir(args.data_dir)):
        if f.endswith(".xyz"):
            atoms = read_from_file(args.data_dir + f)


            slices = generate_slices(atoms, args.layer_height,
                                     args.z_start, args.z_end, args.contour_res, args.channels)

            fourier = generate_fourier_descriptions(slices, args.order)

            feature_vector = generate_feature_vector(atoms)

            if combine_files:
                features.append([fourier, feature_vector])
                labels.append(f[:-4])
            else:
                np.save(args.out_dir + f.replace(".xyz", "-fourier.npy"), fourier)
                np.save(args.out_dir + f.replace(".xyz", "-features.npy"), feature_vector)

            if track_time:
                roling_average_time('iteration')


    if track_time: 
       print("Average encoding time: " + str(roling_average_time.average / roling_average_time.iterations))

    if combine_files:
        np.save(args.out_dir + "features.npy", np.array(features, dtype=object))
        np.save(args.out_dir + "labels.npy", np.array(labels))


if __name__ == "__main__":
    main()
