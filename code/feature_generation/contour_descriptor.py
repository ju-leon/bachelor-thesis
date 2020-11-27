# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/leon/Files/pyefd')
import pyefd

from pyefd import elliptic_fourier_descriptors, normalize_efd, calculate_dc_coefficients, elliptic_fourier_features, reconstruct_contour_from_features
import numpy as np


def fourier_descriptor(contour, order):
    """
    Describe a contour using a fourier descriptor.
    Describtor is normailzed to provide roational invariance:
    In the normalizing process coeffs [0][1], [0][2] will become zero.
    They will be enriched with information about the location of the contour to remove translation invariance.
    """

    if contour == []:
        return np.zeros((order, 4)).flatten()

    #fourier = elliptic_fourier_descriptors(
    #    np.squeeze(contour), order, normalize=False)
    #normals = normalize_efd(fourier, size_invariant=False)
    #A0, C0 = calculate_dc_coefficients(contour)

    # Get the length of the center-offset(locus) vector
    #length = np.sqrt((A0 ** 2) + (C0 ** 2))
    #vec = np.append(normals.flatten(), length)

    fourier = elliptic_fourier_features(np.squeeze(contour), order)

    print(reconstruct_contour_from_features(fourier))
    # Get the center offset of the contour
    [A0, C0] = fourier[-2:]
    # Calculate length from offset vecto
    length = np.sqrt((A0 ** 2) + (C0 ** 2))
    # Affend length to features
    fourier = np.append(fourier[:-2], length)

    return fourier