from pyefd import elliptic_fourier_descriptors, normalize_efd, calculate_dc_coefficients
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

    fourier = elliptic_fourier_descriptors(np.squeeze(contour), order, normalize=False)
    normals = normalize_efd(fourier, size_invariant=False)
    A0, C0 = calculate_dc_coefficients(contour)

    #normals[0][1] = A0
    #normals[0][2] = C0

    return normals.flatten()

