from pyefd import elliptic_fourier_descriptors
import numpy as np


def fourier_descriptor(contour, order):
    """
    Describe a contour using a fourier descriptor.
    Descritor is normailzed to provide roational invariance
    """

    if contour == []:
        return np.zeros((order, 4))

    return elliptic_fourier_descriptors(np.squeeze(contour), order, normalize=True)