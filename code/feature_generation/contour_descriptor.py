from pyefd import elliptic_fourier_descriptors
import numpy as np


def fourier_descriptor(contour, order):
    if contour == []:
        return np.zeros((order, 4))

    return elliptic_fourier_descriptors(np.squeeze(contour), order, normalize=True)