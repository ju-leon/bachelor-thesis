import os
import numpy as np
from numpy import linalg as LA
import networkx as nx
from itertools import product
import math
from .contour_finder import find_contour
from .definitions import Atom, Circle

def radius_at_height(radius, height):
    """
    Returns the radius of a sphere at a height.
    Height is relative to the center of the sphere.
    """
    relativeHeight = (height - (radius)) / radius
    if relativeHeight <= -1 or relativeHeight >= 1:
        return 0
    return np.sin(np.arccos(relativeHeight)) * radius


def slice_catalyst(atoms, layer_height, z_start, z_end, resolution):
    """
    Slices a single catalyst.
    The reaction pocket is ignored and not added to the slices.
    """
    
    slice_heights = np.arange(z_start, z_end, layer_height)

    # Remove reaction pocket from slices
    atoms[1] = Atom("H", [0,0,0], 0)
    atoms[2] = Atom("H", [0,0,0], 0)

    slices = []
    for height in slice_heights:
        circles = []
        for (element, (x, y, z), radius) in atoms:
            delta_z = height - z
            circle_radius = radius_at_height(radius, delta_z)
            if circle_radius != 0:
                circles.append(Circle([x, y], circle_radius))

        slices.append(find_contour(circles, resolution))
    
    return slices
    

