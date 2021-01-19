import numpy as np
import math
from ase.io import read
from ase.build import molecule
from ase import Atom, Atoms
from ase.visualize import view

from numpy import cross, eye, dot
from scipy.linalg import expm, norm


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2, normal):
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    angle = np.arctan2(length(cross), dot)

    if np.dot(normal, cross) < 0:
        angle = -angle

    return angle


def get_normal(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)

    return cp


def get_rotation_matrix(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))


def align_elements(elems):

    for element in elems:
        # Center elements
        element.positions -= element.get_positions()[0]

        positions = element.get_positions()

        mean = positions[-1] / np.linalg.norm(positions[-1])

        normal = get_normal(
            mean, np.array([0, 0, 1]), np.array([0, 0, 0]))

        angle_vec = angle(mean, [0, 0, 1], normal)

        rotation = get_rotation_matrix(normal, angle_vec)

        for x in range(len(positions)):
            element.positions[x] = rotation @ positions[x]

        positions = element.get_positions()

    return elems
