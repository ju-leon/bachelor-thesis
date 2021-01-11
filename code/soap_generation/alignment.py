import numpy as np
import math
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.visualize import view


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    angle = np.arccos(dotproduct(v1, v2) / (length(v1) * length(v2)))
    if(np.amin(np.cross(v1, v2)) < 0):
        angle = -angle
    return(angle)


def cosAngle(v1, v2):
    return dotproduct(v1, v2) / (length(v1) * length(v2))


def sinAngle(v1, v2):
    return dotproduct(v1, v2) / (length(v1) * length(v2))


def get_rotation_matrix(mean):
    """
    Returns a rotation matrix that rotates the space so that mean is on the y axis, so mean has
    coordinates [0,0,z] after transformation.
    """

    # If mean is already alligned with z-axis rotation around z can be ignored
    if (mean[0] == 0 and mean[1] == 0):
        rotZ = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        angleZ = 1 * (angle([1, 0, 0], [mean[0], mean[1], 0]))
        rotZ = np.matrix([[np.cos(angleZ), np.sin(angleZ), 0],
                          [-np.sin(angleZ), np.cos(angleZ), 0],
                          [0, 0, 1]])

    angleY = 1 * (angle([0, 0, 1], mean))
    rotY = np.matrix([[np.cos(angleY), 0, np.sin(angleY)],
                      [0, 1, 0],
                      [-np.sin(angleY), 0, np.cos(angleY)]])

    rot = np.matmul(rotY, rotZ)

    result = np.array(rot) @ mean

    if result[0] > 1e-12:
        rotZ = np.matrix([[np.cos(-angleZ), np.sin(-angleZ), 0],
                          [-np.sin(-angleZ), np.cos(-angleZ), 0],
                          [0, 0, 1]])
        rot = np.matmul(rotY, rotZ)

    return rot


def align_elements(elems):

    for p in range(5):
        for element in elems:
            mean = (element.get_positions()[1] - element.get_positions()[0])

            rotation = get_rotation_matrix(mean)
            positions = element.get_positions()

            for x in range(len(positions)):
                element.positions[x] = rotation @ (positions[x] - positions[0])

    for p in range(5):
        for element in elems:
            mean = ((element.get_positions()[
                    1] + element.get_positions()[2]) / 2) - element.get_positions()[0]

            rotation = get_rotation_matrix(mean)
            positions = element.get_positions()

            for x in range(len(positions)):
                element.positions[x] = rotation @ (positions[x] - positions[0])

    return elems
