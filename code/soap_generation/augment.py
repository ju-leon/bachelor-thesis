import numpy as np
import math
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.visualize import view
import copy


def z_rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def augment_elements(elems, labels, steps):
    steps = np.linspace(0, np.pi * 2, num=steps, endpoint=False)

    elems_ag = []
    labels_ag = []

    for element, label in zip(elems, labels):
        for step in steps:
            rotation = z_rotation(step)

            element_rotated = copy.deepcopy(element)

            positions = element_rotated.get_positions()
            for x in range(len(positions)):
                element_rotated.positions[x] = rotation @ positions[x]

            elems_ag.append(element_rotated)
            labels_ag.append(label)

    return elems_ag, labels_ag