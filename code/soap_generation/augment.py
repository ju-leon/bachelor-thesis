import numpy as np
import math
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.visualize import view
import copy
from tqdm import tqdm


def z_rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def augment_elements(elems, labels, steps, names=None):

    names_available = names is not None

    elems_ag = []
    labels_ag = []
    names_ag = []
    for element, label, index in tqdm(zip(elems, labels, range(len(labels)))):
        random_pos = np.random.uniform(0, np.pi * 2, size=steps)
        for step in random_pos:
            rotation = z_rotation(step)

            element_rotated = copy.deepcopy(element)

            positions = element_rotated.get_positions()
            for x in range(len(positions)):
                element_rotated.positions[x] = rotation @ positions[x]

            elems_ag.append(element_rotated)
            labels_ag.append(label)

            if names_available:
                names_ag.append(names[index])

    if names_available:
        return elems_ag, labels_ag, names_ag

    return elems_ag, labels_ag
