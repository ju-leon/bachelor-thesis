import os
import numpy as np
from numpy import linalg as LA
import networkx as nx
from itertools import product
import math
from .contour_finder import find_contour
from .definitions import Atom, Circle
import cv2 as cv

def radius_at_height(radius, height):
    """
    Returns the radius of a sphere at a height.
    Height is relative to the center of the sphere.
    """
    if radius == 0:
        return 0

    relativeHeight = (height - (radius)) / radius
    if relativeHeight <= -1 or relativeHeight >= 1:
        return 0
    return np.sin(np.arccos(relativeHeight)) * radius


def slice_to_contour(atoms, layer_height, z_start, z_end, resolution, channels=["X"]):
    """
    Slices a single catalyst into contours for each slice.
    The reaction pocket is ignored and not added to the slices.
    Channels:
        X: All, combine all atoms into single channel
        H: Only hydrogen atoms in one channel
        C: Only carbon atoms in one channel
    """

    slice_heights = np.arange(z_start, z_end, layer_height)

    # Remove reaction pocket from slices
    atoms[1] = Atom("H", [0, 0, 0], 0)
    atoms[2] = Atom("H", [0, 0, 0], 0)

    slices = []
    for height in slice_heights:
        channel_circles = []
        for channel in channels:
            circles = []
            for (element, (x, y, z), radius) in atoms:
                delta_z = height - z
                circle_radius = radius_at_height(radius, delta_z)
                if circle_radius != 0 and (channel == "X" or element == channel):
                    circles.append(Circle([x, y], circle_radius))

            channel_circles.append(find_contour(circles, resolution))

        slices.append(channel_circles)

    return slices


def coordinate_to_grid(point, box, scale=1):
    (x,y) = point
    (x_box, y_box) = box

    return [(x * scale) + (x_box / 2), (y * scale) + (y_box / 2)]


def slice_to_map(atoms, layer_height, z_start, z_end, resolution, channels=["X"]):
    """
    Slices a single catalyst into a map for each channel.
    The reaction pocket is ignored and not added to the slices.
    Channels:
        X: All, combine all atoms into single channel
        H: Only hydrogen atoms in one channel
        C: Only carbon atoms in one channel
    """
    slice_heights = np.arange(z_start, z_end, layer_height)

    # Remove reaction pocket from slices
    atoms[1] = Atom("H", [0, 0, 0], 0)
    atoms[2] = Atom("H", [0, 0, 0], 0)

    slices = []
    for height in slice_heights:
        channel_maps = []
        for channel in channels:
            circles = []
            for (element, (x, y, z), radius) in atoms:
                delta_z = height - z
                circle_radius = radius_at_height(radius, delta_z)
                if circle_radius != 0 and (channel == "X" or element == channel):
                    circles.append(Circle([x, y], circle_radius))

            box = [100,100]
            slice_map = np.zeros(box)
            scale = 8

            for circle in circles:
                ellipse_float = (coordinate_to_grid(circle.point, box, scale), (2 * circle.radius * scale, 2 * circle.radius * scale), 0.0)
                cv.ellipse(slice_map, ellipse_float, 1, -1)

            channel_maps.append(slice_map)

        slices.append(np.dstack(channel_maps))

    return slices
