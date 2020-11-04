import math
import numpy as np
from collections import namedtuple
from itertools import product
import networkx as nx
from numpy import linalg as LA
import copy

ContourSection = namedtuple(
    'ContourSection', ['index', 'start_angle', 'end_angle'])


def distance(p1, p2):
    return LA.norm(np.array(p1) - np.array(p2))


def circle_intersection(circle1, circle2):
    """
    Calculates the points of intersection for 2 circles
    """
    ([x0, y0], r0) = circle1
    ([x1, y1], r1) = circle2
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1:
        return []
    # One circle within other
    if d < abs(r0-r1):
        return []
    # coincident circles
    if d == 0 and r0 == r1:
        return []
    else:
        a = (r0**2-r1**2+d**2)/(2*d)
        h = math.sqrt(r0**2-a**2)
        x2 = x0+a*(x1-x0)/d
        y2 = y0+a*(y1-y0)/d
        x3 = x2+h*(y1-y0)/d
        y3 = y2-h*(x1-x0)/d

        x4 = x2-h*(y1-y0)/d
        y4 = y2+h*(x1-x0)/d

        return ([x3, y3], [x4, y4])


def create_graph_from_circles(circles):
    if circles == []:
        return None

    G = nx.Graph()
    indexed_circles = list(zip(range(len(circles)), circles))

    for (index, ([x, y], radius)) in indexed_circles:
        G.add_node(index)

    # Compute distance from every circle to every other circle
    for ((index1, circle1), (index2, circle2)) in [(c1, c2) for c1 in indexed_circles for c2 in indexed_circles]:
        if index1 == index2:
            continue

        dist = distance(circle1.point, circle2.point)

        if dist < (circle1.radius + circle2.radius):
            intersect = circle_intersection(circle1, circle2)
            if intersect != []:
                G.add_edge(index1, index2,
                           intersect=intersect)

    return G


def cyclic_min_larger_than(array, lower_bound):
    """
    Determine index of the first angle greater than the current angle
    """
    normalised_array = (np.array(array) - lower_bound) % (2 * np.pi)
    valid_idx = np.where(normalised_array > 0)[0]
    return valid_idx[normalised_array[valid_idx].argmin()]


def angleFull(v1, v2):
    """
    Angle beweeen 2 vecs from 0 to 2 PI
    """
    (x1, y1) = v1
    (x2, y2) = v2
    dot = x1*x2 + y1*y2      # dot product
    det = x1*y2 - y1*x2      # determinant
    angle = np.arctan2(det, dot)
    return angle % (2*np.pi)


def points_in_section(circle, section):
    """
    Returns points of the contour of a circle in the given section
    """
    (index, start_angle, end_angle) = section
    ([x, y], radius) = circle

    if end_angle < start_angle:
        end_angle += 2*np.pi

    angles = np.arange(start_angle, end_angle, 0.01)

    coords_circle = []
    for angle in angles:
        coords_circle.append(
            ((np.cos(angle)*radius) + x, (np.sin(angle) * radius) + y))

    return coords_circle


def find_contour_for_cc(G, connected_component, circles):
    """
    Returns the contour for a connected compenent of a graph.
    Intersections must be pre-computed
    """

    # Make sure the original graph is not edited
    G = copy.deepcopy(G)

    # If there's only one circle return contour of that circle
    if(len(connected_component) == 1):
        return points_in_section(circles[max(connected_component)], ContourSection(0, 0, 2*np.pi))

    connected_component = list(connected_component)
    # Start at the point furthest to the right. This point is certainly in the contour
    biggest_x = []
    for node in connected_component:
        biggest_x.append(circles[node].point[0] + circles[node].radius)

    # Set node furthest to the right as active node
    active_node = connected_component[np.argmax(biggest_x)]
    current_angle = 0

    sections = []

    # Add a virtual node to the graph to detect when the contour tracer has reached the beginning again
    G.add_node(-1)
    G.add_edge(active_node, -1,
               intersect=[[np.max(biggest_x), circles[active_node].point[1]]])

    # Follow the contour of the circles anti-clockwise
    while True:
        angles = []
        data_for_angle = []
        for neighbor in G.neighbors(active_node):
            for point in G.edges[active_node, neighbor]['intersect']:
                angles.append(angleFull(
                    [1, 0], [point[0] - circles[active_node].point[0], point[1] - circles[active_node].point[1]]))
                data_for_angle.append((neighbor, point))

        next_node=data_for_angle[cyclic_min_larger_than(
            angles, current_angle)][0]
        intersection_point=data_for_angle[cyclic_min_larger_than(
            angles, current_angle)][1]

        contour_section=ContourSection(
            active_node, current_angle, angles[cyclic_min_larger_than(angles, current_angle)])

        sections.append(contour_section)

        # if the start is reached again a full contour is found
        if next_node == (-1):
            break

        current_angle = angleFull([1, 0], [
                                  intersection_point[0] - circles[next_node].point[0], intersection_point[1] - circles[next_node].point[1]])
        active_node = next_node

    # Generate contour poins from sections
    contour_points = []
    for section in sections:
        circle = circles[section.index]
        contour_points.extend(points_in_section(circle, section))

    return contour_points


def find_contour(circles):
    G = create_graph_from_circles(circles)

    if G == None:
        return []

    # Extract the larges connected component
    # TODO: The geograpically largest CC might not be the CC with the most nodes. Rework in future
    largest_cc = max(nx.connected_components(G), key=len)

    return np.array(find_contour_for_cc(G, largest_cc, circles))
