from collections import namedtuple


Point = namedtuple('Point', ['x','y'])


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
