from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

Circle = namedtuple('Circle', ['point', 'radius'])

Atom = namedtuple('Atom', ['element', 'location', 'radius'])
