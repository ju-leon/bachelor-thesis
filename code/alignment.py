from definitions import Atom
import numpy as np
import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def cosAngle(v1, v2):
  return dotproduct(v1, v2) / (length(v1) * length(v2))

def sinAngle(v1, v2):
  return dotproduct(v1, v2) / (length(v1) * length(v2))

def get_rotation_matrix(mean):
    """
    Returns a rotation matrix that rotates the space so that mean is on the y axis, so mean has
    coordinates [0,0,z] after transformation.
    """

    angleY =  -1 * (angle([0,0,1], [mean[0],0,mean[2]]))
    cosAngleY = -1 * (cosAngle([0,0,1], [mean[0],0,mean[2]]))
    #Rotation matric around y-axis
    rotY = np.matrix([[cosAngleY, 0, np.sin(angleY)], [0,1,0], [-np.sin(angleY), 0, cosAngleY]])

    angleX = -1 * (angle([1,0,0], [mean[0], mean[1], 0]))
    cosAngleX = -1 * (cosAngle([1,0,0], [mean[0], mean[1], 0]))
    #Rotation matrix around x-axis
    rotX = np.matrix([[1,0,0], [0,cosAngleX,-np.sin(angleX)], [0, np.sin(angleX),cosAngleX]])

    rot = np.matmul(rotX, rotY)
    return rot
    

def align_catalyst(atoms):
    # Metal should be first atom in xyz file
    metal_location = atoms[0].location

    # Center molecule to metal atom
    atoms_aligned = [Atom(elem, location - metal_location, radius)
             for (elem, location, radius) in atoms]
    
    # The reaction pocket (two hydrogen atoms) should be the second and third atom in xyz file
    mean = (atoms[1].location + atoms[2].location) / 2
    rotation = get_rotation_matrix(mean)

    # Rotate all atoms using the rotation matrix
    atoms_aligned = [Atom(elem, np.asarray(np.dot(rotation, location)).reshape(-1), radius)
             for (elem, location, radius) in atoms]

    return atoms_aligned
    