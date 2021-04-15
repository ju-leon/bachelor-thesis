import numpy as np
import scipy.stats as st
import math
from tqdm import tqdm 


class VoxelGenerator():
    
    
    def __init__(self, species, scale, sigma=4, resolution=100):
        self.species = species
        self.scale = scale
        self.resolution = resolution
        self.sigma = sigma
        
    def gkern(self, middle):
        """Returns a 3D Gaussian kernel.""" 
        middle = np.rint(middle)

        width = math.floor(self.resolution / 2)
        x = np.arange(-width - middle[0],width - middle[0],1)
        y = np.arange(-width - middle[1],width - middle[1],1)
        z = np.arange(-width - middle[2],width - middle[2],1)
        xx, yy, zz = np.meshgrid(x,y,z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*self.sigma**2))
        return kernel


    def voxelise(self, element):
        voxel_rep = np.zeros((len(self.species),self.resolution,self.resolution,self.resolution))

        for atom, location in zip(element.get_chemical_symbols(), element.get_positions()):
            atom_index = self.species.index(atom)
            voxel_rep[atom_index] += self.gkern((location/self.scale) * self.resolution)
            
        voxel_rep = np.stack(voxel_rep, axis=-1)
        
        return voxel_rep
    
    def generate_voxel(self, elements):

        voxels = []

        for element in tqdm(elements):
            voxels.append(self.voxelise(element))
            
        return voxels
        