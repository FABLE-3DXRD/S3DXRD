import numpy as np
import matplotlib.pyplot as plt

from ImageD11 import parameters, grain
from xfab import tools

from strain_fitter.utils.field_converter import FieldConverter
from strain_fitter.utils import measurement_converter as mc

from ASR_core import algebraic_strain_refinement as asr
from ASR_core import mesher as mesher
from ASR_core import illustrate_mesh as illustrate_mesh

class ASR(object):

    def __init__(self, param_file, omegastep, gradient_constraint):

        self.params = parameters.read_par_file( param_file )
        self.omegastep = omegastep
        self.field_converter = FieldConverter()
        self.gradient_constraint = gradient_constraint

    def reconstruct(self, flt, grains, number_y_scans, ymin, ystep, grain_topology_masks):

        rows, cols = grain_topology_masks[0].shape
        field_recons = self.field_converter.initiate_field_dict(rows, cols)

        for i,g in enumerate(grains):
            
            active = grain_topology_masks[i]

            voxels, coordinates = self.run_asr( active, ystep, number_y_scans, ymin, g, flt )

            for voxel,c in zip(voxels, coordinates):
                row = int( (c[0]/ystep)  + rows//2 )
                col = int( (c[1]/ystep)  + rows//2 )
                self.field_converter.add_voxel_to_field(voxel, field_recons, row , col, self.params)

        return field_recons

    def run_asr(self, topology, ystep, number_y_scans, ymin, g, flt):
        origin = np.array([topology.shape[0]//2, topology.shape[0]//2])
        mesh = mesher.create_pixel_mesh(  topology, ystep, origin  )

        distance = self.params.get('distance')
        pixelsize = ( self.params.get('y_size') + self.params.get('z_size') ) / 2.0
        wavelength = self.params.get('wavelength')
        cell_original = [self.params.get('cell__a'),self.params.get('cell__b'),self.params.get('cell__c'),self.params.get('cell_alpha'),self.params.get('cell_beta'),self.params.get('cell_gamma')]

        strains, directions, omegas, dtys, weights, tths, etas, intensity, sc, G_omegas, hkl = mc.extract_strain_and_directions(cell_original, wavelength, distance, pixelsize, g, flt, ymin, ystep, self.omegastep, number_y_scans)
        
        # #With orient
        # UBs = asr.solve_with_orient( g, hkl, G_omegas, omegas, weights, mesh, dtys, ystep, cell_original )
        # voxels = []
        # coordinates = mesher.get_elm_centres( mesh )
        # #print("UBs.shape",UBs.shape)
        # for i in range(UBs.shape[0]):
        #     #print(UBs[i,:,:])
        #     UBI = np.linalg.inv(UBs[i,:,:])
        #     voxels.append( grain.grain( UBI ) )       
        # return voxels, coordinates

        # only strain
        
        # tikhonov_solve
        #eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy = asr.tikhonov_solve( mesh, directions, strains , omegas, dtys, weights, ystep, self.gradient_constraint )
        
        #trust region
        eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy = asr.trust_constr_solve( mesh, etas, hkl, tths, intensity, directions, strains , omegas, dtys, weights, ystep, self.gradient_constraint )

        voxels = []
        coordinates = mesher.get_elm_centres( mesh )
        for i,elm in enumerate(mesh):
            # [e11, e12, e13, e22, e23, e33] 
            epsilon_sample = np.array([[eps_xx[i], eps_xy[i], eps_xz[i]],
                                       [eps_xy[i], eps_yy[i], eps_yz[i]],
                                       [eps_xz[i], eps_yz[i], eps_zz[i]]])
            eps_cry = np.dot(np.dot(np.transpose(g.u),epsilon_sample),g.u)
            eps_cry_flat = [eps_cry[0,0], eps_cry[0,1], eps_cry[0,2], eps_cry[1,1], eps_cry[1,2], eps_cry[2,2] ]
            B = tools.epsilon_to_b(eps_cry_flat, cell_original)
            UBI = np.linalg.inv(np.dot(g.u,B))*(2*np.pi)
            voxels.append( grain.grain( UBI ) )

        return voxels, coordinates


        




