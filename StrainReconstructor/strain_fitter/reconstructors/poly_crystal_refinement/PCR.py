from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from ImageD11 import parameters, grain
from xfab import tools

from strain_fitter.utils.field_converter import FieldConverter

from PCR_core import setup_grain, find_refl_func, jacobian_finder

class PCR(object):

    def __init__(self, param_file, cif_file, omegastep):
        self.params = parameters.read_par_file( param_file )
        self.cif_file = cif_file
        self.omegastep = omegastep
        self.field_converter = FieldConverter()

    def reconstruct(self, flt, grains, number_y_scans, ymin, ystep, grain_topology_masks ):

        rows, cols = grain_topology_masks[0].shape
        field_recons = self.field_converter.initiate_field_dict(rows, cols)

        for i,g in enumerate(grains):

            active = grain_topology_masks[i]
            # plt.imshow(active)
            # plt.show()
            #continue
            voxels_as_grain_objects = self.run_pcr(self.cif_file, \
                                 active, flt, g, number_y_scans, ymin, ystep)

            ii, jj = np.mgrid[ 0:rows, 0:rows ] - rows//2
            ii = -1*ii

            for j,(ix, iy) in enumerate(zip(ii[active], jj[active])):
                index1 = -(ix - ii[0,0])
                index2 = iy - jj[0,0]
                voxel = voxels_as_grain_objects[index1, index2]
                row = ix + rows//2
                col = iy + rows//2
                self.field_converter.add_voxel_to_field(voxel, field_recons, row , col, self.params)
                sys.stdout.flush()
                print('Done PCR for '+str(i+1)+' of '+str(len(grains))+' grains',end='\r')
        print('Done PCR for '+str(len(grains))+' of '+str(len(grains))+' grains\n')
        return field_recons

    def run_pcr(self, cif_file, grain_mask, flt, gr, number_y_scans, ymin, ystep):

        no_voxels = sum(grain_mask[grain_mask==1])

        unit_cell = self.field_converter.extract_cell( self.params )

        UBI = gr.ubi.copy()
        cell = tools.ubi_to_cell( UBI )
        U, strain = tools.ubi_to_u_and_eps( UBI, unit_cell )
        euler = tools.u_to_euler( U )

        average = np.concatenate((strain,euler),axis=None)
        initial_guess = np.concatenate((average,average),axis=None)

        for i in range(no_voxels-2):
            initial_guess = np.concatenate((initial_guess,average),axis=None)

        th_min,th_max = setup_grain.get_theta_bounds(flt,gr)
        param = setup_grain.setup_experiment(self.params, cif_file,no_voxels,ystep,th_min,th_max)
        measured_data = setup_grain.get_measured_data(gr,flt, number_y_scans,ymin,ystep,param)

        hkl = setup_grain.get_hkls(param, flt, gr)
        voxel_positions, C, constraint = setup_grain.get_positions(grain_mask,ystep)

        voxel_data = find_refl_func.find_refl_func(param, hkl, voxel_positions, measured_data, unit_cell, initial_guess, ymin, number_y_scans, C, constraint)

        bounds_low,bounds_high = setup_grain.get_bounds_strain(initial_guess)

        solution = voxel_data.steepest_descent(bounds_low,bounds_high,initial_guess)

        voxels_as_grain_objects = setup_grain.map_solution_to_voxels(solution,grain_mask,no_voxels,voxel_data,var_choice='strain')

        return voxels_as_grain_objects



