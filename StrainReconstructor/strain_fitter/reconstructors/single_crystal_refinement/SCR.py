import numpy as np
import matplotlib.pyplot as plt

from ImageD11 import parameters, grain
from strain_fitter.utils.grain_fitter import GrainFitter
from strain_fitter.utils.field_converter import FieldConverter
import copy

class SCR(object):

    def __init__(self, param_file):
        
        self.params = parameters.read_par_file( param_file )
        
        self.grain_fitter = GrainFitter()
        self.field_converter = FieldConverter()

    def reconstruct(self, flt, grains, number_y_scans, ymin, ystep, grain_topology_masks ):

        rows, cols = grain_topology_masks[0].shape
        field_recons = self.field_converter.initiate_field_dict(rows, cols)

        for i,g in enumerate(grains):

            active = grain_topology_masks[i]
            ii, jj = np.mgrid[ 0:rows, 0:rows ] - rows//2
            ii = -1*ii

            for j,(ix, iy) in enumerate(zip(ii[active], jj[active])):
                voxel = self.fit_one_point( g, flt, self.params, ix, iy, ystep )
                row = ix + rows//2
                col = iy + rows//2
                self.field_converter.add_voxel_to_field(voxel, field_recons, row , col, self.params)

        return field_recons

    def fit_one_point(self, gr, flt, pars, ix, iy, ystep ):
        """
        Take each time the nearest point in dty (not the mask!)
        """

        om = np.radians( flt.omega[gr.mask] )
        co = np.cos( om )
        so = np.sin( om )
        #idtycalc = np.round(-ix * so + iy * co) # this seeems wrong?! why mirror across y!?
        idtycalc = np.round(ix * so + iy * co)
        idty = np.round(flt.dty[ gr.mask ] / ystep)
        #     m = abs(dty - dtycalc) < ystep*0.75
        m = idtycalc == idty

        voxel = grain.grain( gr.ubi )
        voxel.hkl = gr.hkl[:,m]
        voxel.etasigns = gr.etasigns[m]
        inds = np.arange( flt.nrows, dtype=int )
        voxel.mask = np.zeros( flt.nrows, np.bool )

        
        voxel.mask[ inds[gr.mask][m] ] = True

        self.grain_fitter.fit_one_grain( voxel, flt, pars )
        return voxel




