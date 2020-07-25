from __future__ import print_function
import numpy as np
from skimage.transform import iradon, radon
import sys
import matplotlib.pyplot as plt

class Topology(object):

    def __init__(self):
        '''
        Function class to reconstruct topology of grains
        '''
        pass
    
    def FBP_slice(self, grains, flt, omegastep, rcut, ymin, ystep, number_y_scans):
        grain_masks=[]
        grain_recons=[]
        for i,g in enumerate(grains):

            sys.stdout.flush()
            print('Done FBP for '+str(i+1)+' of '+str(len(grains))+' grains',end='\r')

            sinoangles, sino, recon = self.FBP_grain( g, flt, \
                        ymin, ystep, omegastep, number_y_scans )
            normalised_recon = recon/recon.max()
            grain_recons.append(normalised_recon)
            mask = normalised_recon > rcut
            grain_masks.append(mask)

            #plt.imshow(mask)
            #plt.title(i)
            #plt.show()

        print('Done FBP for '+str(len(grains))+' of '+str(len(grains))+' grains\n')
        self.update_grainshapes(grain_recons,grain_masks)
        return grain_masks, grain_recons

    def FBP_grain(self, g, flt, ymin, ystep, omegastep, number_y_scans ):
        '''
        Take an ImageD11 grain object and a flt peaks file.
        Reconstruct the topology based on the peak mask provided
        by the grain object. The used method is Filtered Back Projection.
        '''

        iy = np.round( (flt.dty[ g.mask ] - ymin) / ystep ).astype(int)
        print('')
        print('iy    :    ', iy)
        print('ymin  :    ', ymin)
        print('ystep :    ', ystep)
        print('')

        omega = np.round( flt.omega[ g.mask ] / omegastep ).astype(int)

        keys = [ (hkl[0], hkl[1], hkl[2], int(s))
                for hkl, s in zip(g.hkl.T , g.etasigns)]

        uni = self.uniq(keys)
        akeys = np.array( keys )

        sum_intensity = flt.sum_intensity[ g.mask ]
        assert (sum_intensity > 0).all(), "peaks are positive"

        npks = len( uni )
        sino = np.zeros( ( npks, number_y_scans ), np.float )
        angs = np.zeros( ( npks, number_y_scans ), np.float )

        for refi,u in enumerate(uni):
            mask = (akeys == u).astype(int).sum(axis=1) == 4
            dtypos = iy[mask]
            intensities = sum_intensity[mask]
            angles = omega[mask]
            for yindex, counts, omegapk in zip( dtypos, intensities, angles ):
                if counts > 0:
                    sino[refi][yindex] = sino[refi][yindex] + counts
                    angs[refi][yindex] = omegapk

        sinoangles = np.sum( angs, axis = 1) / np.sum( sino > 0, axis = 1)
        sino = (sino.T/sino.max( axis=1 )).T
        order = np.argsort( sinoangles )
        sinoangles = sinoangles[order]
        ssino = sino[order].T
        output_size = int(number_y_scans)
        back_projection = iradon( ssino, theta=sinoangles, output_size=output_size,  circle = False)

        return sinoangles, ssino, back_projection 

    def uniq(self, vals):
            d = {}
            newvals = []
            for v in vals:
                if v not in d:
                    d[v]=0
                    newvals.append(v)
            return newvals
    
    def update_grainshapes(self, grain_recons, grain_masks):
        '''
        Update a set of grain masks based on their overlap and intensity.
        At each point the grain with strongest intensity is assigned

        Assumes that the grain recons have been normalized
        '''

        for i in range(grain_recons[0].shape[0]):
            for j in range(grain_recons[0].shape[1]):
                if self.conflict_exists(i,j,grain_masks):
                    max_int = 0.0
                    leader  = None
                    for n,grain_recon in enumerate(grain_recons):
                        if grain_recon[i,j]>max_int:
                            leader = n
                            max_int = grain_recon[i,j]

                    #The losers are...
                    for grain_mask in grain_masks:
                        grain_mask[i,j]=0

                    #And the winner is:
                    grain_masks[leader][i,j]=1

    def conflict_exists(self, i,j,grain_masks):
        '''
        Help function for update_grainshapes()
        Checks if two grain shape masks overlap at index i,j
        '''
        claimers = 0
        for grain_mask in grain_masks:
            if grain_mask[i,j]==1:
                claimers+=1
        if claimers>=2:
            return True
        else:
            return False