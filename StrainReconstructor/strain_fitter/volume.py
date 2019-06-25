import numpy as np
from ImageD11 import columnfile
from pyevtk.hl import gridToVTK
from strain_fitter.utils.topology import Topology
from strain_fitter.utils.peak_mapper import PeakMapper
from strain_fitter.utils.field_converter import FieldConverter
from strain_fitter.utils.slice_matcher import SliceMatcher
from ImageD11 import grain
from strain_fitter.utils import measurement_converter as mc
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import copy

class Volume(object):

    def __init__(self, reconstructor, omegastep, hkltol, nmedian, rcut, save_dir="."):
        self.slices = {}
        self.reconstructor = reconstructor

        self.omegastep = omegastep
        self.omslop = omegastep/2.0
        self.hkltol = hkltol
        self.nmedian = nmedian
        self.rcut = rcut
        
        self.save_dir = save_dir

        self.peak_mapper = PeakMapper()
        self.topology = Topology()

    def reconstruct(self, flt_files, grain_files, labels, z_positions=None, selected_grain=None):
        '''
        Take a series of flt file paths (strings) and read them 
        into columnfiles one by one. Runs the reconstruction on
        a per slice basis. The files should be preprocessed with
        spline correction performed and a dty column added. Labels
        are the labels the slices will be saved by.
        '''

        if z_positions is None:
            z_positions = range(len(labels))
        # The reference slice is used to define what grain is to be reconstructed
        # if selected grian is None the reference is simply the first in labels
        if selected_grain is not None:
            indx = selected_grain['index']
            label = selected_grain['label']
            self.sm = SliceMatcher( indx, label )
            for i in range(len(labels)):
                if self.sm.label==labels[i]: start = i
        else:
            start=0
            self.sm = None

        # recon all slices above reference slice and the reference slice
        for i in range(start, len(labels), 1):
            self.recon_slice( flt_files[i], grain_files[i], labels[i])
        
        # recon all slices below reference slice
        for i in range(start-1, -1, -1):
            self.sm.reset_reference()
            self.recon_slice( flt_files[i], grain_files[i], labels[i])

        print('Finished reconstruction of volume, saving to vtk and npy..')
        slices = self.slice_dict_to_list(labels, z_positions)
        self.save_volume_as_vtk(slices, interpolate=True)
        self.save_volume_as_npy(slices)

    def recon_slice(self, flt_file, grain_file, label):

        print("\n \nStarting reconstruction of slice: " + label)
        flt = columnfile.columnfile( flt_file )
        grains = grain.read_grain_file( grain_file )


        self.ymin, self.ystep, self.number_y_scans = self.extract_scan_geometry( flt )
        grain_topology_masks = self.recon_topology( flt, grains )


        g, s = self.select_grains( grains, grain_topology_masks )
        
        recons = self.reconstructor.reconstruct( flt, g, self.number_y_scans,\
                                                 self.ymin, self.ystep, s )
        
        print( "Recon shape: ", recons['E11'].shape )
        print("\n Finished reconstruction of slice: " + label)
        print("Saving results..")
        self.slices[label] = recons
        self.save_slice_as_npy( recons, label )

        return recons


    def recon_topology(self, flt, grains):
        # Assign peaks to grains, updates flt cols and g.mask
        self.peak_mapper.map_peaks(flt, grains, self.reconstructor.params, \
                self.omslop, self.hkltol, self.nmedian, self.rcut,\
                self.ymin, self.ystep, self.number_y_scans)

        # Reconstruct the grain topology
        grain_topology_masks, grain_recons = self.topology.FBP_slice(grains, \
                                                flt, self.omegastep, self.rcut, \
                                                self.ymin, self.ystep, self.number_y_scans)

        return grain_topology_masks


    def select_grains(self, grains, grain_topology_masks):

        if self.sm is None:
            print("sm is none")
            return grains, grain_topology_masks
        elif self.sm.ref_grain is None:
            s = grain_topology_masks[self.sm.index]
            g = grains[self.sm.index]
            self.sm.set_reference(s,g)

        indx = self.sm.match( grains, grain_topology_masks )
        if indx is None:
            print("indx is none")
            return [], grain_topology_masks
        else:
            print("single index")
            self.sm.set_reference( grain_topology_masks[indx], grains[indx] )
            return [grains[indx]], [grain_topology_masks[indx]]

    def save_slice_as_npy(self, recons, label):
        for key in self.reconstructor.field_converter.field_keys:
            np.save(self.save_dir+"/slice_z"+label+"_"+key, recons[key] )

    def save_volume_as_npy(self, slices):
        for key in self.reconstructor.field_converter.field_keys:
            field = [slice[key] for slice in slices]
            arr = np.dstack(field)
            np.save(self.save_dir + "/volume"+"_"+key, arr)

    def save_volume_as_vtk(self, slices, interpolate=True):
        for key in self.reconstructor.field_converter.field_keys:
            field = [slice[key] for slice in slices]
            arr = np.dstack(field)

            if interpolate==True:
                arr = self.interpolate_volume(arr)

            x = np.arange(0, arr.shape[0]+1, dtype=np.int32)
            y = np.arange(0, arr.shape[1]+1, dtype=np.int32)
            z = np.arange(0, arr.shape[2]+1, dtype=np.int32)
            gridToVTK(self.save_dir + "/volume"+"_"+key, x, y, z, cellData = {key: arr})

    def extract_scan_geometry( self, flt ):
        existing_dtys = np.sort( np.unique(flt.dty) )
        ymin = -np.max([abs(existing_dtys[0]),abs(existing_dtys[-1])])
        ystep = np.abs( existing_dtys[0]-existing_dtys[1] )
        number_y_scans = np.round((abs(ymin)/ystep)*2).astype(int)+1
        for i in range(len(existing_dtys)-1):
            assert abs(existing_dtys[i]-existing_dtys[i+1])==ystep
        return ymin, ystep, number_y_scans

    def slice_dict_to_list(self, keys, z_positions):
        empty = copy.deepcopy( self.slices[keys[0]] )
        for key in self.reconstructor.field_converter.field_keys:
            empty[key][:,:] = np.nan
        s = [empty]*(z_positions[-1] - z_positions[0] + 1)
        start = np.min(np.array(z_positions))
        for pos,key in zip(z_positions, keys):
            s[pos-start] = self.slices[key]
        return s

    def interpolate_volume(self, volume):
        '''
        Fill volume by linear interpolation between slices.
        Can be usefull in the case of a non volume filling
        sampling in z.
        '''
        dx,dy,dz = volume.shape

        points = np.where( ~np.isnan(volume) )
        if dz==1:
            points = (points[0],points[1])
            values = volume[points]
            grid_x, grid_y = np.mgrid[0:dx-1:dx*1j, 0:dy-1:dy*1j]
            return griddata(points, values, (grid_x, grid_y), method='linear')
        else:
            values = volume[points]
            grid_x, grid_y, grid_z = np.mgrid[0:dx-1:dx*1j, 0:dy-1:dy*1j, 0:dz-1:dz*1j]
            return griddata(points, values, (grid_x, grid_y, grid_z), method='linear')



    def sinogram( self, grain_no, flt_file, grain_file ):

        flt = columnfile.columnfile( flt_file )
        grains = grain.read_grain_file( grain_file )
        g = grains[grain_no]

        params = self.reconstructor.params
        self.ymin, self.ystep, self.number_y_scans = self.extract_scan_geometry( flt )
        self.peak_mapper.map_peaks(flt, grains, params, \
                self.omslop, self.hkltol, self.nmedian, self.rcut,\
                self.ymin, self.ystep, self.number_y_scans)
        
        distance = params.get('distance')
        pixelsize = ( params.get('y_size') + params.get('z_size') ) / 2.0
        wavelength = params.get('wavelength')
        cell_original = [params.get('cell__a'),params.get('cell__b'),params.get('cell__c'),params.get('cell_alpha'),params.get('cell_beta'),params.get('cell_gamma')]
        
        strains, directions, omegas, dtys, weights, tths, etas, intensity, sc, G_omegas, hkl = mc.extract_strain_and_directions(cell_original, wavelength, distance, pixelsize, g, flt, self.ymin, self.ystep, self.omegastep, self.number_y_scans)

        hkl_eta = np.zeros((hkl.shape[0],4))
        hkl_eta[:,0:3]=hkl[:,:]
        hkl_eta[:,3] = np.sign(etas[:])

        sinogram_strain = np.zeros(( int(-2*(self.ymin/self.ystep)), len(np.unique(hkl_eta,axis=0)) ))
        sinogram_tth = np.zeros(( int(-2*(self.ymin/self.ystep)), len(np.unique(hkl_eta,axis=0)) ))
        sinogram_eta = np.zeros(( int(-2*(self.ymin/self.ystep)), len(np.unique(hkl_eta,axis=0)) ))
        sinogram_int = np.zeros(( int(-2*(self.ymin/self.ystep)), len(np.unique(hkl_eta,axis=0)) ))
        indx = np.argsort(omegas)
        hkl_sorted=hkl_eta[indx,:]
        _,ind = np.unique(hkl_sorted,axis=0,return_index=True)
        miller = hkl_sorted[np.sort(ind),:]

        for (s,o,y,w,I,m,t,e) in zip(strains[indx], omegas[indx], dtys[indx], weights[indx], intensity[indx], hkl_sorted, tths[indx], etas[indx]):
            i = int( (y+self.ymin)/self.ystep )
            #j = (np.min( np.abs(omega_range-o) )==np.abs(omega_range-o))
            j = np.where( (miller==m).all(axis=1) )[0]
            assert len(j)==1
            #print( np.unique(hkl[indx,:],axis=0) )
            #print(o,j,miller)

            sinogram_int[i,j] = I
            sinogram_strain[i,j] = s*w
            sinogram_tth[i,j] = t
            print(t)
            sinogram_eta[i,j] = e

        #TODO: Morivate why we do this.. ask Jon...
        #      make sinograms for tth shifts, eta shifts and strain
        #      make script to generate data of reconstructed slice.
        #      put all resulting nine sinograms in the paper.

        sinogram_int = sinogram_int / np.max(sinogram_int, axis=0)
        sino_mask = sinogram_int>0.2
        sinogram_strain = sinogram_strain*sino_mask
        sinogram_tth = sinogram_tth*sino_mask
        sinogram_eta = sinogram_eta*sino_mask

        #Relative deviation from reflection mean value
        m = np.sum(sinogram_tth, axis=0)/np.sum(sino_mask,axis=0)
        mean_tth = np.tile(m, (sinogram_tth.shape[0],1))
        print(mean_tth)
        sinogram_tth = sinogram_tth - mean_tth

        m = np.sum(sinogram_eta, axis=0)/np.sum(sino_mask,axis=0)
        mean_eta = np.tile(m, (sinogram_eta.shape[0],1))
        sinogram_eta = sinogram_eta - mean_eta

        m = np.sum(sinogram_strain, axis=0)/np.sum(sino_mask,axis=0)
        mean_strain = np.tile(m, (sinogram_strain.shape[0],1))
        sinogram_strain = sinogram_strain - mean_strain

        sinogram_tth[~sino_mask] = np.nan
        sinogram_eta[~sino_mask] = np.nan
        sinogram_strain[~sino_mask] = np.nan

        s = 20
        t = 15
        plt.figure(1)
        plt.imshow(sinogram_int.T)
        plt.xlabel(r'sample y-translation',size=s)
        plt.ylabel(r'Miller reflections (sorted by $\omega$)',size=s)
        plt.title(r'Intensity',size=s)
        c1 = plt.colorbar()
        c1.ax.tick_params(labelsize=t)
        plt.tick_params(labelsize=t)

        plt.figure(2)
        plt.imshow(sinogram_strain.T)
        plt.xlabel(r'sample y-translation',size=s)
        plt.ylabel(r'Miller reflections (sorted by $\omega$)',size=s)
        plt.title(r'Shift in strain $\varepsilon_m$',size=s)
        c2 = plt.colorbar()
        c2.ax.tick_params(labelsize=t)
        plt.tick_params(labelsize=t)

        plt.figure(3)
        plt.imshow(sinogram_tth.T)
        plt.xlabel(r'sample y-translation',size=s)
        plt.ylabel(r'Miller reflections (sorted by $\omega$)',size=s)
        plt.title(r'Shift in $2\theta$',size=s)
        c3 = plt.colorbar()
        c3.ax.tick_params(labelsize=t)
        plt.tick_params(labelsize=t)

        plt.figure(4)
        plt.imshow(sinogram_eta.T)
        plt.xlabel(r'sample y-translation',size=s)
        plt.ylabel(r'Miller reflections (sorted by $\omega$)',size=s)
        plt.title(r'Shift in $\eta$',size=s)
        c4 = plt.colorbar()
        c4.ax.tick_params(labelsize=t)
        plt.tick_params(labelsize=t)

        plt.show()
