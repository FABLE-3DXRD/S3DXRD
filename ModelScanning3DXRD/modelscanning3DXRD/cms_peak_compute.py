'''
Class to perform merging of delta peaks.

It should be realised that the deployed algorithm for merging does not
account for the 3D projection of the voxel upon the detector. i.e
each delta peak is simply assigned to a single pixel center, while in reality
the resolution, in this aspect, can actually be better as the diffracted ray may
overlap two pixels and thus render a cms more true to it's state.

The algorithm here is, however, very much faster than any 3D projection algorithm.
At least according to the firm belif of the author :=)
'''

from xfab import detector
import numpy as np
from modelscanning3DXRD import variables
import matplotlib.pyplot as plt
import cluster
import copy

A_id = variables.refarray().A_id

class PeakMerger(object):

    def __init__(self, param):
        """
        PeakMerger is used to build upp all the 3D framestacks which will merge
        the delta peaks into cms peak positions such that the resolution is
        limited to the omega stepsize and the detector pixel size.

        The idea is offcourse that the merging of peaks is a more realistic model.
        """

        # Experimental setup
        self.voxel_size = param['beam_width']

        self.param = param

        self.omega_step = param['omega_step']
        self.omega_start = param['omega_start']
        self.omega_end = param['omega_end']
        self.omegas = np.arange(self.omega_start, self.omega_end, self.omega_step)

        self.dtys_index = self.get_dtys_index() #in voxel units

        self.peaks_by_dty={}
        for dty in self.dtys_index:
            self.peaks_by_dty[dty]=[]

        self.merged_peaks = []

    def add_reflection_to_images(self,reflection):
        """
        Add reflection to dty dictionary.

        reflection is a list that must follow the variables.refarray() format.

        Recall: detz and dety is zero in the rightmost bottom pixel. y and z increases as:

                    ^ z
                    |
            y       |
            <------ o

        """

        dty = int(np.round((reflection[A_id['dty']]/1000.)/self.voxel_size)) #since dty is in microns
        binary_refl = copy.copy(reflection)
        binary_refl[A_id['detz']] = np.round(reflection[A_id['detz']])
        binary_refl[A_id['dety']] = np.round(reflection[A_id['dety']])

        binary_refl[A_id['omega']] = np.radians( self.omegas[ np.argmin( np.abs( self.omegas - np.degrees(reflection[A_id['omega']]) ) ) ] )

        self.peaks_by_dty[dty].append(binary_refl)


    def analyse_images_as_clusters(self):
        '''
        Merge all delta peaks located in the dictionary self.peaks_by_dty.

        * self.peaks_by_dty uses discrete coordinates, i.e peaks are located at
          pixel centers and multiples of omegastep/2. only.

        * Clusters delta peaks by distance such that peaks in neighboring pixels
          are grouped together.

        * Comptes CMS based on coordinates and intenisty of all dleta peaks in the
          cluster.

        This method will build up the self.merged_peaks list attribute such that
        the peakmerger has all merged peaks in that list after execution.

        The clustering is based on sorting the peak set along each dimension
        such that the effort needed is proportional to sorting the delta peak set three times.
        '''

        index_dty = A_id['dty']
        index_z = A_id['detz']
        index_y = A_id['dety']
        index_om = A_id['omega']
        index_int = A_id['Int']
        index_nbr_of_pixels = A_id['Number_of_pixels']

        # Display diffraction image for debug
        # image = np.zeros((self.param['detz_size'],self.param['dety_size']))
        # for key in self.peaks_by_dty:
        #     for p in self.peaks_by_dty[key]:
        #         #if p[A_id['omega']==1]:
        #         i=self.param['detz_size'] - 1 - int(p[A_id['detz']])
        #         j=self.param['dety_size'] - 1 - int(p[A_id['dety']])
        #         if p[A_id['omega']]-1*180./np.pi<0.001:
        #             image[i,j]=1
        # image = detector.trans_orientation(image,self.param['o11'],self.param['o12'],self.param['o21'],self.param['o22'],'inverse')
        # plt.imshow(image,cmap='gray')
        # plt.show()

        tol_z = 20 #units of pixels
        tol_y = 20 #units of pixels
        tol_om = 2.01*(1/180.)*np.pi*self.omega_step #units of radians use 1.001 to avoid numerical error

        self.merged_peaks = []

        for key in self.peaks_by_dty:
            peaks = self.peaks_by_dty[key] #list of peaks for a single dty [[],[],[]]
            if peaks==[]:
                continue

            peak_clusters = cluster.cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)
            merged_peaks_dty = cluster.merge_many_clusters(peak_clusters,index_z,index_y,index_om,index_int,index_nbr_of_pixels)
            #self.plot_peak(peak_clusters,merged_peaks_dty,key, index_z, index_y, index_om)

            for peak in merged_peaks_dty:
                self.merged_peaks.append(peak)

        return self.merged_peaks

    def plot_cluster_3D(self,cluster,cms, dty, index_z, index_y, index_om):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p in cluster:
            print('delta peak at',p[index_y],p[index_om],p[index_z] )
            ax.scatter(p[index_y],p[index_om],p[index_z],marker='^',c="orange")
            #ax.text(g[0][index_y],g[0][index_om],g[0][index_z],str(i),size=20, zorder=1, color='k')
        ax.set_title('Delta peak cluster at dty_index='+str(dty))
        ax.set_xlabel('y')
        ax.set_ylabel('omega')
        ax.set_zlabel('z')
        ax.scatter(cms[index_y],cms[index_om],cms[index_z],marker='o', c="purple")
        plt.show()


    def plot_cluster_2D(self, cluster, cms, dty, index_z, index_y, index_om):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for p in cluster:
            print('delta peak at',p[index_y],p[index_z] )
            ax.scatter(p[index_y],p[index_z],marker='^',c="orange")
            #ax.text(g[0][index_y],g[0][index_om],g[0][index_z],str(i),size=20, zorder=1, color='k')
        ax.set_title('Delta peak cluster at dty_index='+str(dty))
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.scatter(cms[index_y],cms[index_z],marker='o', c="purple")
        ax.text(cms[index_y],cms[index_z],'CMS',size=20, zorder=1, color='k')
        ax.grid()
        ax.ticklabel_format(useOffset=False)
        plt.axis('equal')
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()


    def plot_peak(self, peak_clusters, merged_peaks_dty, dty, index_z, index_y, index_om):
        '''
        Take all peak clusters and plot a selected peak as a scatter plot
        '''

        for i,(cluster,cms) in enumerate(zip(peak_clusters,merged_peaks_dty)):
            print('cluster',i, 'cms', cms[index_y], cms[index_z])
            if len(cluster)>9:
                if abs(cms[index_y]-730)<5 and abs(cms[index_z]-349)<5:
                    self.plot_cluster_2D(cluster,cms,dty, index_z, index_y, index_om)


    def get_dtys_index(self):
        """
        Determine the list of all beam y-settings that could possible
        give rise to nonzero intensity. return settings in voxel units.

        e.g. +3 means +3 voxel widths positive in y (sample frame).
        """
        radius = np.zeros((self.param['no_voxels'],))
        Delta = (1/2.)*self.param['beam_width']
        for voxel_nbr in range( self.param['no_voxels'] ):
            voxel_pos = np.array(self.param['pos_voxels_%s' %(self.param['voxel_list'][voxel_nbr])])
            if voxel_pos[0]>0:
                x = voxel_pos[0] + Delta
            else:
                x = voxel_pos[0] - Delta
            if voxel_pos[1]>1:
                y = voxel_pos[1] + Delta
            else:
                y = voxel_pos[1] - Delta
            radius[voxel_nbr] = np.sqrt( x*x + y*y )

        max_radius = int(np.ceil( np.max(radius)/self.param['beam_width'] ))
        print(max_radius)
        return range(-max_radius, max_radius+1, 1)



