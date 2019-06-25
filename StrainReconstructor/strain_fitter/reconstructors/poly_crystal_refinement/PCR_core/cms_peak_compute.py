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
import variables
import matplotlib.pyplot as plt
import cluster
#import cluster_new
import copy

voxel_reflections_id = variables.refarray().voxel_reflections_id

class PeakMerger(object):

    def __init__(self, param, voxel, voxel_position, dety_tol, detz_tol, om_tol, ymin):
        """
        PeakMerger is used to build upp all the 3D framestacks which will merge
        the delta peaks into cms peak positions such that the resolution is
        limited to the omega stepsize and the detector pixel size.

        The idea is offcourse that the merging of peaks is a more realistic model.
        """

        #Used for merging delta peaks into cms positions
        self.dety_tol = dety_tol
        self.detz_tol = detz_tol
        self.om_tol = om_tol


        # Experimental setup
        self.voxel = voxel #grant acces to the list of voxels and their reflections
        self.voxel_position = voxel_position
        self.voxel_size = param['beam_width']
        self.ymin = ymin

        self.param = param

        self.omega_step = param['omega_step']
        self.omega_start = param['omega_start']
        self.omega_end = param['omega_end']
        self.omegas = np.arange(self.omega_start, self.omega_end, np.round(self.omega_step))

        self.dtys_index = self.get_dtys_index() #in voxel units
        # print('self.dtys_index',self.dtys_index)
        # raise
        # print("self.dtys_index ",self.dtys_index)
        # print("-self.ymin + self.voxel_size*1000. = > ",self.get_index_of_dty(-2*self.ymin + self.voxel_size*1000.))
        # print("self.voxel_size*1000.  ",self.voxel_size*1000.)
        # print("-*self.ymin",-self.ymin)
        # print("15.25 = > ",self.get_index_of_dty(15.25))
        # print("self.ymin: ",self.ymin)
        # raise

        self.peaks_by_dty={}
        for dty in self.dtys_index:
            self.peaks_by_dty[dty]=[]

        self.merged_peaks = []

        # test fast merge
        # ---------------------
        self.clustered_delta_peaks = []
        self.voxel_peak_map = {}
        self.framestack_max = np.pi
        self.no_frame_binns = 36
        # ---------------------

    def get_index_of_dty(self, dty_in_microns):
        '''
        assumes self.ymin in microns and self.beam_width in mm
        '''
        return np.round( ( dty_in_microns-self.ymin )/( 1000.*self.voxel_size ) ).astype(int)

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

        dty = reflection[voxel_reflections_id['dty_as_index']]
        #refl = copy.copy(reflection)
        # binary_refl[voxel_reflections_id['detz']] = np.round(reflection[voxel_reflections_id['detz']])
        # binary_refl[voxel_reflections_id['dety']] = np.round(reflection[voxel_reflections_id['dety']])
        # binary_refl[voxel_reflections_id['omega']] = np.radians( np.floor(reflection[voxel_reflections_id['omega']]*180./np.pi) + 0.5*self.omega_step )
        #self.peaks_by_dty[dty].append(refl)

        # if dty<0:
        #     print("reflection",reflection)
        #     print("voxel_reflections_id['dty_as_index']",voxel_reflections_id['dty_as_index'])
        #     print("dty as index: ",dty)
        #     raise

        self.peaks_by_dty[dty].append(reflection)





    def analyse_images_as_clusters(self, iteration, save_dir=None, reflections=None):
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


        self.peaks_by_dty={}
        for dty in self.dtys_index:
            self.peaks_by_dty[dty]=[]
        for v in self.voxel:
            for reflection in v.refs:
                self.add_reflection_to_images(reflection)



        index_dty = voxel_reflections_id['dty']
        index_z = voxel_reflections_id['detz']
        index_y = voxel_reflections_id['dety']
        index_om = voxel_reflections_id['omega']
        index_int = voxel_reflections_id['Int']
        index_h = voxel_reflections_id['h']
        index_k = voxel_reflections_id['k']
        index_l = voxel_reflections_id['l']

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

        tol_z = self.detz_tol #units of pixels
        tol_y = self.dety_tol #units of pixels
        tol_om = np.radians( self.om_tol ) #units of radians

        self.merged_peaks = []
        self.clustered_delta_peaks = []
        self.voxel_peak_map = {}
        for v in range(len(self.voxel)):
            self.voxel_peak_map[v] = []

        for key in self.peaks_by_dty:
            peaks = self.peaks_by_dty[key]
            self.clustered_delta_peaks.append([])

            # if len(peaks)>0:
            #     if peaks[0][index_dty]==-0.5:
            #         print(peaks)
            #         raise

            if peaks==[]:
                self.merged_peaks.append([])
                continue

            #TODO: figure out what is wrong with this method:
            #peak_clusters = cluster.cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)
            peak_clusters = cluster.cluster(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)
            #Not good cosider 180 symetry hkl in different omega, then wee need to cluster in omega aswell...
            #peak_clusters = cluster.cluster_in_z_y_omega(peaks,index_h,index_k,index_l,0.1,0.1,0.1)

            # for a_cluster in peak_clusters:
            #     for peak in a_cluster:
            #         if peak[index_dty]==-0.5:
            #             if peak[index_h]==-4 and peak[index_k]==-1 and peak[index_l]==-1:
            #                 print("cluster",a_cluster)

            # Ensure that peak_clusters and merged_peaks_dty have same indexation
            merged_peaks_dty, peak_clusters = cluster.merge_many_clusters(peak_clusters,index_z,index_y,index_om,index_int)

            if save_dir is not None:
                self.plot_peak(peak_clusters,merged_peaks_dty,key, index_z, index_y, index_om, index_int, iteration, save_dir=save_dir)

            self.merged_peaks.append( merged_peaks_dty )

            for a_cluster in peak_clusters:
                self.clustered_delta_peaks[-1].append( a_cluster )

        for i,dty in enumerate(self.clustered_delta_peaks):
            for j,a_cluster in enumerate(dty):
                for peak in a_cluster:
                    voxel_nbr = peak[voxel_reflections_id['voxel_id']]
                    self.voxel_peak_map[ voxel_nbr ].append([ i, j])


        self.box_map = self.get_box_map(self.clustered_delta_peaks, index_om, index_y, index_z, self.framestack_max, self.no_frame_binns  )

        return self.merged_peaks


#OLD CODE:
    # def analyse_images_as_clusters(self, iteration, save_dir=None, reflections=None):
    #     '''
    #     Merge all delta peaks located in the dictionary self.peaks_by_dty.

    #     * self.peaks_by_dty uses discrete coordinates, i.e peaks are located at
    #       pixel centers and multiples of omegastep/2. only.

    #     * Clusters delta peaks by distance such that peaks in neighboring pixels
    #       are grouped together.

    #     * Comptes CMS based on coordinates and intenisty of all dleta peaks in the
    #       cluster.

    #     This method will build up the self.merged_peaks list attribute such that
    #     the peakmerger has all merged peaks in that list after execution.

    #     The clustering is based on sorting the peak set along each dimension
    #     such that the effort needed is proportional to sorting the delta peak set three times.
    #     '''


    #     self.peaks_by_dty={}
    #     for dty in self.dtys_index:
    #         self.peaks_by_dty[dty]=[]
    #     for v in self.voxel:
    #         for reflection in v.refs:
    #             self.add_reflection_to_images(reflection)



    #     index_dty = voxel_reflections_id['dty']
    #     index_z = voxel_reflections_id['detz']
    #     index_y = voxel_reflections_id['dety']
    #     index_om = voxel_reflections_id['omega']
    #     index_int = voxel_reflections_id['Int']
    #     index_h = voxel_reflections_id['h']
    #     index_k = voxel_reflections_id['k']
    #     index_l = voxel_reflections_id['l']

    #     # Display diffraction image for debug
    #     # image = np.zeros((self.param['detz_size'],self.param['dety_size']))
    #     # for key in self.peaks_by_dty:
    #     #     for p in self.peaks_by_dty[key]:
    #     #         #if p[A_id['omega']==1]:
    #     #         i=self.param['detz_size'] - 1 - int(p[A_id['detz']])
    #     #         j=self.param['dety_size'] - 1 - int(p[A_id['dety']])
    #     #         if p[A_id['omega']]-1*180./np.pi<0.001:
    #     #             image[i,j]=1
    #     # image = detector.trans_orientation(image,self.param['o11'],self.param['o12'],self.param['o21'],self.param['o22'],'inverse')
    #     # plt.imshow(image,cmap='gray')
    #     # plt.show()

    #     tol_z = self.detz_tol #units of pixels
    #     tol_y = self.dety_tol #units of pixels
    #     tol_om = np.radians( self.om_tol ) #units of radians

    #     self.merged_peaks = []

    #     for key in self.peaks_by_dty:
    #         peaks = self.peaks_by_dty[key]


    #         if peaks==[]:
    #             self.merged_peaks.append([])
    #             continue

    #         peak_clusters = cluster.cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)

    #         #BAD so far: se note at function
    #         #peak_clusters = cluster.cluster_by_hkl(peaks, index_h, index_k, index_l)

    #         merged_peaks_dty = cluster.merge_many_clusters(peak_clusters,index_z,index_y,index_om,index_int)
    #         if save_dir is not None:
    #             self.plot_peak(peak_clusters,merged_peaks_dty,key, index_z, index_y, index_om, index_int, iteration, save_dir=save_dir)

    #         self.merged_peaks.append(merged_peaks_dty)

    #     return self.merged_peaks
#OLD END

    def plot_cluster_2D(self,cluster,cms, dty, index_z, index_y, index_om, index_int, iteration, save_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cm = plt.cm.get_cmap('tab10')

        max_int = 0
        max_omega =0
        min_omega =180.
        for p in cluster:
            omega = np.degrees(p[index_om])
            if p[index_int]>max_int: max_int = p[index_int]
            if omega>max_omega: max_omega=omega
            if omega<min_omega: min_omega=omega
        omegas=[]
        ys=[]
        zs=[]
        ss=[]
        for p in cluster:
            omegas.append(np.degrees(p[index_om]))
            ys.append(p[index_y])
            zs.append(p[index_z])
            ss.append(300*p[index_int]/max_int)

        sc = ax.scatter(ys,zs,c=omegas, vmin=min_omega, vmax=max_omega, marker='^',s=ss, cmap=cm)
        ax.set_title('Delta peak cluster, with '+str(len(cluster))+' peaks at dty: '+str(cluster[0][voxel_reflections_id['dty']])+r' $\mu$m, omega frame: '+str(np.floor(np.degrees(cluster[0][voxel_reflections_id['omega']])).astype(int))+', iteration '+ str(iteration))
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.scatter(cms[index_y],cms[index_z],marker='o', c="purple")
        ax.text(cms[index_y],cms[index_z],'CMS',size=20, zorder=1, color='k')
        ax.grid(color='k', linestyle='-', linewidth=2)
        cbar = plt.colorbar(sc)
        cbar.ax.set_title(r'$\omega$ [degrees]')
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        lx = np.round(xticks[0]).astype(int)
        hx = np.round(xticks[-1]).astype(int)
        xticks = np.arange(lx-2,hx+2,1)
        ax.set_xticks(xticks)
        ly = np.round(yticks[0]).astype(int)
        hy = np.round(yticks[-1]).astype(int)
        yticks = np.arange(ly-2,hy+2,1)
        ax.set_yticks(yticks)
        plt.axis('equal')

        #plt.show()

        if save_dir is not None:
            filename = 'delta_peak_cluster_iter_'+str(iteration)
            with open(save_dir+'/'+filename,'w') as f:
                # files should be of format such that
                # for l in open(fn).readlines(): exec(l)
                # can run them as python code for ease
                f.write( '#Delta peak cluster at dty index '+str(dty)+'\n' )
                f.write( '#Format of peak entries follows:'+'\n' )
                f.write( '#dety   detz   Int   omega   dty   h   k   l   voxel_id   dty_as_index'+'\n' )
                for key in voxel_reflections_id: f.write( key+" = "+str(voxel_reflections_id[key])+'\n' )
                f.write('iteration = '+str(iteration)+'\n')
                f.write(''+str())
                f.write('max_int = '+str(max_int)+'\n')
                f.write('max_omega = '+str(max_omega)+'\n')
                f.write('min_omega = '+str(min_omega)+'\n')
                f.write('\n')
                f.write('cms = '+ str(cms)+'\n')
                out = ''
                out = 'cluster = ['
                for p in cluster: out += str(p)+','
                out += ']'
                f.write(out)
            filename_plot = 'cluster_peak_plot_iter_'+str(iteration)
            fig.savefig(save_dir+"/"+filename_plot)
        # print(cms[index_y],cms[index_z])
        # plt.show()
        plt.close('all')



    def plot_peak(self, peak_clusters, merged_peaks_dty, dty, index_z, index_y, index_om, index_int, iteration, save_dir=None):
        '''
        Take all peak clusters and plot a selected peak as a scatter plot
        '''

        for i,(cluster,cms) in enumerate(zip(peak_clusters,merged_peaks_dty)):
            if len(cluster)>20:
                #301 1055
                # self.plot_cluster_2D(cluster,cms,dty, index_z, index_y, index_om,index_int, iteration, save_dir=save_dir)
                if abs(cluster[0][voxel_reflections_id['dty']]+10)<0.01 and np.floor(np.degrees(cluster[0][index_om])).astype(int)==176:
                    if abs(cluster[0][index_y]-301)<40 and abs(cluster[0][index_z]-1055)<40:
                        self.plot_cluster_2D(cluster,cms,dty, index_z, index_y, index_om,index_int, iteration, save_dir=save_dir)
            # for i,(cluster,cms) in enumerate(zip(peak_clusters,merged_peaks_dty)):
            #     if len(cluster)>3:
            #         peak_min_y=2050
            #         peak_max_y=0
            #         peak_min_z=2050
            #         peak_max_z=0
            #         for peak in cluster:
            #             if peak[index_y]<peak_min_y: peak_min_y=peak[index_y]
            #             if peak[index_y]>peak_max_y: peak_max_y=peak[index_y]
            #             if peak[index_z]<peak_min_z: peak_min_z=peak[index_z]
            #             if peak[index_z]>peak_max_z: peak_max_z=peak[index_z]

            #         if abs(peak_min_y-peak_max_y)>2. or abs(peak_min_z-peak_max_z)>2:
            #             self.plot_cluster_2D(cluster,cms,dty, index_z, index_y, index_om,index_int, iteration, save_dir=save_dir)



    def get_dtys_index(self):
        #Assume ymax = -ymin
        return range(0, self.get_index_of_dty(-self.ymin + self.voxel_size*1000.), 1)
        # """
        # Determine the list of all beam y-settings that could possible
        # give rise to nonzero intensity. return settings in voxel units.

        # e.g. +3 means +3 voxel widths positive in y (sample frame).
        # """
        # radius = np.zeros((self.param['no_voxels'],))
        # Delta = (1/2.)*self.param['beam_width']
        # for voxel_nbr in range( self.param['no_voxels'] ):
        #     voxel_pos = self.voxel_position[voxel_nbr]
        #     if voxel_pos[0]>0:
        #         x = voxel_pos[0] + Delta
        #     else:
        #         x = voxel_pos[0] - Delta
        #     if voxel_pos[1]>1:
        #         y = voxel_pos[1] + Delta
        #     else:
        #         y = voxel_pos[1] - Delta
        #     radius[voxel_nbr] = np.sqrt( x*x + y*y )

        # max_radius = int(np.ceil( np.max(radius)/self.param['beam_width'] ))
        # #print(max_radius)
        # return range(-max_radius, max_radius+1, 1)


    def analyse_images_as_clusters_gradient_mode(self, reflections, voxel_number):
        '''
        Peak merging for gradient mode, a lot of cheating is possible :=)

        We try to make less computations as only a handfull of peaks will
        be affected by the pertubation during jacobian evaluation.

        remember that only one variable of one voxel is shifted at the time
        so the number of peaks affected is surely less than 1% of the entire data.

        this is neccesary if we want the algorithm to scale with grainsize..
        '''


        index_dty = voxel_reflections_id['dty']
        index_z = voxel_reflections_id['detz']
        index_y = voxel_reflections_id['dety']
        index_om = voxel_reflections_id['omega']

        # copy the merged set
        merged_peaks_copy = self.make_copy_merged_peaks()

        for delta_peak in reflections:
            dty_index, cluster_index = self.find_correct_box(delta_peak, self.box_map, self.framestack_max, self.no_frame_binns, index_om, index_y, index_z )
            if dty_index==None or cluster_index==None:
                continue
            merged_peak = self.merged_peaks[dty_index][cluster_index]
            old_delta_peaks = self.clustered_delta_peaks[dty_index][cluster_index]
            merged_peaks_copy[dty_index][cluster_index] = self.recompute_merged_peak(merged_peak, old_delta_peaks, delta_peak )

        return merged_peaks_copy


    def make_copy_merged_peaks(self):
        '''
        Relatively fast way of producing a copy of a framstack corridor
        ( 20-40 times faster than copy.deepcopy() )
        '''
        merged_peaks_copy = []
        for i,dty in enumerate(self.merged_peaks):
            merged_peaks_copy.append([])
            for j,peak in enumerate(dty):
                merged_peaks_copy[i].append( peak[:] )
        return merged_peaks_copy


    def recompute_merged_peak(self, merged_peak, old_delta_peaks, new_delta_peak ):
        '''

        old_delta_peaks are complete and merged_peak was constructed from them.
        new_delta_peak is the peak which shall be replaced.

        this method is only good if each voxel has but one contribution to each peak.
        '''

        sum_int = merged_peak[ voxel_reflections_id['Int'] ]
        new_merged_dety = merged_peak[ voxel_reflections_id['dety'] ]*sum_int
        new_merged_detz = merged_peak[ voxel_reflections_id['detz'] ]*sum_int
        new_merged_om = merged_peak[ voxel_reflections_id['omega'] ]*sum_int
        new_sum_int = sum_int

        new_merged_peak = copy.copy( merged_peak )

        for old_peak in old_delta_peaks:
            #print("old_peak ",old_peak)
            #print("new_delta ", new_delta_peak)
            if old_peak[ voxel_reflections_id['voxel_id'] ]==new_delta_peak[ voxel_reflections_id['voxel_id'] ]:

                old_int = old_peak[ voxel_reflections_id['Int'] ]
                old_dety = old_peak[ voxel_reflections_id['dety'] ]
                old_detz = old_peak[ voxel_reflections_id['detz'] ]
                old_om = old_peak[ voxel_reflections_id['omega'] ]

                new_int = new_delta_peak[ voxel_reflections_id['Int'] ]
                new_dety = new_delta_peak[ voxel_reflections_id['dety'] ]
                new_detz = new_delta_peak[ voxel_reflections_id['detz'] ]
                new_om = new_delta_peak[ voxel_reflections_id['omega'] ]

                new_sum_int = new_sum_int - old_int + new_int
                new_merged_dety = (new_merged_dety - old_dety*old_int + new_dety*new_int)/new_sum_int
                new_merged_detz = (new_merged_detz - old_detz*old_int + new_detz*new_int)/new_sum_int
                new_merged_om = (new_merged_om - old_om*old_int + new_om*new_int)/new_sum_int
                break
        else:
            # In the rare case that a new delta peka emerges as a result of the
            # Jacobian pertubation, we may simply ignore it as the intensity
            # will be extreemely small anyways, not contributing to the cms of the merged
            # peak at all. The same goes for peaks dissapearing. Entering or dissapearing from
            # a cluster means in practice to be very slightly graced by the beam so this
            # approximation sufficses, altought the "correct way" of doing things would be
            # to recompute the entire set, but the we lose all our scaling ability..
            return new_merged_peak


        new_merged_peak[ voxel_reflections_id['Int'] ] = new_sum_int
        new_merged_peak[ voxel_reflections_id['dety'] ] = new_merged_dety
        new_merged_peak[ voxel_reflections_id['detz'] ] = new_merged_detz
        new_merged_peak[ voxel_reflections_id['omega'] ] = new_merged_om

        for i,(n,o) in enumerate(zip(new_merged_peak,merged_peak)):
            if abs(n-o)>0.001 and i!=2 and i<4:
                print("new_merged_peak", new_merged_peak)
                print("merged_peak", merged_peak)
                print("")
                print("old_delta_peaks", old_delta_peaks)
                print("new_delta_peak", new_delta_peak)
                raise

        #print(merged_peak[voxel_reflections_id['dety']], merged_peak[voxel_reflections_id['detz']], merged_peak[voxel_reflections_id['omega']])
        #print(new_merged_peak[voxel_reflections_id['dety']], new_merged_peak[voxel_reflections_id['detz']], new_merged_peak[voxel_reflections_id['omega']] )
        return new_merged_peak




    def get_box_map(self, delta_peak_clusters, index_om, index_dety, index_detz, framestack_max, no_frame_binns  ):
        '''
        Produce a data structure containing the padded boundaries of
        each delta peak cluster. If a delta peak fall within a padded
        box during Jacobian evaluation it belongs to the corresponding
        cluster.
        '''
        box_map = []
        padding_det = 1
        padding_om = (framestack_max/no_frame_binns)/(10.)

        for dty_index, clusters_in_dty in enumerate( delta_peak_clusters ):

            box_map.append( [] )
            for i in range(0, no_frame_binns):
                box_map[-1].append([])

            for cluster_index, cluster in enumerate( clusters_in_dty ):
                box = self.get_box_from_cluster( cluster, dty_index, cluster_index, index_om, index_dety, index_detz, padding_det, padding_om  )
                index_low = self.get_frame_bin(box[0], framestack_max, no_frame_binns )
                index_high = self.get_frame_bin(box[1], framestack_max, no_frame_binns )

                if index_low!=index_high:
                    box_map[-1][index_low].append( box )
                    box_map[-1][index_high].append( box )
                else:
                    box_map[-1][index_low].append( box )

        return box_map

    def get_box_from_cluster(self, cluster, dty_index, cluster_index, index_om, index_dety, index_detz, padding_det, padding_om ):

        max_omega = cluster[0][index_om]
        min_omega = cluster[0][index_om]
        max_dety = cluster[0][index_dety]
        min_dety = cluster[0][index_dety]
        max_detz = cluster[0][index_detz]
        min_detz = cluster[0][index_detz]
        for peak in cluster:
            if peak[index_om]>max_omega: max_omega=peak[index_om]
            if peak[index_om]<min_omega: min_omega=peak[index_om]
            if peak[index_dety]>max_dety: max_dety=peak[index_dety]
            if peak[index_dety]<min_dety: min_dety=peak[index_dety]
            if peak[index_detz]>max_detz: max_detz=peak[index_detz]
            if peak[index_detz]<min_detz: min_detz=peak[index_detz]

        box = [ max_omega + padding_om, min_omega - padding_om, \
                max_dety + padding_det, min_dety - padding_det, \
                max_detz + padding_det, min_detz - padding_det, \
                dty_index, cluster_index ]

        return box

    def find_correct_box(self, delta_peak, box_map, framestack_max, no_frame_binns, index_om, index_dety, index_detz ):

        dty = delta_peak[ voxel_reflections_id['dty_as_index'] ]
        frame_bin = self.get_frame_bin(delta_peak[index_om], framestack_max, no_frame_binns )

        for box in box_map[dty][frame_bin]:
            if delta_peak[index_om]<box[0] and delta_peak[index_om]>box[1]:
                if delta_peak[index_dety]<box[2] and delta_peak[index_dety]>box[3]:
                    if delta_peak[index_detz]<box[4] and delta_peak[index_detz]>box[5]:
                        return box[6],box[7]
        return None, None

    def get_frame_bin(self, omega, framestack_max, no_frame_binns ):
        return int( round( (omega/framestack_max)*(no_frame_binns-1)) )