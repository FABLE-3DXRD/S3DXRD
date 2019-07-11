'''
Welcome to the heart of ModelScanning3DXRD!
Here the forward model is executed and reflections are
computed for. The algorithm is found in the
run() method. The only part of the physics that is not
described in here or called from in here will probably reside
only in reflections.py where structure factors and
miller planes are computed for.
'''

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from xfab import tools
from xfab import detector
from xfab.structure import int_intensity
from . import variables,check_input,file_io,cms_peak_compute,convexPolygon
import sys
from shapely.geometry import Polygon
import pylab as pl

A_id = variables.refarray().A_id

class find_refl:
    def __init__(self,param,hkl,killfile=None):
        '''
        A find_refl object is used to hold information
        that otherwise would have to be recomputed for
        during the algorithm execution in run().
        '''
        self.killfile = killfile
        self.param = param
        self.hkl = hkl
        self.voxel = [None]*self.param['no_voxels']

        # Simple transforms of input and set constants
        self.K = -2*np.pi/self.param['wavelength']
        self.S = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

        # Detector tilt correction matrix
        self.R = tools.detect_tilt(self.param['tilt_x'],
                                   self.param['tilt_y'],
                                   self.param['tilt_z'])

        # wedge NB! wedge is in degrees
        # The sign is reversed for wedge as the parameter in
        # tools.find_omega_general is right handed and in ImageD11
        # it is left-handed (at this point wedge is defined as in ImageD11)
        self.wy = -1.*self.param['wedge']*np.pi/180.
        self.wx = 0.

        w_mat_x = np.array([[1, 0        , 0         ],
                           [0, np.cos(self.wx), -np.sin(self.wx)],
                           [0, np.sin(self.wx),  np.cos(self.wx)]])
        w_mat_y = np.array([[ np.cos(self.wy), 0, np.sin(self.wy)],
                           [0         , 1, 0        ],
                           [-np.sin(self.wy), 0, np.cos(self.wy)]])
        self.r_mat = np.dot(w_mat_x,w_mat_y)

        # Spatial distortion
        if self.param['spatial'] != None:
            from ImageD11 import blobcorrector
            self.spatial = blobcorrector.correctorclass(self.param['spatial'])

        # %No of images
        self.nframes = (self.param['omega_end']-self.param['omega_start'])/self.param['omega_step']


        self.peak_merger = cms_peak_compute.PeakMerger( self.param )


    def run(self, start_voxel, end_voxel, log=True, parallel=None):
        '''
        This is the main algorithmic loop that will calculate the diffraction pattern
        based on a per-voxel definition of the sample. The algorithm deals with one voxel at the
        time, computing all reflections that would be generated for that voxel in a single go.
        Therefore, the run time is linear with voxel number regardless of the number of y-scans.

        Input: start_voxel: Used for running in parrallel, equal to 0 for 1 cpu
               end_voxel:   Used for running in parrallel, equal to number of voxels for 1 cpu
               log:         used to determine of this process should print to stdout or be quiet.
               parallel:    Flag to indicate parallel mode. Such a mode requires that we pass the result
                            of the run to the calling process.
        '''
        self.param['lorentz_apply']=1


        tth_lower_bound = self.param['two_theta_lower_bound']*np.pi/180.
        tth_upper_bound = self.param['two_theta_upper_bound']*np.pi/180.


        spot_id = 0

        # Compute used quanteties once for speed
        #--------------------------------------------
        # Don't make static dictonary calls in loop, slower..
        no_voxels = self.param['no_voxels']
        beam_width = self.param['beam_width']
        omega_start = self.param['omega_start']*np.pi/180
        omega_end = self.param['omega_end']*np.pi/180
        wavelength = self.param['wavelength']
        scale_Gw = wavelength/(4.*np.pi)
        detz_center = self.param['detz_center']
        dety_center = self.param['dety_center']
        z_size = self.param['z_size']
        y_size = self.param['y_size']
        distance = self.param['distance']
        wavelength = self.param['wavelength']


        if log==True:
            print('no of voxels ',no_voxels)
        # print(start_voxel, end_voxel)
        # with open('solution','w') as f:
        #     for voxel_nbr in range(start_voxel, end_voxel):
        #         if len(self.param['phase_list']) == 1:
        #             phase = self.param['phase_list'][0]
        #         else:
        #             phase = self.param['phase_voxels_%s' %(self.param['voxel_list'][voxel_nbr])]
        #         unit_cell = self.param['unit_cell_phase_%i' %phase]
        #         U = self.param['U_voxels_%s' %(self.param['voxel_list'][voxel_nbr])]
        #         voxel_eps = np.array(self.param['eps_voxels_%s' %(self.param['voxel_list'][voxel_nbr])])
        #         B = tools.epsilon_to_b(voxel_eps,unit_cell)
        #         UB = np.dot(U,B)
        #         ub = UB.ravel()
        #         f.write("np.array([")
        #         for val in ub:
        #             f.write(str(val)+", ")
        #         f.write("])\n")
        # raise
        for voxel_nbr in range(start_voxel, end_voxel):
            A = []
            U = self.param['U_voxels_%s' %(self.param['voxel_list'][voxel_nbr])]
            if len(self.param['phase_list']) == 1:
                phase = self.param['phase_list'][0]
            else:
                phase = self.param['phase_voxels_%s' %(self.param['voxel_list'][voxel_nbr])]
            unit_cell = self.param['unit_cell_phase_%i' %phase]
            self.voxel[voxel_nbr] = variables.voxel_cont(U)
            voxel_pos = np.array(self.param['pos_voxels_%s' %(self.param['voxel_list'][voxel_nbr])])
            voxel_eps = np.array(self.param['eps_voxels_%s' %(self.param['voxel_list'][voxel_nbr])])
            # Calculate the B-matrix based on the strain tensor for each voxel
            B = tools.epsilon_to_b(voxel_eps,unit_cell)
            # add B matrix to voxel container
            self.voxel[voxel_nbr].B = B
            V = tools.cell_volume(unit_cell)
            voxel_vol = np.pi/6 * self.param['size_voxels_%s' %self.param['voxel_list'][voxel_nbr]]**3
            nrefl = 0

            for hkl in self.hkl[self.param['phase_list'].index(phase)]:

                check_input.interrupt(self.killfile)
                Gc = np.dot(B,hkl[0:3])
                Gw = np.dot(self.S,np.dot(U,Gc))
                # if hkl[0]==0 and hkl[1]==-2 and hkl[2]==0:
                #     print(np.dot(U,B))
                #     print(hkl[0:3])
                #     print(Gw)
                #     raise
                tth = tools.tth2(Gw,wavelength)

                # If specified in input file do not compute for two
                # thetaoutside range [tth_lower_bound, tth_upper_bound]
                # if tth_upper_bound<tth or tth<tth_lower_bound :
                #     continue

                costth = np.cos(tth)
                (Omega, Eta, Omega_mat, Gts) = self.find_omega_general(Gw*scale_Gw,scale_Gw,tth)

                for omega,eta,Om,Gt in zip(Omega, Eta, Omega_mat, Gts):

                    if  omega_start < omega and omega < omega_end:
                        #print(np.degrees(omega))
                        # Calc crystal position at present omega
                        [tx,ty,tz]= np.dot(Om,voxel_pos)

                        #The 3 beam y positions that could illuminate the voxel
                        beam_centre_1 = np.round(ty/beam_width)*beam_width
                        beam_centre_0 = beam_centre_1+beam_width
                        beam_centre_2 = beam_centre_1-beam_width

                        #Compute precentual voxel beam overlap for the three regions
                        overlaps_info = convexPolygon.compute_voxel_beam_overlap(tx,ty,omega,beam_centre_1, beam_width)

                        regions = [ beam_centre_0, beam_centre_1,beam_centre_2]
                        for beam_centre_y,info in zip(regions,overlaps_info):

                            # perhaps we should have a less restrictive
                            # comparision here, like: if intensity_modifier<0.0001,
                            # there is some possibilty to simulate nosie here

                            intensity_modifier = info[0]

                            if intensity_modifier==0:
                                continue


                            #Lorentz factor
                            if self.param['lorentz_apply'] == 1:
                                if eta != 0:
                                    L=1/(np.sin(tth)*abs(np.sin(eta)))
                                else:
                                    L=np.inf;
                            else:
                                L = 1.0

                            intensity = L*intensity_modifier*hkl[3]

                            # If no unit cells where activated we continue

                            dty = beam_centre_y*1000. # beam cetre in microns
                            cx = info[1]
                            cy = info[2]


                            # Compute peak based on cms of illuminated voxel fraction
                            tx = cx
                            ty = cy - beam_centre_y # The sample moves not the beam!
                            # print("tx",tx)
                            # print("ty",ty)
                            # print("")
                            tz = 0
                            (dety, detz) = self.det_coor(Gt,
                                                        costth,
                                                        wavelength,
                                                        distance,
                                                        y_size,
                                                        z_size,
                                                        dety_center,
                                                        detz_center,
                                                        self.R,
                                                        tx,ty,tz)

                            #If shoebox extends outside detector exclude it
                            if (-0.5 > dety) or\
                            (dety > self.param['dety_size']-0.5) or\
                            (-0.5 > detz) or\
                            (detz > self.param['detz_size']-0.5):
                                continue


                            if self.param['spatial'] != None :
                                # To match the coordinate system of the spline file
                                (x,y) = detector.detyz_to_xy([dety,detz],
                                                            self.param['o11'],
                                                            self.param['o12'],
                                                            self.param['o21'],
                                                            self.param['o22'],
                                                            self.param['dety_size'],
                                                            self.param['detz_size'])
                                # Do the spatial distortion
                                (xd,yd) = self.spatial.distort(x,y)
                                # transform coordinates back to dety,detz
                                (detyd,detzd) = detector.xy_to_detyz([xd,yd],
                                                        self.param['o11'],
                                                        self.param['o12'],
                                                        self.param['o21'],
                                                        self.param['o22'],
                                                        self.param['dety_size'],
                                                        self.param['detz_size'])
                            else:
                                detyd = dety
                                detzd = detz



                            if self.param['beampol_apply'] == 1:
                                #Polarization factor (Kahn, J. Appl. Cryst. (1982) 15, 330-337.)
                                rho = np.pi/2.0 + eta + self.param['beampol_direct']*np.pi/180.0
                                P = 0.5 * (1 + costth*costth -\
                                        self.param['beampol_factor']*np.cos(2*rho)*np.sin(tth)**2)
                            else:
                                P = 1.0

                            overlaps = 0 # set the number overlaps to zero

                            # if self.param['intensity_const'] != 1:
                            #     intensity = intensity_modifier*int_intensity(hkl[3],
                            #                                                 L,
                            #                                                 P,
                            #                                                 self.param['beamflux'],
                            #                                                 self.param['wavelength'],
                            #                                                 V,
                            #                                                 voxel_vol)

                            # else:
                            #     intensity = intensity_modifier*hkl[3]



                            #TODO: try and build up the images as we go
                            # Mapp reflection to the correct image

                            self.peak_merger.add_reflection_to_images([self.param['voxel_list'][voxel_nbr],
                                                        nrefl,spot_id,
                                                        hkl[0],hkl[1],hkl[2],
                                                        tth,omega,eta,
                                                        dety,detz,
                                                        detyd,detzd,
                                                        Gw[0],Gw[1],Gw[2],
                                                        L,P,hkl[3],intensity,overlaps,dty,1])

                            A.append([self.param['voxel_list'][voxel_nbr],
                                    nrefl,spot_id,
                                    hkl[0],hkl[1],hkl[2],
                                    tth,omega,eta,
                                    dety,detz,
                                    detyd,detzd,
                                    Gw[0],Gw[1],Gw[2],
                                    L,P,hkl[3],intensity,overlaps,dty,1])
                            nrefl = nrefl+1

                            #TODO: When run in parallel the spot id might be duplicated..
                            #but who cares about spot id anyways huh..
                            spot_id = spot_id+1


            # A = np.array(A)

            # if len(A) > 0:
            #     # sort rows according to omega
            #     A = A[np.argsort(A,0)[:,A_id['omega']],:]

            #     # Renumber the reflections
            #     A[:,A_id['ref_id']] = np.arange(nrefl)

            #     # Renumber the spot_id
            #     A[:,A_id['spot_id']] = np.arange(np.min(A[:,A_id['spot_id']]),
            #                                 np.max(A[:,A_id['spot_id']])+1)
            # else:
            #     A = np.zeros((0,len(A_id)))

            # save reflection info in voxel container, if we are not in parallel mode this
            # will be the same object that we ran the run() method upon. Otherwise it is by
            # defualt a copy of that object and we will need to pass the result to the calling
            # process at the end of the algorithm

            self.voxel[voxel_nbr].refs = A

            if parallel and log==True:
                progress = int(100*(voxel_nbr+1-start_voxel)/float(end_voxel-start_voxel))
                print('\rDone approximately %3i percent' %progress, end=' ')
                sys.stdout.flush()
            elif log==True:
                print('\rDone approximately %3i voxel(s) of %3i' %(voxel_nbr+1,self.param['no_voxels']), end=' ')
                sys.stdout.flush()

        if log==True:
            print('\n')


        # if we are running in parrallel the memory cannot be shared (easily)
        # so wee return our result in order to merge all results
        # into a single find_refl.py Object.

        if parallel:
            result = {"start_voxel": start_voxel,
                      "end_voxel": end_voxel,
                      "refl_list": self.voxel}
            return result
        else:
            # if not in parallel we may merge here
            # this allows us to include analyse_images_as_clusters()
            # easy in profiling
            self.peak_merger.analyse_images_as_clusters()




    def find_omega_general(self, g_w, scale_Gw, twoth):
        """
        For gw find the omega rotation (in radians) around an axis
        tilted by w_x radians around x (chi) and w_y radians around y (wedge)
        Furthermore find eta (in radians).

        Output: Omega, Eta, Omega rotation matrix, Gt (G vector in lab frame)

        A similar method can be found in the xfab module, here it has been modified for
        computational efficency. This also means that it could be hard to read, see xfab
        for more easily read code.
        """

        a = g_w[0]*self.r_mat[0][0] + g_w[1]*self.r_mat[0][1]
        b = g_w[0]*self.r_mat[1][0] - g_w[1]*self.r_mat[0][0]
        c = - np.dot(g_w, g_w) - g_w[2]*self.r_mat[0][2]
        d = a*a + b*b - c*c

        omega = []
        eta = []
        omega_mat = []
        Gt = []

        if d < 0:
            pass
        else:
            # Compute numbers that are reused twice for speed
            #-------------------------------------------------
            sq_d = np.sqrt(d)
            sinfact = 2/np.sin(twoth)
            ac = a*c
            bc = b*c
            bsqd = b*sq_d
            asqd = a*sq_d

            # Find Omega number 0
            #-------------------------------------------------
            omega.append(np.arctan2((bc - asqd), (ac + bsqd)))
            if omega[0] > np.pi:
                omega[0] = omega[0] - 2*np.pi

            cosOm = np.cos(omega[0])
            sinOm = np.sin(omega[0])
            omega_mat.append(np.dot(self.r_mat,np.array([[cosOm, -sinOm,  0],
                                                       [sinOm,  cosOm,  0],
                                                       [  0,  0  ,  1]])))
            g_t = np.dot(omega_mat[0], g_w)
            sineta = -g_t[1]*sinfact
            coseta =  g_t[2]*sinfact
            eta.append(np.arctan2(sineta, coseta))
            Gt.append(g_t/scale_Gw)

            #Find omega number 1
            #-------------------------------------------------
            omega.append(np.arctan2((bc + asqd), (ac - bsqd)))
            if omega[1] > np.pi:
                omega[1] = omega[1] - 2*np.pi
            cosOm = np.cos(omega[1])
            sinOm = np.sin(omega[1])
            omega_mat.append(np.dot(self.r_mat,np.array([[cosOm, -sinOm,  0],
                                                       [sinOm,  cosOm,  0],
                                                       [  0,  0  ,  1]])))
            g_t = np.dot(omega_mat[1], g_w)
            sineta = -g_t[1]*sinfact
            coseta =  g_t[2]*sinfact
            eta.append(np.arctan2(sineta, coseta))
            Gt.append(g_t/scale_Gw)
            #-------------------------------------------------

        return np.array(omega), np.array(eta), omega_mat, Gt


    def det_coor(self, Gt, costth, wavelength, distance, y_size, z_size,
             dety_center, detz_center, R_tilt, tx,ty,tz):
        """
        Calculates detector coordinates dety,detz

        INPUT:
        Gt is the g-vector
        y_size and z_size are the detector pixel size in y, and z (in microns)
        (dety_center, detz-center) is the beam center of the detector (in pixels)
        R_tilt is the rotation matrix of the detector
        (tx, ty, tz) is the position of the voxel at the present omega

        OUTPUT:
        [dety, detz]
        """

        # Unit directional vector for reflection
        v = np.array([costth, wavelength/(2*np.pi)*Gt[1], wavelength/(2*np.pi)*Gt[2]])
        t = (R_tilt[0, 0]*distance - np.dot(R_tilt[:, 0],np.array([tx, ty, tz])))/np.dot(R_tilt[:, 0],v)
        Ltv = np.array([tx-distance, ty, tz])+ t*v
        dety = np.dot(R_tilt[:, 1],Ltv)/y_size + dety_center
        detz = np.dot(R_tilt[:, 2],Ltv)/z_size + detz_center
        return [dety, detz]

    def save(self,voxel_nbr=None):
        """
        write ModelScanning3DXRD ref file
        """
        file_io.write_ref(self.param,self.voxel,voxel_nbr)


    def write_delta_gve(self):
        """
        Write gvector .gve file
        """
        file_io.write_delta_gve(self.param,self.voxel,self.hkl)


    def write_ini(self):
        """
        write input file for voxelSpotter
        """
        file_io.write_ini(self.param,self.hkl)


    def write_delta_flt(self):
        """
         Write filtered peaks .flt file
        """
        file_io.write_delta_flt(self.param,self.voxel)

    def write_merged_flt(self):
        """
         Write filtered peaks .flt file fro merged peaks
        """
        file_io.write_merged_flt(self.param,self.peak_merger.merged_peaks)

