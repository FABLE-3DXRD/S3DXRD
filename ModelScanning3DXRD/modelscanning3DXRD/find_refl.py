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
import numpy as n
from xfab import tools
from xfab import detector
from xfab.structure import int_intensity
from . import variables,check_input,file_io
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
        self.K = -2*n.pi/self.param['wavelength']
        self.S = n.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

        # Detector tilt correction matrix
        self.R = tools.detect_tilt(self.param['tilt_x'],
                                   self.param['tilt_y'],
                                   self.param['tilt_z'])

        # wedge NB! wedge is in degrees
        # The sign is reversed for wedge as the parameter in
        # tools.find_omega_general is right handed and in ImageD11
        # it is left-handed (at this point wedge is defined as in ImageD11)
        self.wy = -1.*self.param['wedge']*n.pi/180.
        self.wx = 0.

        w_mat_x = n.array([[1, 0        , 0         ],
                           [0, n.cos(self.wx), -n.sin(self.wx)],
                           [0, n.sin(self.wx),  n.cos(self.wx)]])
        w_mat_y = n.array([[ n.cos(self.wy), 0, n.sin(self.wy)],
                           [0         , 1, 0        ],
                           [-n.sin(self.wy), 0, n.cos(self.wy)]])
        self.r_mat = n.dot(w_mat_x,w_mat_y)

        # Spatial distortion
        if self.param['spatial'] != None:
            from ImageD11 import blobcorrector
            self.spatial = blobcorrector.correctorclass(self.param['spatial'])

        # %No of images
        self.nframes = (self.param['omega_end']-self.param['omega_start'])/self.param['omega_step']


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
        spot_id = 0

        # Compute used quanteties once for speed
        #--------------------------------------------
        # Don't make static dictonary calls in loop, slower..
        no_voxels = self.param['no_voxels']
        beam_width = self.param['beam_width']
        omega_start = self.param['omega_start']*n.pi/180
        omega_end = self.param['omega_end']*n.pi/180
        wavelength = self.param['wavelength']
        scale_Gw = wavelength/(4.*n.pi)
        detz_center = self.param['detz_center']
        dety_center = self.param['dety_center']
        z_size = self.param['z_size']
        y_size = self.param['y_size']
        distance = self.param['distance']
        wavelength = self.param['wavelength']


        w = beam_width/2.
        #beam_length = 10000000 + 10*beam_width*no_voxels*no_voxels # just make sure this is large enough
        #Dx = n.array([w,0,0])
        #Dy = n.array([0,w,0])
        voxel_area = beam_width**2
        #---------------------------------------------

        if log==True:
            print('no of voxels ',no_voxels)
        # print(start_voxel, end_voxel)
        for voxel_nbr in range(start_voxel, end_voxel):
            A = []
            U = self.param['U_voxels_%s' %(self.param['voxel_list'][voxel_nbr])]
            if len(self.param['phase_list']) == 1:
                phase = self.param['phase_list'][0]
            else:
                phase = self.param['phase_voxels_%s' %(self.param['voxel_list'][voxel_nbr])]
            unit_cell = self.param['unit_cell_phase_%i' %phase]
            self.voxel[voxel_nbr] = variables.voxel_cont(U)
            voxel_pos = n.array(self.param['pos_voxels_%s' %(self.param['voxel_list'][voxel_nbr])])
            voxel_eps = n.array(self.param['eps_voxels_%s' %(self.param['voxel_list'][voxel_nbr])])
            # Calculate the B-matrix based on the strain tensor for each voxel
            B = tools.epsilon_to_b(voxel_eps,unit_cell)
            # add B matrix to voxel container
            self.voxel[voxel_nbr].B = B
            V = tools.cell_volume(unit_cell)
            voxel_vol = n.pi/6 * self.param['size_voxels_%s' %self.param['voxel_list'][voxel_nbr]]**3
            nrefl = 0
            for hkl in self.hkl[self.param['phase_list'].index(phase)]:
                check_input.interrupt(self.killfile)
                Gc = n.dot(B,hkl[0:3])
                Gw = n.dot(self.S,n.dot(U,Gc))
                tth = tools.tth2(Gw,wavelength)
                costth = n.cos(tth)
                (Omega, Eta, Omega_mat, Gts) = self.find_omega_general(Gw*scale_Gw,scale_Gw,tth)
                for omega,eta,Om,Gt in zip(Omega, Eta, Omega_mat, Gts):
                    if  omega_start < omega and omega < omega_end:

                        # Calc crystal position at present omega
                        [tx,ty,tz]= n.dot(Om,voxel_pos)

                        #The 3 beam y positions that could illuminate the voxel
                        beam_centre_1 = n.round(ty/beam_width)*beam_width
                        beam_centre_0 = beam_centre_1+beam_width
                        beam_centre_2 = beam_centre_1-beam_width

                        #Compute precentual voxel beam overlap for the three regions
                        int_mod_0,int_mod_1,int_mod_2 = self.calc_int_mod(omega,tx,ty,beam_width,voxel_area,beam_centre_0,beam_centre_1,beam_centre_2)

                        # Debug code. Something like this can be used to check if calc_int_mod() is sound
                        # If you want to improve calc_int_mod() this is usefull, so I will leave it here for now
                        # The idea is to use Shapelys polygon package as a reference calculation.
                        #-----------------------------------------------------------------------
                        #Compute voxel as polygon
                        # [tx1,ty1,tz1]= n.dot(Om,voxel_pos+Dx+Dy)
                        # [tx2,ty2,tz2]= n.dot(Om,voxel_pos+Dx-Dy)
                        # [tx3,ty3,tz3]= n.dot(Om,voxel_pos-Dx-Dy)
                        # [tx4,ty4,tz4]= n.dot(Om,voxel_pos-Dx+Dy)
                        # voxel = Polygon([(tx1,ty1), (tx2,ty2), (tx3,ty3), (tx4,ty4)])
                        #Compute edge beams as polygons
                        # beam_0 = Polygon([(beam_length,beam_centre_0+w), (-beam_length,beam_centre_0+w), (-beam_length,beam_centre_0-w), ((beam_length,beam_centre_0-w))])
                        # beam_2 = Polygon([(beam_length,beam_centre_2+w), (-beam_length,beam_centre_2+w), (-beam_length,beam_centre_2-w), ((beam_length,beam_centre_2-w))])

                        #Compute intensity modifiers based on percentual voxel-beam overlap
                        #i.e int_mod_0+int_mod_1+int_mod_2 = 1
                        # int_mod_0 = voxel.intersection(beam_0).area/voxel_area
                        # int_mod_2 = voxel.intersection(beam_2).area/voxel_area
                        # int_mod_1 = 1 - int_mod_0 - int_mod_2

                        #Temporary sanity check
                        # tol = 0.0001
                        # error = n.array([int_0,int_1,int_2])-n.array([int_mod_0,int_mod_1,int_mod_2])
                        # if n.abs(error[0])>tol or n.abs(error[1])>tol or n.abs(error[2])>tol:
                        #     print("ERROR!")
                        #     print(error)
                        #     print("omega",omega)
                        #     print("ty ",ty)
                        #     print("Shapely: ",[int_mod_0,int_mod_1,int_mod_2])
                        #     print("AXEL: ",[int_0,int_1,int_2])
                        #     raise KeyboardInterrupt

                        #beam_1 = Polygon([(beam_length,beam_centre_1+w), (-beam_length,beam_centre_1+w), (-beam_length,beam_centre_1-w), ((beam_length,beam_centre_1-w))])
                        #plots the detected overlaps
                        # for j,reg in enumerate([[beam_0,int_mod_0],[beam_1,int_mod_1],[beam_2,int_mod_2]]):
                        #     beam = reg[0]
                        #     intensity_modifier = reg[1]
                        #     vox_x,vox_y = voxel.exterior.xy
                        #     beam_x,beam_y = beam.exterior.xy
                        #     pl.figure()
                        #     pl.plot(vox_x,vox_y)
                        #     pl.plot(beam_x,beam_y)
                        #     pl.axis('equal')
                        #     pl.axis([tx1-5*w, tx1+5*w, ty1-4*w, ty1+4*w])
                        #     pl.title("for beam nbr "+str(j)+", int_mod = "+str(intensity_modifier))
                        #     pl.show()
                        # print(beam_width)
                        # print("Voxel number: "+str(voxel_nbr))
                        # print("Voxel coordinate: "+str(tx)+","+str(ty))
                        # print("Omega: "+str(omega))
                        # print(beam_centre_0,beam_centre_1,beam_centre_2)
                        #-----------------------------------------------------------------

                        regions = [ [int_mod_0,beam_centre_0], [int_mod_1,beam_centre_1], [int_mod_2,beam_centre_2] ]
                        for region in regions:

                            # perhaps we should have a less restrictive
                            # comparision here, like: if region[0]<0.01,
                            # there is some possibilty to simulate nosie here

                            dty = region[1]*1000. # beam cetre in microns
                            intensity_modifier = region[0]

                            # If no unit cells where activated we continue
                            if intensity_modifier==0:
                                continue

                            #TODO:
                            # Perhaps we should fix this so that the peak is
                            # not calculated for voxel centre but from the
                            # illuminated parts cms. On the other hand, it will
                            # hardly ever matter if we are considering the resolution
                            # of the detector.

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
                                rho = n.pi/2.0 + eta + self.param['beampol_direct']*n.pi/180.0
                                P = 0.5 * (1 + costth*costth -\
                                        self.param['beampol_factor']*n.cos(2*rho)*n.sin(tth)**2)
                            else:
                                P = 1.0
                            #Lorentz factor
                            if self.param['lorentz_apply'] == 1:
                                if eta != 0:
                                    L=1/(n.sin(tth)*abs(n.sin(eta)))
                                else:
                                    L=n.inf;
                            else:
                                L = 1.0

                            overlaps = 0 # set the number overlaps to zero

                            if self.param['intensity_const'] != 1:
                                intensity = intensity_modifier*int_intensity(hkl[3],
                                                                            L,
                                                                            P,
                                                                            self.param['beamflux'],
                                                                            self.param['wavelength'],
                                                                            V,
                                                                            voxel_vol)

                            else:
                                intensity = intensity_modifier*hkl[3]

                            A.append([self.param['voxel_list'][voxel_nbr],
                                    nrefl,spot_id,
                                    hkl[0],hkl[1],hkl[2],
                                    tth,omega,eta,
                                    dety,detz,
                                    detyd,detzd,
                                    Gw[0],Gw[1],Gw[2],
                                    L,P,hkl[3],intensity,overlaps,dty])
                            nrefl = nrefl+1

                            #TODO: When run in parallel the spot id might be duplicated..
                            spot_id = spot_id+1

            A = n.array(A)

            if len(A) > 0:
                # sort rows according to omega
                A = A[n.argsort(A,0)[:,A_id['omega']],:]

                # Renumber the reflections
                A[:,A_id['ref_id']] = n.arange(nrefl)

                # Renumber the spot_id
                A[:,A_id['spot_id']] = n.arange(n.min(A[:,A_id['spot_id']]),
                                            n.max(A[:,A_id['spot_id']])+1)
            else:
                A = n.zeros((0,len(A_id)))

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

        # if we are running in parrallel the memory cannot be shared
        # so wee need to return our result in order to merge all results
        # into a single find_refl.py Object.
        if parallel:
            result = {"start_voxel": start_voxel,
                      "end_voxel": end_voxel,
                      "refl_list": self.voxel}
            return result



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
        c = - n.dot(g_w, g_w) - g_w[2]*self.r_mat[0][2]
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
            sq_d = n.sqrt(d)
            sinfact = 2/n.sin(twoth)
            ac = a*c
            bc = b*c
            bsqd = b*sq_d
            asqd = a*sq_d

            # Find Omega number 0
            #-------------------------------------------------
            omega.append(n.arctan2((bc - asqd), (ac + bsqd)))
            if omega[0] > n.pi:
                omega[0] = omega[0] - 2*n.pi

            cosOm = n.cos(omega[0])
            sinOm = n.sin(omega[0])
            omega_mat.append(n.dot(self.r_mat,n.array([[cosOm, -sinOm,  0],
                                                       [sinOm,  cosOm,  0],
                                                       [  0,  0  ,  1]])))
            g_t = n.dot(omega_mat[0], g_w)
            sineta = -g_t[1]*sinfact
            coseta =  g_t[2]*sinfact
            eta.append(n.arctan2(sineta, coseta))
            Gt.append(g_t/scale_Gw)

            #Find omega number 1
            #-------------------------------------------------
            omega.append(n.arctan2((bc + asqd), (ac - bsqd)))
            if omega[1] > n.pi:
                omega[1] = omega[1] - 2*n.pi
            cosOm = n.cos(omega[1])
            sinOm = n.sin(omega[1])
            omega_mat.append(n.dot(self.r_mat,n.array([[cosOm, -sinOm,  0],
                                                       [sinOm,  cosOm,  0],
                                                       [  0,  0  ,  1]])))
            g_t = n.dot(omega_mat[1], g_w)
            sineta = -g_t[1]*sinfact
            coseta =  g_t[2]*sinfact
            eta.append(n.arctan2(sineta, coseta))
            Gt.append(g_t/scale_Gw)
            #-------------------------------------------------

        return n.array(omega), n.array(eta), omega_mat, Gt


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
        v = n.array([costth, wavelength/(2*n.pi)*Gt[1], wavelength/(2*n.pi)*Gt[2]])
        t = (R_tilt[0, 0]*distance - n.dot(R_tilt[:, 0],n.array([tx, ty, tz])))/n.dot(R_tilt[:, 0],v)
        Ltv = n.array([tx-distance, ty, tz])+ t*v
        dety = n.dot(R_tilt[:, 1],Ltv)/y_size + dety_center
        detz = n.dot(R_tilt[:, 2],Ltv)/z_size + detz_center
        return [dety, detz]


    def calc_int_mod(self,omega,tx,ty,beam_width,v_area,bcy0,bcy1,bcy2):
        '''
        Compute intensity modifiers based on voxel beam overlap
        for the specific case of a square voxel and infinite rectangle
        beam. This method is meant to be faster than any geometry
        package for general polygons such as shapely and the like.
        e.g.

        |       |       |       |
        |    |||||||    |       |
        |    |||||||    |       |
        |    |||||||    |       |
        |region0|region1|region2|
        |       |       |       |
        => return 0.5, 0.5, 0

        Output: percentual overlap between voxel and the three
                beam regions possibly grazing the voxel:
                overlap leftmost region, overlap centre region, overlap rightmost resion

        '''

        alpha = omega - (n.pi/2.)*n.floor((omega/(n.pi/2.)))
        cos_neg = n.cos((n.pi/4.)-alpha)
        cos_pos = n.cos((n.pi/4.)+alpha)
        sin_neg = n.sin((n.pi/4.)-alpha)
        sin_pos = n.sin((n.pi/4.)+alpha)
        tan_factor = 0.5*(n.tan( alpha ) + n.tan( (n.pi/2.) - alpha ))
        diag = beam_width/n.sqrt(2)
        a = [-diag*cos_neg, diag*sin_neg]
        b = [diag*cos_pos, diag*sin_pos]
        c = [-a[0],-a[1]]
        d = [-b[0],-b[1]]

        int_mod_0=0
        int_mod_1=0
        int_mod_2=0

        if (-a[0]+ty)>(bcy1+beam_width*0.5):

            # a pertrudes bcy0
            if (-c[0]+ty)<(bcy1-beam_width*0.5):

                # Triangles in bcy0 and bcy2 with central hexagon in bc0
                xl = ty-(bcy1+beam_width*0.5)
                xr = ty-(bcy1-beam_width*0.5)
                Area_left_triangle = tan_factor*(xl+diag*cos_neg)*(xl+diag*cos_neg)
                Area_right_triangle = tan_factor*(diag*cos_neg-xr)*(diag*cos_neg-xr)
                Area_central_hexagon = v_area - Area_left_triangle - Area_right_triangle
                int_mod_0 = Area_left_triangle/v_area
                int_mod_1 = Area_central_hexagon/v_area
                int_mod_2 = Area_right_triangle/v_area
            else:


                # Tetragons in bcy0 and bcy1 or triangle and pentagon. Nothing in bcy2!
                xl_r = ty-(bcy1+beam_width*0.5)
                if alpha<(n.pi/4.):
                # b to the right of d
                    xl_t = n.min( n.array([d[0], ty-(bcy1+beam_width*0.5)]) )
                    #if alpha equals zero identicaly the computer cannot handle the limit as tan_fact=>inf
                    if alpha!=0:
                        Area_left_romboid = 2*tan_factor*(xl_t+diag*cos_neg)*(xl_r-xl_t)
                    else:
                        Area_left_romboid = (xl_r-xl_t)*beam_width
                else:
                # b to the left of d
                    xl_t = n.min( n.array([b[0], ty-(bcy1+beam_width*0.5)]) )
                    Area_left_romboid = 2*tan_factor*(xl_t+diag*cos_neg)*(xl_r-xl_t)
                Area_left_triangle = tan_factor*(xl_t+diag*cos_neg)*(xl_t+diag*cos_neg)
                Area_left_polygon = Area_left_triangle+Area_left_romboid#to much
                Area_right_polygon = v_area - Area_left_polygon
                int_mod_0 = Area_left_polygon/v_area
                int_mod_1 = Area_right_polygon/v_area
                int_mod_2 = 0
        else:
            # c pertrudes bcy2 (or alpha=0)
            # Tetragons in bcy1 and bcy2 or pentagon and triangle. Nothing in bcy0!
            if (-c[0]+ty)<(bcy1-beam_width*0.5): #we know central always has most points
                #compute triangle+romboid in bcy2!
                xr_r = ty-(bcy1-beam_width*0.5)
                if alpha<(n.pi/4.):
                # b to the right of d
                    xr_t = n.max( n.array([b[0], ty-(bcy1-beam_width*0.5)]) )
                    #if alpha equals zero identicaly the computer cannot handle the limit as tan_fact=>inf
                    if alpha!=0:
                        Area_right_romboid = 2*tan_factor*(diag*cos_neg-xr_t)*(xr_t-xr_r)
                    else:
                        Area_right_romboid = (xr_t-xr_r)*beam_width
                else:
                # b to the left of d
                    xr_t = n.max( n.array([d[0], ty-(bcy1-beam_width*0.5)]) )
                    Area_right_romboid = 2*tan_factor*(diag*cos_neg-xr_t)*(xr_t-xr_r)

                Area_right_triangle = tan_factor*(diag*cos_neg-xr_t)*(diag*cos_neg-xr_t)
                Area_right_polygon = Area_right_triangle+Area_right_romboid
                Area_left_polygon = v_area - Area_right_polygon
                int_mod_0 = 0
                int_mod_1 = Area_left_polygon/v_area
                int_mod_2 = Area_right_polygon/v_area
            else:
                #everything in bcy0
                return 0,1,0

        return int_mod_0, int_mod_1, int_mod_2


    def save(self,voxel_nbr=None):
        """
        write ModelScanning3DXRD ref file
        """
        file_io.write_ref(self.param,self.voxel,voxel_nbr)


    def write_gve(self):
        """
        Write gvector .gve file
        """
        file_io.write_gve(self.param,self.voxel,self.hkl)


    def write_ini(self):
        """
        write input file for voxelSpotter
        """
        file_io.write_ini(self.param,self.hkl)


    def write_flt(self):
        """
         Write filtered peaks .flt file
        """
        file_io.write_flt(self.param,self.voxel)
