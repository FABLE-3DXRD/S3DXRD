'''
OBS: This module has not been adapted for ModelScanning3DXRD v.1.0 !
This module is used to produce .edf detector images from the reflection information.
Such images are intensity images, like grayscale images from a camera.
This is basically the raw data one would recive from an experiment.
OBS: This module has not been adapted for ModelScanning3DXRD v.1.0 !
'''

from __future__ import absolute_import
from __future__ import print_function
import sys,time
from xfab import tools,detector
from fabio import edfimage,tifimage
from . import variables,check_input
from scipy import ndimage
from scipy.stats import norm
import numpy as n
from PIL import Image

#TODO: adapt for ModelScanning3DXRD

A_id = variables.refarray().A_id

# Set for 2kx2k detector, 10 deg corresponding to 1k
# and then reduced by a factor of 2-3 to get smooth peaks
Delta_tth = 0.005 #approx size of one pixel in deg tth
Delta_eta = 0.02 #approx size of one pixel in deg eta

class make_image:
    def __init__(self,voxeldata,killfile=None):
        self.voxeldata = voxeldata
        self.killfile = killfile

    def make_image(self,frame_number=None):
        """
        makeimage script produces edf diffraction images using the reflection information
        """
        peakshape = self.voxeldata.param['peakshape']

        if peakshape[0] == 0: # spike peak, 2x2 pixels
            peak_add = 1
            frame_add = 1
            peakwsig = 0
        elif peakshape[0] == 1: # 3d Gaussian peak
            peak_add = max(1,int(round(peakshape[1])))
            frame_add = max(1,int(round(peakshape[1])))
            peakwsig = peakshape[2]
        elif peakshape[0] == 3: # 3d Gaussian peak in 2theta,eta,omega
            peak_add = 1
            frame_add = 1
            cen_tth = int(1.5*peakshape[1]/Delta_tth)
            frame_tth = 2*cen_tth + 1
            fwhm_tth = peakshape[1]/Delta_tth
            cen_eta = int(1.5*peakshape[2]/Delta_eta)
            frame_eta = 2*cen_eta + 1
            fwhm_eta = peakshape[2]/Delta_eta
            raw_tth_eta = n.zeros((frame_tth,frame_eta))
            raw_tth_eta[cen_tth,cen_eta] = 1
            filter_tth_eta = ndimage.gaussian_filter(raw_tth_eta,[0.5*fwhm_tth,0.5*fwhm_eta])
            peakwsig = 1.

        framedimy = int(self.voxeldata.param['dety_size']+2*frame_add)
        framedimz = int(self.voxeldata.param['detz_size']+2*frame_add)

        totalrefl = 0
        if frame_number == None:
            no_frames = list(range(len(self.voxeldata.frameinfo)))
            print('Generating diffraction images')
        else:
            no_frames = [frame_number]

        for i in no_frames:
            check_input.interrupt(self.killfile)
            t1 = time.clock()
            nrefl = 0
            frame = n.zeros((framedimy,framedimz))
            omega = self.voxeldata.frameinfo[i].omega
            omega_step = self.voxeldata.param['omega_step']
            # Hack to add relative movement of sample and detector, modelled to be Gaussian in y and z direction with a spread of 1 micron
                # movement of 1 micron along x judged to be irrelevant, at least for farfield data
            y_move = n.random.normal(0,1./self.voxeldata.param['dety_size'])
            z_move = n.random.normal(0,1./self.voxeldata.param['detz_size'])
                # loop over voxels
            for j in range(self.voxeldata.param['no_voxels']):
                # loop over reflections for each voxel
                gr_pos = n.array(self.voxeldata.param['pos_voxels_%s' %j])
                for k in range(len(self.voxeldata.voxel[j].refs)):
                    # exploit that the reflection list is sorted according to omega
                    if self.voxeldata.voxel[j].refs[k,A_id['omega']]*180/n.pi > \
                            omega+omega_step+2*peakwsig:
                        break
                    elif self.voxeldata.voxel[j].refs[k,A_id['omega']]*180/n.pi < \
                            omega-2*peakwsig:
                        continue
                    dety = self.voxeldata.voxel[j].refs[k,A_id['detyd']]   # must be spot position after
                    detz = self.voxeldata.voxel[j].refs[k,A_id['detzd']]   # applying spatial distortion
                #apply hack
#                   dety = self.voxeldata.voxel[j].refs[k,A_id['dety']] + y_move
#                   detz = self.voxeldata.voxel[j].refs[k,A_id['detz']] + z_move
                    ndety = int(round(dety))
                    ndetz = int(round(detz))
                    yrange = list(range(ndety+frame_add-peak_add,ndety+frame_add+peak_add+1))
                    zrange = list(range(ndetz+frame_add-peak_add,ndetz+frame_add+peak_add+1))
                    intensity = int(round(self.voxeldata.voxel[j].refs[k,A_id['Int']]))
                    nrefl = nrefl + 1
                    totalrefl = totalrefl + 1
                    # Gaussian along omega
                    if peakshape[0] == 1 or peakshape[0] == 3:
                        fraction = norm.cdf((omega-self.voxeldata.voxel[j].refs[k,A_id['omega']]*180/n.pi+omega_step)/(0.5*peakwsig))\
                                  -norm.cdf((omega-self.voxeldata.voxel[j].refs[k,A_id['omega']]*180/n.pi)/(0.5*peakwsig))
                    else:
                        fraction = 1.
                    if peakshape[0] == 3:
                    # Gaussian peaks along 2theta,eta
                        tth = self.voxeldata.voxel[j].refs[k,A_id['tth']]
                        eta = self.voxeldata.voxel[j].refs[k,A_id['eta']]
                        Om = tools.form_omega_mat_general(self.voxeldata.voxel[j].refs[k,A_id['omega']],0,-1.*self.voxeldata.param['wedge']*n.pi/180.)
                        [tx,ty,tz]= n.dot(Om,gr_pos)
                        for t in range(frame_tth):
                            tth_present = tth + (t-cen_tth)*Delta_tth*n.pi/180.
                            for e in range(frame_eta):
                                eta_present = eta + (e-cen_eta)*Delta_eta*n.pi/180.
                                [dety_present, detz_present] = detector.det_coor2(tth_present,
                                                                                  eta_present,
                                                                                  self.voxeldata.param['distance'],
                                                                                  self.voxeldata.param['y_size'],
                                                                                  self.voxeldata.param['z_size'],
                                                                                  self.voxeldata.param['dety_center'],
                                                                                  self.voxeldata.param['detz_center'],
                                                                                  self.voxeldata.R,
                                                                                  tx,
                                                                                  ty,
                                                                                  tz,)

                                if self.voxeldata.param['spatial'] != None :
                                    from ImageD11 import blobcorrector
                                    self.spatial = blobcorrector.correctorclass(self.voxeldata.param['spatial'])
                                    # To match the coordinate system of the spline file
                                    # SPLINE(i,j): i = detz; j = (dety_size-1)-dety
                                    # Well at least if the spline file is for frelon2k
                                    (x,y) = detector.detyz_to_xy([dety_present,detz_present],
                                                              self.voxeldata.param['o11'],
                                                              self.voxeldata.param['o12'],
                                                              self.voxeldata.param['o21'],
                                                              self.voxeldata.param['o22'],
                                                              self.voxeldata.param['dety_size'],
                                                              self.voxeldata.param['detz_size'])
                                    # Do the spatial distortion
                                    (xd,yd) = self.spatial.distort(x,y)

                                    # transform coordinates back to dety,detz
                                    (dety_present,detz_present) = detector.xy_to_detyz([xd,yd],
                                                                    self.voxeldata.param['o11'],
                                                                    self.voxeldata.param['o12'],
                                                                    self.voxeldata.param['o21'],
                                                                    self.voxeldata.param['o22'],
                                                                    self.voxeldata.param['dety_size'],
                                                                    self.voxeldata.param['detz_size'])

                                y = round(dety_present)
                                z = round(detz_present)
                                try:
                                    frame[y+frame_add,z+frame_add] = frame[y+frame_add,z+frame_add] + fraction*intensity*filter_tth_eta[t,e]
                                except:
                                    pass



                    else:
                    # Generate spikes, 2x2 pixels
                        for y in yrange:
                            for z in zrange:
                                if y > 0 and y < framedimy and z > 0 and z < framedimz and abs(dety+frame_add-y) < 1 and abs(detz+frame_add-z) < 1:
#                                   frame[y-1,z] = frame[y-1,z] + fraction*intensity*(1-abs(dety+frame_add-y))*(1-abs(detz+frame_add-z))
                                    frame[y,z] = frame[y,z] + fraction*intensity*(1-abs(dety+frame_add-y))*(1-abs(detz+frame_add-z))

            # 2D Gaussian on detector
            if peakshape[0] == 1:
                frame = ndimage.gaussian_filter(frame,peakshape[1]*0.5)
            # add background
            if self.voxeldata.param['bg'] > 0:
                frame = frame + self.voxeldata.param['bg']*n.ones((framedimy,framedimz))
            # add noise
            if self.voxeldata.param['noise'] != 0:
                frame = n.random.poisson(frame)
            # apply psf
            if self.voxeldata.param['psf'] != 0:
                frame = ndimage.gaussian_filter(frame,self.voxeldata.param['psf']*0.5)
            # resize, convert to integers and flip to same orientation as experimental frames
            frame = frame[frame_add:framedimy-frame_add,frame_add:framedimz-frame_add]

                    # limit values above 16 bit to be 16bit
            frame = n.clip(frame,0,2**16-1)
                        # convert to integers
            frame = n.uint16(frame)

            #flip detector orientation according to input: o11, o12, o21, o22
            frame = detector.trans_orientation(frame,
                               self.voxeldata.param['o11'],
                               self.voxeldata.param['o12'],
                               self.voxeldata.param['o21'],
                               self.voxeldata.param['o22'],
                               'inverse')
            # Output frames
            if '.edf' in self.voxeldata.param['output']:
                self.write_edf(i,frame)
            if '.tif' in self.voxeldata.param['output']:
                self.write_tif(i,frame)
            if '.tif16bit' in self.voxeldata.param['output']:
                self.write_tif16bit(i,frame)
            print('\rDone frame %i took %8f s' %(i+1,time.clock()-t1), end=' ')
            sys.stdout.flush()

    def write_edf(self,framenumber,frame):
        e=edfimage.edfimage()
        e.data=frame
        e.dim2,e.dim1=frame.shape
        e.header = {}
        e.header['origin']='ModelScanning3DXRD'
        e.header['Dim_1']=e.dim1
        e.header['Dim_2']=e.dim2
        e.header['col_end']=e.dim1-1
        e.header['row_end']=e.dim2-1
        e.header['DataType']='UnsignedShort'
        e.header['Image']=1
        e.header['ByteOrder']='Low'
        e.header['time']=time.asctime()
        e.header['Omega']= self.voxeldata.frameinfo[framenumber].omega +\
            self.voxeldata.param['omega_step']/2.0
        e.header['OmegaStep']=self.voxeldata.param['omega_step']
        e.header['voxelfile']='%s/%s_%0.4dvoxels.txt' \
            %(self.voxeldata.param['direc'],self.voxeldata.param['stem'],self.voxeldata.param['no_voxels'])
        e.write('%s%s' %(self.voxeldata.frameinfo[framenumber].name,'.edf'))

    def write_tif(self,framenumber,frame):
        e=tifimage.tifimage()
        e.data=frame
        e.write('%s%s' %(self.voxeldata.frameinfo[framenumber].name,'.tif'))
    def write_tif16bit(self,framenumber,frame):
        size = frame.shape[:2][::-1]
        pilimage = Image.frombuffer('I',size,frame.tostring(),"raw",'I;16',0,1)
        pilimage.save('%s%s' %(self.voxeldata.frameinfo[framenumber].name,'.tif'))



