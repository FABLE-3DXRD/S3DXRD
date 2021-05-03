'''
file_io.py is the module that actually writes results from find_refl.py to file.
Here formating of the results are done such that the standard fileformats
used at the ESRF are achived.
'''

from __future__ import absolute_import, print_function
import numpy as n
from xfab import tools,detector
from . import variables
import sys


A_id = variables.refarray().A_id



def write_merged_flt(param,merged_peaks):
    """
     Write filtered peaks (flt) file, for format see
     http://fable.wiki.sourceforge.net/imaged11+-+file+formats
    """

    filename = '%s/%s%s.flt' %(param['direc'],'merged_peaks_',param['stem'])

    with open(filename,'w') as f:

        out = '#  sc  fc  omega  Number_of_pixels  avg_intensity  s_raw  f_raw  sigs  sigf  covsf  sigo  covso  covfo  sum_intensity  sum_intensity^2  IMax_int  IMax_s  IMax_f  IMax_o  Min_s  Max_s  Min_f  Max_f  Min_o  Max_o  dety  detz  onfirst  onlast  spot3d_id  labels  tth_per_voxel  eta_per_voxel  h  k  l dty\n'
        f.write(out)
        out = ''
        #Local implementation of detecter.detyz_to_xy() for speed
        #--------------------------------------------------------
        if abs(param['o11']) == 1:
            if (abs(param['o22']) != 1) or (param['o12'] != 0) or (param['o21'] != 0):
                raise ValueError('detector orientation makes no sense 1')
        elif abs(param['o12']) == 1:
            if abs(param['o21']) != 1 or (param['o11'] != 0) or (param['o22'] != 0):
                raise ValueError('detector orientation makes no sense 2')
        else:
            raise ValueError('detector orientation makes no sense 3')

        omat = n.array([[param['o11'], param['o12']],
                        [param['o21'], param['o22']]])
        omat = n.linalg.inv(omat)
        det_size = n.array([param['detz_size']-1,
                            param['dety_size']-1])
        term2 = n.clip(n.dot(omat, det_size),-n.max(det_size), 0)
        def detyz_to_xy(coor,omat,det_size,term2):
            coor = n.array([coor[1], coor[0]])
            coor = n.dot(omat, coor) - term2
            return coor
        #-------------------------------------------------------------
        #    format = "%f "*3 + "%i "*1 +"%f "*12 + "%i "*2   +"%f "*1 + "%i "*4 +"%f "*4 + "%i "*3+"\n"
        format = "%f "*3 + "%i "*1 +"%f "*12 + "%i "*2   +"%f "*1 + "%i "*4 +"%f "*4 + "%i "*4 +"%f "*2 + "%i "*3 + "%f "*1 +"\n"

        for peak in merged_peaks:
            (sc, fc) = detector.detyz_to_xy([peak[A_id['dety']],peak[A_id['detz']]],
                                            param['o11'],
                                            param['o12'],
                                            param['o21'],
                                            param['o22'],
                                            param['dety_size'],
                                            param['detz_size'])
            if param['spatial'] == None:
                sr = sc
                fr = fc
            else:
                (sr, fr) = detector.detyz_to_xy([peak[A_id['detyd']],peak[A_id['detzd']]],
                                                param['o11'],
                                                param['o12'],
                                                param['o21'],
                                                param['o22'],
                                                param['dety_size'],
                                                param['detz_size'])
            out = out + (format %(sc,
                        fc,
                        peak[A_id['omega']]*180/n.pi,
                        peak[A_id['Number_of_pixels']],
                        peak[A_id['Int']]/peak[A_id['Number_of_pixels']],
                        sr,
                        fr,
                        1,
                        1,
                        0,
                        1,
                        0,
                        0,
                        peak[A_id['Int']],
                        (peak[A_id['Int']]/25)**2, #best estimate of sum_i I_i^2
                        peak[A_id['Int']]/10,
                        sc,
                        fc,
                        peak[A_id['omega']]*180/n.pi,
                        sc-2,
                        sc+2,
                        fc-2,
                        fc+2,
                        peak[A_id['omega']]*180/n.pi-param['omega_step'],
                        peak[A_id['omega']]*180/n.pi+param['omega_step'],
                        peak[A_id['dety']],
                        peak[A_id['detz']],
                        0,
                        0,
                        peak[A_id['spot_id']],
                        peak[A_id['voxel_id']],
                        peak[A_id['tth']],
                        peak[A_id['eta']],
                        peak[A_id['h']],
                        peak[A_id['k']],
                        peak[A_id['l']],
                        peak[A_id['dty']]
                        ) )
        f.write(out)
    from ImageD11 import columnfile as c
    peaks = c.columnfile(filename)
    print( "omegas", n.unique(peaks.omega), len(n.unique(peaks.omega)) )
    print("npeaks", peaks.nrows )

def write_delta_flt(param,voxel):
    if not memory_safety_check(param):
        return

    """
     Write filtered peaks (flt) file, for format see
     http://fable.wiki.sourceforge.net/imaged11+-+file+formats
    """

    filename = '%s/%s%s.flt' %(param['direc'],'delta_peaks_',param['stem'])

    with open(filename,'w') as f:

        out = '#  sc  fc  omega  Number_of_pixels  avg_intensity  s_raw  f_raw  sigs  sigf  covsf  sigo  covso  covfo  sum_intensity  sum_intensity^2  IMax_int  IMax_s  IMax_f  IMax_o  Min_s  Max_s  Min_f  Max_f  Min_o  Max_o  dety  detz  onfirst  onlast  spot3d_id  labels  tth_per_voxel  eta_per_voxel  h  k  l dty\n'
        f.write(out)

        # TODO: if we are writing huge files we need to be a little
        # bit tender with the ram usage. Which is still kinda stupid
        # as whoever tries to read such a file to RAM will get a
        # nasty suprise. Perhaps a better solution is to write several .flt files
        # Anyway, this should not be a problem once peakmerging is implemented

        if param['no_voxels']>100:
            collected_voxels = 0
            ranges = []
            while( True ):
                low_line = collected_voxels
                top_line = collected_voxels + 100
                if top_line < param['no_voxels']:
                    ranges.append([low_line,top_line])
                else:
                    ranges.append([low_line,param['no_voxels']])
                    break
                collected_voxels = top_line
        else:
            ranges = [[0,param['no_voxels']]]



        #Local implementation of detecter.detyz_to_xy() for speed
        #--------------------------------------------------------
        if abs(param['o11']) == 1:
            if (abs(param['o22']) != 1) or (param['o12'] != 0) or (param['o21'] != 0):
                raise ValueError('detector orientation makes no sense 1')
        elif abs(param['o12']) == 1:
            if abs(param['o21']) != 1 or (param['o11'] != 0) or (param['o22'] != 0):
                raise ValueError('detector orientation makes no sense 2')
        else:
            raise ValueError('detector orientation makes no sense 3')

        omat = n.array([[param['o11'], param['o12']],
                        [param['o21'], param['o22']]])
        omat = n.linalg.inv(omat)
        det_size = n.array([param['detz_size']-1,
                            param['dety_size']-1])
        term2 = n.clip(n.dot(omat, det_size),-n.max(det_size), 0)
        def detyz_to_xy(coor,omat,det_size,term2):
            coor = n.array([coor[1], coor[0]])
            coor = n.dot(omat, coor) - term2
            return coor
        #-------------------------------------------------------------
    #    format = "%f "*3 + "%i "*1 +"%f "*12 + "%i "*2   +"%f "*1 + "%i "*4 +"%f "*4 + "%i "*3+"\n"
        format = "%f "*3 + "%i "*1 +"%f "*12 + "%i "*2   +"%f "*1 + "%i "*4 +"%f "*4 + "%i "*4 +"%f "*2 + "%i "*3 + "%f "*1 +"\n"
        nrefl=0
        for ran in ranges:
            A = voxel[ ran[0] ].refs
            for voxelno in range(ran[0]+1,ran[1]):
                A = n.concatenate((A,voxel[voxelno].refs))
            #A = A[n.argsort(A,0)[:,A_id['omega']],:] # sort rows according to omega
            out =""
            for i in range(A.shape[0]):
                (sc, fc) = detector.detyz_to_xy([A[i,A_id['dety']],A[i,A_id['detz']]],
                                                param['o11'],
                                                param['o12'],
                                                param['o21'],
                                                param['o22'],
                                                param['dety_size'],
                                                param['detz_size'])
                if param['spatial'] == None:
                    sr = sc
                    fr = fc
                else:
                    (sr, fr) = detector.detyz_to_xy([A[i,A_id['detyd']],A[i,A_id['detzd']]],
                                                    param['o11'],
                                                    param['o12'],
                                                    param['o21'],
                                                    param['o22'],
                                                    param['dety_size'],
                                                    param['detz_size'])

                out = out + (format %(sc,
                            fc,
                            A[i,A_id['omega']]*180/n.pi,
                            A[i,A_id['Number_of_pixels']],
                            A[i,A_id['Int']]/A[i,A_id['Number_of_pixels']],
                            sr,
                            fr,
                            1,
                            1,
                            0,
                            1,
                            0,
                            0,
                            A[i,A_id['Int']],
                            (A[i,A_id['Int']]/25)**2, #best estimate of sum_i I_i^2
                            A[i,A_id['Int']]/10,
                            sc,
                            fc,
                            A[i,A_id['omega']]*180/n.pi,
                            sc-2,
                            sc+2,
                            fc-2,
                            fc+2,
                            A[i,A_id['omega']]*180/n.pi-param['omega_step'],
                            A[i,A_id['omega']]*180/n.pi+param['omega_step'],
                            A[i,A_id['dety']],
                            A[i,A_id['detz']],
                            0,
                            0,
                            A[i,A_id['spot_id']],
                            A[i,A_id['voxel_id']],
                            A[i,A_id['tth']],
                            A[i,A_id['eta']],
                            A[i,A_id['h']],
                            A[i,A_id['k']],
                            A[i,A_id['l']],
                            A[i,A_id['dty']]
                            ) )

            f.write(out)
            nrefl = nrefl + A.shape[0]
            print('\rFinished %3i of %3i voxels. Written %3i reflections to file' %(ran[1],param['no_voxels'],nrefl), end=' ')
            sys.stdout.flush()
        print("\n")





def write_voxels(param):
    '''
    Save the generated voxel parameters, pos, U and eps

    INPUT: The parameter set from the input file and the voxel generator
    OUTPUT: voxelno x y z phi1 PHI phi2 U11 U12 U13 U21 U22 U23 U31 U32 U33 eps11 eps12 eps13 eps22 eps23 eps33
    '''

    filename = '%s/%s.gff' %(param['direc'],param['stem'])

    with open(filename,'w') as f:
    #    format = "%d "*1 + "%f "*1 + "%e"*1 + "%f"*18 + "\n"
        format = "%d "*1 + "%d "*1 + "%f "*1 + "%e "*1 + "%f "*6 + "%0.12f "*9 + "%0.12f "*9 + "%e "*6 +"\n"
        out = "# voxel_id phase_id voxelsize voxelvolume x y z phi1 PHI phi2 U11 U12 U13 U21 U22 U23 U31 U32 U33 UBI11 UBI12 UBI13 UBI21 UBI22 UBI23 UBI31 UBI32 UBI33 eps11 eps12 eps13 eps22 eps23 eps33 \n"
        f.write(out)
        for i in range(param['no_voxels']):
            euler = 180/n.pi*tools.u_to_euler(param['U_voxels_%s' %(param['voxel_list'][i])])
            if len(param['phase_list']) == 1:
                phase = param['phase_list'][0]
            else:
                phase = param['phase_voxels_%s' %(param['voxel_list'][i])]
            b = tools.epsilon_to_b(param['eps_voxels_%s' %(param['voxel_list'][i])],
                                param['unit_cell_phase_%i' %phase ])
            u = param['U_voxels_%s' %(param['voxel_list'][i])]
            ubi = n.linalg.inv(n.dot(u,b))*2*n.pi

            out = format %(param['voxel_list'][i],
                        phase,
                        param['size_voxels_%s' %(param['voxel_list'][i])],
                        n.pi/6*(param['size_voxels_%s' %(param['voxel_list'][i])])**3.,
                        param['pos_voxels_%s' %(param['voxel_list'][i])][0],
                        param['pos_voxels_%s' %(param['voxel_list'][i])][1],
                        param['pos_voxels_%s' %(param['voxel_list'][i])][2],
                        euler[0],
                        euler[1],
                        euler[2],
                        param['U_voxels_%s' %(param['voxel_list'][i])][0,0],
                        param['U_voxels_%s' %(param['voxel_list'][i])][0,1],
                        param['U_voxels_%s' %(param['voxel_list'][i])][0,2],
                        param['U_voxels_%s' %(param['voxel_list'][i])][1,0],
                        param['U_voxels_%s' %(param['voxel_list'][i])][1,1],
                        param['U_voxels_%s' %(param['voxel_list'][i])][1,2],
                        param['U_voxels_%s' %(param['voxel_list'][i])][2,0],
                        param['U_voxels_%s' %(param['voxel_list'][i])][2,1],
                        param['U_voxels_%s' %(param['voxel_list'][i])][2,2],
                        ubi[0,0],
                        ubi[0,1],
                        ubi[0,2],
                        ubi[1,0],
                        ubi[1,1],
                        ubi[1,2],
                        ubi[2,0],
                        ubi[2,1],
                        ubi[2,2],
                        param['eps_voxels_%s' %(param['voxel_list'][i])][0],
                        param['eps_voxels_%s' %(param['voxel_list'][i])][1],
                        param['eps_voxels_%s' %(param['voxel_list'][i])][2],
                        param['eps_voxels_%s' %(param['voxel_list'][i])][3],
                        param['eps_voxels_%s' %(param['voxel_list'][i])][4],
                        param['eps_voxels_%s' %(param['voxel_list'][i])][5],
                            )
            f.write(out)


def write_hkl(param,hkl):
    """
    Write hkl file, for format
    """
    for phase in param['phase_list']:
        filename = '%s/%s_phase_%i.hkl' %(param['direc'],param['stem'],phase)
        f = open(filename,'w')
        with open(filename,'w') as f:
            for refl in range(len(hkl[phase])):
                (h,k,l,intint) = hkl[phase][refl]
                out = '%4i %4i %4i %8g\n'\
                    %(h,k,l, intint)
                f.write(out)


def write_fcf(param,hkl):
    """
    Write fcf file, cif hkl file

    python translation: March 31 2008
    changed October 1, 2008 to new .gve format including detector.par and [xl,yl,zl]
    """
    for phase in param['phase_list']:
        filename = '%s/%s_phase_%i.fcf' %(param['direc'],param['stem'],phase)

        with open(filename,'w') as f:
            f.write('data_phase_%i\n' %phase)
            f.write('loop_\n')
            f.write('_refln_index_h\n')
            f.write('_refln_index_k\n')
            f.write('_refln_index_l\n')
            f.write('_refln_F_squared_calc\n')
            for refl in range(len(hkl[phase])):
                (h,k,l,intint) = hkl[phase][refl]
                out = '%4i %4i %4i %10g\n'\
                    %(h,k,l, intint)
                f.write(out)
            f.write('\n')


def write_delta_gve(param,voxel,hkl):
    if not memory_safety_check(param):
        return
    """
    Write gvector (gve) file, for format see
    http://fable.wiki.sourceforge.net/imaged11+-+file+formats

    python translation: March 31 2008
    changed October 1, 2008 to new .gve format including detector.par and [xl,yl,zl]
    """


    # from detector.par
    (z_center, y_center) = detector.detyz_to_xy([param['dety_center'],param['detz_center']],
                            param['o11'],
                            param['o12'],
                            param['o21'],
                            param['o22'],
                            param['dety_size'],
                            param['detz_size'])
    dout = "# chi 0.0\n"
    dout = dout + "# distance %f\n" %(param['distance']*1000.)
    dout = dout + "# fit_tolerance 0.5\n"
    dout = dout + "# o11 %i\n" %param['o11']
    dout = dout + "# o12 %i\n" %param['o12']
    dout = dout + "# o21 %i\n" %param['o21']
    dout = dout + "# o22 %i\n" %param['o22']
    dout = dout + "# omegasign %f\n" %param['omega_sign']
    dout = dout + "# t_x 0\n"
    dout = dout + "# t_y 0\n"
    dout = dout + "# t_z 0\n"
    dout = dout + "# tilt_x %f\n" %param['tilt_x']
    dout = dout + "# tilt_y %f\n" %param['tilt_y']
    dout = dout + "# tilt_z %f\n" %param['tilt_z']
    dout = dout + "# y_center %f\n" %y_center
    dout = dout + "# y_size %f\n" %(param['y_size']*1000.)
    dout = dout + "# z_center %f\n" %z_center
    dout = dout + "# z_size %f\n" %(param['z_size']*1000.)

    for phase in param['phase_list']:
        if param['no_phases'] > 1:
            filename = '%s/%s%s_phase_%i.gve' %(param['direc'],'delta_peaks_',param['stem'],phase)
        else:
            filename = '%s/%s%s.gve' %(param['direc'],'delta_peaks_',param['stem'])

        with open(filename,'w') as f:
            lattice = param['sgname_phase_%i' %phase][0]
            format = "%f "*6 + "%s "*1 +"\n"
            unit_cell = param['unit_cell_phase_%i' %phase]
            out = format %(unit_cell[0],unit_cell[1],unit_cell[2],
                        unit_cell[3],unit_cell[4],unit_cell[5],
                        lattice)
            out = out + "# wavelength = %s\n" %(param['wavelength'])
            out = out + "# wedge = %f\n" %(param['wedge'])
            out = out + "# axis = 0.000 0.0000 1.0000\n"
            # insert detector.par as comment
            out = out + "# cell__a %s\n" %unit_cell[0]
            out = out + "# cell__b %s\n" %unit_cell[1]
            out = out + "# cell__c %s\n" %unit_cell[2]
            out = out + "# cell_alpha %s\n" %unit_cell[3]
            out = out + "# cell_beta %s\n" %unit_cell[4]
            out = out + "# cell_gamma %s\n" %unit_cell[5]
            out = out + "# cell_lattice_[P,A,B,C,I,F,R] %s\n" %param['sgname_phase_%i' %phase][0]
            out = out + dout
            # continue with gve format
            out = out +"# ds h k l\n"
            f.write(out)

            thkl = hkl[param['phase_list'].index(phase)].copy()
            ds = n.zeros((thkl.shape[0],1))

            for i in range(thkl.shape[0]):
                ds[i] = 2*tools.sintl(unit_cell,thkl[i,0:3])

            #Add ds values to the thkl array
            thkl = n.concatenate((thkl,ds),1)

            # sort rows according to ds, descending
            thkl = thkl[n.argsort(thkl,0)[:,4],:]

            # Output format
            format = "%f "*1 + "%d "*3 +"\n"

            for i in range(thkl.shape[0]):
                out = format %(thkl[i,4],
                            thkl[i,0],
                            thkl[i,1],
                            thkl[i,2]
                            )
                f.write(out)

            R_tilt = tools.detect_tilt(param['tilt_x'],param['tilt_y'],param['tilt_z'])

            out = "# xr yr zr xc yc ds eta omega spot3d_id xl yl zl\n"
            f.write(out)

            # TODO: if we are writing huge files we need to be a little
            # bit tender with the ram usage. Which is still kinda stupid
            # as whoever tries to read such a file to RAM will get a
            # nasty suprise. Perhaps a better solution is to write several .flt files
            # Anyway, this should not be a problem once peakmerging is implemented

            if param['no_voxels']>100:
                collected_voxels = 0
                ranges = []
                while( True ):
                    low_line = collected_voxels
                    top_line = collected_voxels + 100
                    if top_line < param['no_voxels']:
                        ranges.append([low_line,top_line])
                    else:
                        ranges.append([low_line,param['no_voxels']])
                        break
                    collected_voxels = top_line
            else:
                ranges = [[0,param['no_voxels']]]


            #Local implementation of detecter.detyz_to_xy() for speed
            #--------------------------------------------------------
            if abs(param['o11']) == 1:
                if (abs(param['o22']) != 1) or (param['o12'] != 0) or (param['o21'] != 0):
                    raise ValueError('detector orientation makes no sense 1')
            elif abs(param['o12']) == 1:
                if abs(param['o21']) != 1 or (param['o11'] != 0) or (param['o22'] != 0):
                    raise ValueError('detector orientation makes no sense 2')
            else:
                raise ValueError('detector orientation makes no sense 3')

            omat = n.array([[param['o11'], param['o12']],
                            [param['o21'], param['o22']]])
            omat = n.linalg.inv(omat)
            det_size = n.array([param['detz_size']-1,
                                param['dety_size']-1])
            term2 = n.clip(n.dot(omat, det_size),-n.max(det_size), 0)
            def detyz_to_xy(coor,omat,det_size,term2):
                coor = n.array([coor[1], coor[0]])
                coor = n.dot(omat, coor) - term2
                return coor
            #-------------------------------------------------------------

            format = "%f "*8 + "%i "*1+ "%f "*3+"\n"
            nrefl = 0
            for ran in ranges:
                out=""
                A = voxel[ ran[0] ].refs
                for voxelno in range(ran[0]+1,ran[1]):
                        A = n.concatenate((A,voxel[voxelno].refs))
                for i in range(A.shape[0]):
                    (sc, fc) = detector.detyz_to_xy([A[i,A_id['dety']],A[i,A_id['detz']]],
                                                param['o11'],
                                                param['o12'],
                                                param['o21'],
                                                param['o22'],
                                                param['dety_size'],
                                                param['detz_size'])

                    [xl,yl,zl] = detector.detector_to_lab(A[i,A_id['dety']],A[i,A_id['detz']],
                                                        param['distance'],
                                                        param['y_size'],param['z_size'],
                                                        param['dety_center'],param['detz_center'],
                                                    R_tilt)
                    out = out + (format %(A[i,A_id['gv1']]/(2*n.pi),
                                A[i,A_id['gv2']]/(2*n.pi),
                                A[i,A_id['gv3']]/(2*n.pi),
                                sc,#A[i,A_id['detz']],
                                fc,#param['dety_size']-A[i,A_id['dety']],
                                (2*n.sin(.5*A[i,A_id['tth']])/param['wavelength']),
                                A[i,A_id['eta']]*180/n.pi,
                                A[i,A_id['omega']]*180/n.pi,
                                A[i,A_id['spot_id']],
                                xl*1000.,
                                yl*1000.,
                                zl*1000.
                                ) )

                f.write(out)
                nrefl = nrefl + A.shape[0]
                print('\rFinished %3i of %3i voxels. Written %3i reflections to file' %(ran[1],param['no_voxels'],nrefl), end=' ')
                sys.stdout.flush()
            print("\n")




def write_ini(param,hkl):
    """
    Write ini file for voxelspotter, see
    http://fable.wiki.sourceforge.net/voxelspotter
    """


    for phase in param['phase_list']:
        out = '! input file for voxelSpotter made by ModelScanning3DXRD\n'

        if param['no_phases'] > 1:
            filename = '%s/%s_phase_%i.ini' %(param['direc'],param['stem'],phase)
            out = out + 'filespecs %s/%s_phase_%i.gve %s/%s_phase_%i.log\n' %(param['direc'],
                                                                              param['stem'],
                                                                              phase,
                                                                              param['direc'],
                                                                              param['stem'],
                                                                              phase)
        else:
            filename = '%s/%s.ini' %(param['direc'],param['stem'])
            out = out + 'filespecs %s/%s.gve %s/%s.log\n' %(param['direc'],
                                                            param['stem'],
                                                            param['direc'],
                                                            param['stem'])

        unit_cell = param['unit_cell_phase_%i' %phase]

        thkl = hkl[param['phase_list'].index(phase)].copy()
        ds = n.zeros((thkl.shape[0],1))

        for i in range(thkl.shape[0]):
            ds[i] = 2*tools.sintl(unit_cell,thkl[i,0:3])

        ds.sort()
        ds = ds.round(9)
        ds = n.unique(ds)
        families=  len(ds)
        Nhkls = n.min([families, 8])
        extra_hkls = 4
        if families-Nhkls < extra_hkls:
            tth_max = 2*param['theta_max']
        else:
            ds_max = ds[Nhkls+extra_hkls-1]+ 0.001
            tth_max = 2*n.arcsin(ds_max*param['wavelength']/2.)*180./n.pi

        ds_min = ds[0] - 0.001
        if ds_min < 0.0:
            ds_min = 0.
        tth_min = 2*n.arcsin(ds_min*param['wavelength']/2.)*180./n.pi

        out = out + 'spacegroup %i\n' %param['sgno_phase_%i' %phase]
        out = out + 'etarange %f %f\n'%(0.0, 360.0)
        out = out + 'domega %f\n' %param['omega_step']
        out = out + 'omegarange %f %f\n'   %(param['omega_start'],param['omega_end'])
        out = out + 'cuts %i %f %f\n' %(8, 0.6, 0.75)
        out = out + 'eulerstep %f\n' %(5.0)
        out = out + 'uncertainties %f %f %f\n' %(.05, 0.5, 1.0)
        out = out + 'nsigmas %f\n' %(2.0)
        out = out + 'Nhkls_in_indexing %i\n' %(Nhkls)
        out = out + 'tthrange %f %f\n' %(tth_min,tth_max)
        out = out + 'minfracg %f\n'%(0.95)


        with open(filename,'w') as f:
            f.write(out)

def write_par(param):
    """
    Save the detector parameters

    INPUT: The detector info used for the simulations
    OUTPUT: The corresponding detector.par file for ImageD11
    """

    #Prepare detector part of p.par output
    #Calc beam center in ImageD11 coordinate system
    (z_center, y_center) = detector.detyz_to_xy([param['dety_center'],param['detz_center']],
                            param['o11'],
                            param['o12'],
                            param['o21'],
                            param['o22'],
                            param['dety_size'],
                            param['detz_size'])

    dout = "chi 0.0\n"
    dout = dout + "distance %f\n" %(param['distance']*1000.)
    dout = dout + "fit_tolerance 0.5\n"
    dout = dout + "o11 %i\n" %param['o11']
    dout = dout + "o12 %i\n" %param['o12']
    dout = dout + "o21 %i\n" %param['o21']
    dout = dout + "o22 %i\n" %param['o22']
    dout = dout + "omegasign %f\n" %param['omega_sign']
    dout = dout + "t_x 0\n"
    dout = dout + "t_y 0\n"
    dout = dout + "t_z 0\n"
    dout = dout + "tilt_x %f\n" %param['tilt_x']
    dout = dout + "tilt_y %f\n" %param['tilt_y']
    dout = dout + "tilt_z %f\n" %param['tilt_z']
    dout = dout + "wavelength %f\n" %param['wavelength']
    dout = dout + "wedge %f\n" %(param['wedge'])
    dout = dout + "y_center %f\n" %y_center
    dout = dout + "y_size %f\n" %(param['y_size']*1000.)
    dout = dout + "z_center %f\n" %z_center
    dout = dout + "z_size %f\n" %(param['z_size']*1000.)

    for phase in param['phase_list']:
        if param['no_phases'] > 1:
                filename = '%s/%s_phase_%i.par' %(param['direc'],param['stem'],phase)
        else:
                filename = '%s/%s.par' %(param['direc'],param['stem'])

        with open(filename,'w') as f:
            unit_cell = param['unit_cell_phase_%i' %phase]
            out = "cell__a %s\n" %unit_cell[0]
            out = out + "cell__b %s\n" %unit_cell[1]
            out = out + "cell__c %s\n" %unit_cell[2]
            out = out + "cell_alpha %s\n" %unit_cell[3]
            out = out + "cell_beta %s\n" %unit_cell[4]
            out = out + "cell_gamma %s\n" %unit_cell[5]
            out = out + "cell_lattice_[P,A,B,C,I,F,R] %s\n" %param['sgname_phase_%i' %phase][0]
            out = out + dout

            f.write(out)




def write_ref(param,voxel,voxelno=None):
    """
    write ModelScanning3DXRD ref file
    """

    if voxelno == None:
        savevoxels = list(range(len(voxel)))
    else:
        savevoxels = voxelno

    for voxelno in savevoxels:
        A = voxel[voxelno].refs
        setno = 0
        filename = '%s/%s_gr%0.4d_set%0.4d.ref' \
            %(param['direc'],param['stem'],param['voxel_list'][voxelno],setno)

        with open(filename,'w') as f:
            format = "%d "*6 + "%f "*14 + "%d "*1 + "\n"
            out = "#"
            A_col = dict([[v,k] for k,v in A_id.items()])
            for col in A_col:
                out = out + ' %s' %A_col[col]
            out = out +"\n"

            f.write(out)
            # Only write reflections to file if some present
            if len(A) > 0:
                ( nrefl, ncol ) = A.shape
                for i in range(nrefl):
                    out = format %(A[i,A_id['voxel_id']],
                                A[i,A_id['ref_id']],
                                A[i,A_id['spot_id']],
                                A[i,A_id['h']],
                                A[i,A_id['k']],
                                A[i,A_id['l']],
                                A[i,A_id['tth']]*180/n.pi,
                                A[i,A_id['omega']]*180/n.pi,
                                A[i,A_id['eta']]*180/n.pi,
                                A[i,A_id['dety']],
                                A[i,A_id['detz']],
                                A[i,A_id['detyd']],
                                A[i,A_id['detzd']],
                                A[i,A_id['gv1']],
                                A[i,A_id['gv2']],
                                A[i,A_id['gv3']],
                                A[i,A_id['L']],
                                A[i,A_id['P']],
                                A[i,A_id['F2']],
                                A[i,A_id['Int']],
                                A[i,A_id['overlaps']]
                        )
                    f.write(out)



def write_res(param,filename=None):
    """
    Save the generated voxel parameters in an input file to facilitate restart of ModelScanning3DXRD with same voxels

    INPUT: ModelScanning3DXRD input and generated voxel parameters
    OUTPUT: .res file in ModelScanning3DXRD input format
    """
    if filename == None:
        filename = '%s/%s.res' %(param['direc'],param['stem'])

    with open(filename,'w') as f:
        #initialise and sort keys alphabetically
        out = ""
        keys = list(param.keys())
        keys.sort()

        for item in keys:
            # rule out None entries
            if param[item] is not None:
                # treat all strings, remember quotation marks
                if type(param[item]) == str:
                    out += "%s '%s'\n" %(item,param[item])
                # treat all lists, special case for strings
                elif type(param[item]) == list:
                    out += '%s' %item
                    for i in range(len(param[item])):
                        if type(param[item][i]) == str:
                            out += " '%s'" %param[item][i]
                        else:
                            out += ' %s' %param[item][i]
                    out += '\n'
                # treat all arrays, loop over one or two dimensions
                elif type(param[item]) == n.ndarray:
                    out += '%s' %item
                    dim = len(n.shape(param[item]))
                    if dim == 1:
                        for i in range(len(param[item])):
                            out += ' %s' %param[item][i]
                    elif dim == 2:
                        for i in range(len(param[item])):
                            for j in range(len(param[item][i])):
                                out += ' %s' %param[item][i][j]
                    out += '\n'
                # remaining entries; integers and floats
                else:
                    out += "%s %s\n" %(item,param[item])

        f.write(out)

def write_ubi(param):
    '''
    Construct UBI matrices for each voxel and save them to file.
    If the voxels have been mapped to grains we also output the grain average
    UBI matrices.
    '''

    # If we specified which voxels belong to which grains
    # we also get a per voxel average UBI file
    if param['voxel_grain_map'] is not None:

        m = param['voxel_grain_map']
        grain_U = {}
        grain_eps = {}
        #Loop over grain id:s (val = grain number)
        for val in m.values():
            if (val not in grain_U.keys()) and (val not in grain_eps.keys()) :
                #construct dictionary where the key is grain number and the value is the
                #list of properties of the voxels populating that grain
                grain_U[val] = []
                grain_eps[val] = []

        #loop over all voxels and fill up the dictionaries
        for voxel in range(param['no_voxels']):
            voxelId = param['voxel_list'][voxel]
            U = param['U_voxels_%s' % voxelId]
            gr_eps = n.array(param['eps_voxels_%s' % voxelId])
            voxel_grain_id = m[voxelId]
            grain_U[voxel_grain_id].append(U)
            grain_eps[voxel_grain_id].append(gr_eps)


        filename = '%s/grain_average_%s.ubi' %(param['direc'],param['stem'])
        with open(filename,'w') as f:
            format = "%f "*3 + "\n"

            for voxel in grain_U.keys():
                avg_U = n.zeros((3,3))
                avg_eps = n.array([0.,0.,0.,0.,0.,0.])
                for U,eps in zip(grain_U[voxel],grain_eps[voxel]):
                    avg_U += U
                    avg_eps += eps
                avg_U[:,0] = avg_U[:,0]/n.linalg.norm(avg_U[:,0])
                avg_U[:,1] = avg_U[:,1]/n.linalg.norm(avg_U[:,1])
                avg_U[:,2] = avg_U[:,2]/n.linalg.norm(avg_U[:,2])
                avg_eps = avg_eps/len(grain_eps[voxel])
                #TODO: implement several phases!
                if len(param['phase_list'])>1:
                    raise ValueError('Currently only single phase simulations are supported')
                phase = param['phase_list'][0]
                avg_B = tools.epsilon_to_b(avg_eps,param['unit_cell_phase_%i' %phase])/(2*n.pi)
                avg_UBI = n.linalg.inv(n.dot(avg_U,avg_B))

                for j in range(3):
                    out = format %(avg_UBI[j,0],avg_UBI[j,1],avg_UBI[j,2])
                    f.write(out)
                out = "\n"
                f.write(out)


    filename = '%s/voxel_%s.ubi' %(param['direc'],param['stem'])

    with open(filename,'w') as f:
        format = "%f "*3 + "\n"
        for i in range(param['no_voxels']):
            U = param['U_voxels_%s' %(param['voxel_list'][i])]
            gr_eps = n.array(param['eps_voxels_%s' %(param['voxel_list'][i])])
            if param['no_phases'] == 1:
                phase = param['phase_list'][0]
            else:
                phase = param['phase_voxels_%s' %(param['voxel_list'][i])]
            # Clculate the B-matrix based on the strain tensor for each voxel
            B = tools.epsilon_to_b(gr_eps,param['unit_cell_phase_%i' %phase])/(2*n.pi)
            UBI = n.linalg.inv(n.dot(U,B))
            for j in range(3):
                out = format %(UBI[j,0],UBI[j,1],UBI[j,2])
                f.write(out)
            out = "\n"
            f.write(out)


def memory_safety_check(param):
    if param['no_voxels']<250:
        return True
    else:
        print("WARNING")
        print("---------------------------------------------------------")
        print("It is not safe to write delta peaks to file")
        print("for this many voxels. The files will be to large")
        print("Make do with the merged .flt or if you are bold and")
        print("have a lot of RAM remove this code and try to use delta peaks")
        print("---------------------------------------------------------")
        return False
