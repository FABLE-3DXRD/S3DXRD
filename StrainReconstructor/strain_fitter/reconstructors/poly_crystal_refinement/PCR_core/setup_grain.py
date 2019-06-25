'''
Module for creating a find_refl_func object
properly representing the grain investigated for simul fit
'''
import numpy as np
from ImageD11 import columnfile,grain
import reflections
from xfab import tools
from xfab import detector

def get_hkls(param,flt,grain):
    hkl = []
    phase=0 #currently only single phase samples used
    xtal_structure = reflections.open_structure(param,phase)
    print('Generating miller indices')
    hkl_tmp = reflections.gen_miller(param,phase)
    print('Structure factor calculation')
    hkl.append(reflections.calc_intensity(hkl_tmp, xtal_structure))

    active_hkls = get_active_hkls(flt,grain)

    # for a in active_hkls:
    #     if a[0]==5 and a[1]==6:
    #         print(a)
    # print("active_hkls",active_hkls)
    # raise

    print("len(hkl) before discarding",len(hkl[0]))
    hkl = discard_unused_hkls(active_hkls,hkl)
    print("len(hkl) after discarding",len(hkl[0]))

    # for a in hkl:
    #     for b in a:
    #         if b[0]==5 and b[1]==6:
    #             print(b)
    # raise

    return hkl

def get_active_hkls(flt,grain):
    active_hkls = []

    hs  = flt.h[grain.mask]
    ks  = flt.k[grain.mask]
    ls  = flt.l[grain.mask]
    for h,k,l in zip(hs,ks,ls):
        active_hkls.append([h,k,l])

    return active_hkls

def discard_unused_hkls(active_hkls,computed_hkls):

    updated_hkls=[]
    for hkl in computed_hkls[0]:
        if [hkl[0],hkl[1],hkl[2]] in active_hkls:
            updated_hkls.append([hkl[0],hkl[1],hkl[2],hkl[3]])
    return [np.asarray(updated_hkls)]

def get_positions(grain_mask,beam_width):
    '''
    beam_width needs to be in microns it is here converted to mm

    Assume center of rotation at center of grain_mask matrix
    '''

    beam_width_mm = beam_width/1000.
    center_rotation_x = (grain_mask.shape[0]/2.)*beam_width_mm
    center_rotation_y = (grain_mask.shape[1]/2.)*beam_width_mm
    N = np.sum(grain_mask)
    voxel_positions=np.zeros((N,3))
    n=0
    for i in range(grain_mask.shape[0]):
        for j in range(grain_mask.shape[1]):
            if grain_mask[i,j]!=0:
                # voxel_positions[n,0] = (i*beam_width_mm  + beam_width_mm /2.) - center_rotation_x
                # voxel_positions[n,1] = (j*beam_width_mm  + beam_width_mm /2.) - center_rotation_y
                # voxel_positions[n,2] = 0
                # n+=1
                voxel_positions[n,0] = -((i*beam_width_mm  + beam_width_mm /2.) - center_rotation_x)
                voxel_positions[n,1] = (j*beam_width_mm  + beam_width_mm /2.) - center_rotation_y
                voxel_positions[n,2] = 0
                n+=1

    # print(voxel_positions)
    # import matplotlib.pyplot as plt
    # plt.scatter(np.asarray(1000*voxel_positions)[:,1],1000*np.asarray(voxel_positions)[:,0])
    # plt.show()
    #raise
    C, constraint = build_connectivity(voxel_positions, beam_width_mm)
    return voxel_positions, C, constraint

def build_connectivity(voxel_positions, beam_width_mm):
    nvox = voxel_positions.shape[0]
    nvars = nvox*9
    p=np.array([1]*9)
    n=np.array([-1]*9)
    s = 5*(10**(-4))
    a = np.radians( 2.5 )
    C = []
    connected = []
    for i,pos0 in enumerate(voxel_positions):
        for j,pos1 in enumerate(voxel_positions):
            if np.linalg.norm(pos0-pos1)<=np.sqrt(2)*beam_width_mm and [i,j] not in connected:
                for k in range(0,6):
                    row = np.zeros(nvars)
                    row[i*9+k]=1/s
                    row[j*9+k]=-1/s
                    C.append(row)
                for k in range(6,9):
                    row = np.zeros(nvars)
                    row[i*9+k]=1/a
                    row[j*9+k]=-1/a
                    C.append(row)
                connected.append([i,j])
                connected.append([j,i])
    C = np.array(C)
    #constraint = np.array( [s,s,s,s,s,s,a,a,a]*int(C.shape[0]/9.0) )
    constraint = np.ones( C.shape[0] )
    return C, constraint


def map_solution_to_voxels(solution,grain_mask,no_voxels,voxel_data,var_choice='strain'):
    '''
    Take the solution array and create crystal grain objects
    mapping the right voxel to the proper UB.

    the output map gives the voxel of the index which is the key

    The index refers to the index of the grain_mask
    I.e. the thresholded reconstruction matrix from the FBP
    '''

    #Convert to matrix format
    solution_as_matrices = []
    #print(solution)
    #print(len(solution))
    for i in range(len(solution)//9):
        low = i*9
        mid = low+6
        high = low + 9
        #print(solution[low:mid])
        if var_choice=='strain':
            UB = voxel_data.strain_and_euler_to_ub(solution[low:mid],solution[mid:high])
        elif var_choice=='cell':
            UB = voxel_data.cell_and_euler_to_ub(solution[low:mid],solution[mid:high])
        solution_as_matrices.append(np.asarray(UB))
    solution_as_matrices = np.asarray(solution_as_matrices)

    #solution_as_matrices = solution.reshape((no_voxels,3,3))
    #Convert to ubi
    for i,ub in enumerate(solution_as_matrices):
        ubi = np.linalg.inv(ub)*(2*np.pi) # U and B as defined in xfab.tools (not ID11 indexing)
        solution_as_matrices[i] = ubi

    voxels_as_grains={}
    n=0
    for i in range(grain_mask.shape[0]):
        for j in range(grain_mask.shape[1]):
            if grain_mask[i,j]!=0:
                voxel = grain.grain( solution_as_matrices[n] )
                voxels_as_grains[i,j]=voxel
                n+=1
    return voxels_as_grains

def get_theta_bounds(flt,grain):
    theta = flt.tth[grain.mask]/2.
    # print(flt.tth[0:100])
    #print(theta[0:15])
    theta_min = min(theta)
    theta_max = max(theta)
    # print("theta_min",theta_min)
    # print("theta_max",theta_max)
    #raise
    return theta_min,theta_max


def setup_experiment(pars, cif_file, no_voxels,beam_width,theta_min,theta_max):
    '''
    beam_width needs to be in microns it is here converted to mm
    '''
    o11 = pars.parameters['o11']
    o12 = pars.parameters['o12']
    o21 = pars.parameters['o21']
    o22 = pars.parameters['o22']
    dety_size = 2048
    detz_size = 2048
    (dety_center,detz_center) = detector.xy_to_detyz([ pars.parameters['z_center'], pars.parameters['y_center'] ], o11, o12, o21, o22, dety_size, detz_size)


    param = {
    'no_voxels':no_voxels,
    'beam_width':beam_width/1000.,
    'structure_phase_0': cif_file,
    'omega_start':0,
    'omega_end':180,
    'omega_step': 1.0,
    'omega_sign':pars.parameters['omegasign'],
    'wavelength':pars.parameters['wavelength'],
    'detz_center': detz_center,#1041.55049636,#detz_center
    'dety_center': dety_center,#998.798998,#dety_center
    'z_size':pars.parameters['z_size']/1000.,
    'y_size':pars.parameters['y_size']/1000.,
    'distance':pars.parameters['distance']/1000.,
    'o11':pars.parameters['o11'],
    'o12':pars.parameters['o12'],
    'o21':pars.parameters['o21'],
    'o22':pars.parameters['o22'],
    'dety_size':dety_size,
    'detz_size':detz_size,
    'lorentz_apply':1,
    'theta_min':0,
    'theta_max':12,#i.e only two theta = 2*theta_max or lower is computed for
    'tilt_x':pars.parameters['tilt_x'],
    'tilt_y':pars.parameters['tilt_y'],
    'tilt_z':pars.parameters['tilt_z'],
    'wedge':pars.parameters['wedge'],
    }


    return param

def get_measured_data(grain,flt,number_y_scans,ymin,ystep, param):
    '''
    np array size (no_dtys,no_refl,3)

    The data is sorted in falling priority:
    1. sorted by dty low to high
    2. sorted by h low to high
    3. sorted by k low to high
    4. sorted by l low to high
    '''
    o11 = param['o11']
    o12 = param['o12']
    o21 = param['o21']
    o22 = param['o22']
    dety_size = param['dety_size']
    detz_size = param['detz_size']

    n = len(flt.sc[grain.mask])
    sc = flt.sc[grain.mask]
    fc = flt.fc[grain.mask]
    omega = flt.omega[grain.mask]
    dty = flt.dty[grain.mask]
    h = flt.h[grain.mask]
    k = flt.k[grain.mask]
    l = flt.l[grain.mask]
    all_measured_data = []

    for i in range(n):
        (dety, detz) = detector.xy_to_detyz([ sc[i], fc[i] ], o11, o12, o21, o22, dety_size, detz_size)
        all_measured_data.append([dety,detz,omega[i],dty[i],h[i],k[i],l[i]])

    all_measured_data.sort(key=lambda x: x[3])
    data_by_dty = []


    k=len(all_measured_data)
    print("before binning in dty",k)


    for i in range(number_y_scans):
        data_by_dty.append([])
    print("number_y_scans",number_y_scans)

    # currdata=[]
    # currdata.append(all_measured_data[0])
    # dty_curr = all_measured_data[0][3]
    # for i in range(1,len(all_measured_data)):
    #     if dty_curr!=all_measured_data[i][3]:
    #         currdata_sorted = sorted(currdata, key=lambda x: (x[4], x[5], x[6]))
    #         iy = np.round( (dty_curr - ymin) / ystep ).astype(int)
    #         data_by_dty[iy] = np.asarray(currdata_sorted)
    #         currdata = []
    #         dty_curr = all_measured_data[i][3]
    #     currdata.append(all_measured_data[i])
    p=0
    for peak in all_measured_data:
        iy = np.round( (peak[3] - ymin) / ystep ).astype(int)
        # print(peak[3],"=>",iy)
        data_by_dty[iy].append(peak)
        p+=1
    print("appended",p)

    for i,dty in enumerate(data_by_dty):
        data_by_dty[i] = sorted(dty, key=lambda x: (x[4], x[5], x[6]))

    k=0
    for dty in data_by_dty:
        k+=len(dty)
    print("after sorting and binning in dty",k)
    #raise

    return data_by_dty

def get_bounds_cell(x0):
    '''
    Take an initial guess and return two numpy arrays
    first contains lower bounds second upper.
    '''
    bounds_low=np.zeros((len(x0),))
    bounds_high=np.zeros((len(x0),))
    for i in range(len(x0)//9):
        low = i*9
        mid_low = low+3
        mid_high = low+6
        high = low + 9

        bounds_low[low:mid_low] = x0[low:mid_low]*0.99
        bounds_high[low:mid_low] = x0[low:mid_low]*1.01

        bounds_low[mid_low:mid_high] = x0[mid_low:mid_high] - 0.1
        bounds_high[mid_low:mid_high] = x0[mid_low:mid_high] + 0.1

        bounds_low[mid_high:high] = x0[mid_high:high] - (0.1*np.pi/180.)
        bounds_high[mid_high:high] = x0[mid_high:high] + (0.1*np.pi/180.)
    return bounds_low,bounds_high

def get_bounds_strain(x0):
    '''
    Take an initial guess and return two numpy arrays
    first contains lower bounds second upper.
    '''
    bounds_low=np.zeros((len(x0),))
    bounds_high=np.zeros((len(x0),))
    for i in range(len(x0)//9):
        low = i*9
        mid_low = low+3
        mid_high = low+6
        high = low + 9

        deps = 0.05
        bounds_low[low:mid_low] = x0[low:mid_low] - deps
        bounds_high[low:mid_low] = x0[low:mid_low] + deps

        bounds_low[mid_low:mid_high] = x0[mid_low:mid_high] - deps
        bounds_high[mid_low:mid_high] = x0[mid_low:mid_high] + deps

        bounds_low[mid_high:high] = x0[mid_high:high] - (0.1*np.pi/180.)
        bounds_high[mid_high:high] = x0[mid_high:high] + (0.1*np.pi/180.)

    return bounds_low,bounds_high

