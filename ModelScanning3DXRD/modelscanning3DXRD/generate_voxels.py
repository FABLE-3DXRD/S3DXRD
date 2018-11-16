'''
generate_voxels.py is used to generate random voxels.
'''

from __future__ import absolute_import
from __future__ import print_function
import numpy as n
from xfab import tools
from xfab import sg
from xfab import symmetry


def generate_U(no_voxels,sgi):
	'''
	generate random U (orientations) for no_voxels voxels
	All U must be in same fundamental zone, thus trace of U must be maximal using the allowed permutations
	INPUT: no_voxels
	OUTPUT: [[U11 U12 U13],[U21 U22 U23],[U31 U32 U33]] for each voxel
	'''

	print('voxel orientations will be randomly generated\n')
	U = n.zeros((no_voxels,3,3))
	Urot = n.zeros((3,3))
	for i in range(no_voxels):
	    #http://www.cognitive-antics.net/mw/index.php?title=Uniform_random_orientation
	    tilt_z = n.random.rand()*2*n.pi
	    tilt_y = n.arcsin(n.random.rand())
	    tilt_x  = n.pi*(2*n.random.rand()*n.pi-1)
	    U[i] = tools.detect_tilt(tilt_x, tilt_y, tilt_z)
	    t = 0
	    Ut = U[i].copy()
	    symmetries = ['triclinic','monoclinic', 'orthorhombic','tetragonal','trigonal','hexagonal','cubic']
	    crystal_system = symmetries.index(sgi.crystal_system)+1
	    rot = symmetry.rotations(crystal_system)
	    for j in range(len(rot)):
	        Urot = n.dot(U[i],rot[j])
	        trace = Urot.trace()
	        if trace > t:
	            t = trace
	            Ut = Urot
	    U[i] = Ut
	return U

def generate_pos(no_voxels,gen_pos,sample_xyz=None,sample_cyl=None):
	'''
	generate random positions for no_voxels voxels for a given sample shape,
	EITHER a box (sample_xyz != None) OR a cylinder (sample_cyl != None)
	If no shape is given or gen_pos == 0 default to (0,0,0)
	INPUT: 	no_voxels
					gen_pos (==0 default to (0,0,0), else random positions)
		sample_xyz (optional) sample dimensions along x, y and z
		sample_cyl (optional) sample cylinder diameter and length (along z)
	OUTPUT: [x y z] for each voxel
	'''

	pos = n.zeros((no_voxels,3))
	if sample_xyz != None and gen_pos != 0:
		print('voxel positions will be randomly generated within a box of ', sample_xyz[0], sample_xyz[1], sample_xyz[2], 'mm\n')
		for i in range(no_voxels):
			for j in range(3):
				pos[i,j] = (n.random.rand()-0.5)*sample_xyz[j]

	elif sample_cyl != None and gen_pos != 0:
		print('voxel positions will be randomly generated')
		print('within a cylinder of diameter ', sample_cyl[0], ' and length ', sample_cyl[1], 'mm\n')
		for i in range(no_voxels):
			r = n.random.rand()*sample_cyl[0]/2.
			w = n.random.rand()*2*n.pi
			z = (n.random.rand()-0.5)*sample_cyl[1]
			pos[i] = [r*n.cos(w), r*n.sin(w), z]

	else:
		print('voxel positions will be set to (0,0,0)\n')

	return pos

def generate_eps(no_voxels,gen_eps):
	'''
	generate strain tensor components for no_voxels voxels using a normal distribution with
	mean and spread for diagonal and off-diagonal elements as specified by gen_eps

	INPUT: no_voxels, gen_eps =[mean_diag,spread_diag,mean_offdiag,spread_offdiag]
	OUTPUT: [eps11, eps12, eps13, eps22, eps23, eps33] for each voxel
	'''

	eps = n.zeros((no_voxels,6))
	diag = [0,3,5]
	offdiag = [1,2,4]

	if gen_eps == [0,0,0,0]:
		print('No strain\n')
	else:
		print('voxel strain tensors will be randomly generated using a normal distribution')
		print('For diagonal elements, mean: ', gen_eps[0], ' and spread: ', gen_eps[1])
		print('For off-diagonal elements, mean: ', gen_eps[2], ' and spread: ', gen_eps[3],'\n')

		for i in range(no_voxels):
			if gen_eps[1] == 0:   # non-zero spread of diagonal elements
				for j in diag:
					eps[i,j] = gen_eps[0]

			else:
				for j in diag:
					eps[i,j] = n.random.normal(gen_eps[0],gen_eps[1])

			if gen_eps[3] == 0:   # non-zero spread of off-diagonal elements
				for j in offdiag:
					eps[i,j] = gen_eps[2]

			else:
				for j in offdiag:
					eps[i,j] = n.random.normal(gen_eps[2],gen_eps[3])

	return eps

def voxel_size(no_voxels,voxel_size,voxel_min_max,sample_vol=None):
	'''
	generate voxel sizes (mean diameters) from a lognormal distribution
	with a specified mean (voxel_size) and upper and lower cut-offs given by voxel_min_max
	For specified sample shape and size calculate the space filling fraction

	INPUT: no_voxels, gen_size (mean size),
	              voxel_min_max (optional)[min,max]
	              sample_xyz (optional) sample volume
	OUTPUT: mean diameter (in mm) for each voxel
	'''


	size = n.zeros((no_voxels))
	voxel_vol = 0
	if voxel_size < 0:
		print('All voxels have the following mean diameter: ', abs(voxel_size), ' mm\n')
		for i in range(no_voxels):
			size[i] = abs(voxel_size)
			voxel_vol = voxel_vol + n.pi/6*size[i]**3

	else:
	# NB the standard lognormal distribution
        # lognormal() = exp(normal(mean=0, spread=1) has a mean of exp(.5)
		print('voxel sizes from a lognormal distribution with a mean diameter of: ',\
		abs(voxel_size), ' mm\n')
		for i in range(no_voxels):
			while size[i] <= voxel_min_max[0] or size[i] > voxel_min_max[1]:
				size[i] = voxel_size*n.random.lognormal()/n.exp(.5)

			voxel_vol = voxel_vol + n.pi/6*size[i]**3

	if sample_vol != None:
		fraction = voxel_vol/sample_vol
		print('The generated voxels cover the following fraction of the sample volume: %6f' %fraction)
	return size


def generate_voxels(param):
	# If pick out voxel for phase
    if param['gen_phase'][0] != 0:
		#Making random list of voxel numbers
        rand_voxel_order = n.argsort(n.random.randn(param['no_voxels']))

        no_picked_voxels = 0
        for i in range(param['no_phases']):
            phase = param['gen_phase'][i*2+1]
            no_voxels_phase = int(param['gen_phase'][i*2+2])
            param['no_voxels_phase_%i' %phase] = no_voxels_phase
            param['voxel_list_phase_%i' %phase] = \
                n.array(param['voxel_list'])[rand_voxel_order[no_picked_voxels:no_picked_voxels+no_voxels_phase]]
            no_picked_voxels += no_voxels_phase
            for voxel in param['voxel_list_phase_%i' %phase]:
                param['phase_voxels_%i' %voxel] = phase
    param['gen_phase'][0] = 0  # Done

	# Generate U if gen_U on
    if param['gen_U'] != 0:
        for phase in param['phase_list']:
            U = generate_U(param['no_voxels_phase_%i' %phase],sgi = sg.sg(sgname=param['sgname_phase_%i' %phase]))
            for i in range(param['no_voxels_phase_%i' %phase]):
                param['U_voxels_%s' %(param['voxel_list_phase_%i' %phase][i])] = U[i]
        param['gen_U'] = 0

	# Generate pos if gen_pos on
    if param['gen_pos'][0] != 0:
        pos = generate_pos(param['no_voxels'],param['gen_pos'][1],sample_xyz=param['sample_xyz'],sample_cyl=param['sample_cyl'])
        for i in range(param['no_voxels']):
            param['pos_voxels_%s' %(param['voxel_list'][i])] = pos[i]
        param['gen_pos'][0] = 0

	# Generate eps if gen_eps on
    for phase in param['phase_list']:
        if param['gen_eps_phase_%i' %phase][0] != 0:
            eps = generate_eps(param['no_voxels_phase_%i' %phase],param['gen_eps_phase_%i' %phase][1:5])
            for i in range(param['no_voxels_phase_%i' %phase]):
                param['eps_voxels_%s' %(param['voxel_list_phase_%i' %phase][i])] = eps[i]
            param['gen_eps_phase_%i' %phase][0] = 0
    if 'gen_eps' in param:
        del param['gen_eps']


	# Generate size if gen_size on
    for phase in param['phase_list']:
        if param['gen_size_phase_%i' %phase][0] != 0:
            size = voxel_size(param['no_voxels_phase_%i' %phase],
                              param['gen_size_phase_%i' %phase][1],
                              param['gen_size_phase_%i' %phase][2:4],
                              param['sample_vol']*param['vol_frac_phase_%i' %phase])
            for i in range(param['no_voxels_phase_%i' %phase]):
                param['size_voxels_%s' %(param['voxel_list_phase_%i' %phase][i])] = size[i]
                param['gen_size_phase_%i' %phase][0] = 0
    if 'gen_size' in param:
        del param['gen_size']


def gen_odf(sigma,pos,mapsize):
    """
    Generate a 3D orietation distribution function as a 3D Gaussian
    """
    odf = n.zeros(tuple([int(m) for m in mapsize]))

    for i in range(int(mapsize[0])):
        for j in range(int(mapsize[1])):
            for k in range(int(mapsize[2])):
                odf[i,j,k] = -0.5*(((i-pos[0])/sigma[0])**2\
			          +((j-pos[1])/sigma[1])**2\
                                  +((k-pos[2])/sigma[2])**2)

    odf = 1/(2*n.pi*n.sqrt(2*n.pi)*sigma[0]*sigma[1]*sigma[2])*n.exp(odf)
    return odf
