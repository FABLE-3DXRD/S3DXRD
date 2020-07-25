from strain_fitter.reconstructors.single_crystal_refinement.SCR import SCR
from strain_fitter.reconstructors.poly_crystal_refinement.PCR import PCR
from strain_fitter.reconstructors.algebraic_strain_refinement.ASR import ASR
from volume import Volume

'''Example reconstruction of simulated nonconvex grain data'''

# Define reconstruction parameters
zpos=[0, 1, 2, 3, 4, 5, 6]
labels = ['z'+str(zp) for zp in zpos]
grain_files = ['test_data/nonconvex_'+zz+'.ubi' for zz in labels]
param_file = 'test_data/nonconvex_z3.par'
flt_files = ['test_data/nonconvex_'+zz+'.flt' for zz in labels]
omegastep = 1.0    # rotation interval
hkltol = 0.05      # Indexing tolerance
nmedian = 10000    # for removing outliers (should be inf for noise free data)
rcut = 0.35        # Segment threshold for FBP
ystep = 10
ymin = -70
ymax = 70
yunit = 10**(-6) # (microns)
savedir = 'test_saves'
gradient_constraint = 5*(10**-4)

# Pick reconstruction methods
reconstructors = {}
reconstructors[savedir+'/SCR'] =  SCR( param_file )
#reconstructors[savedir+'/PCR'] = PCR( param_file, cif_file='test_data/Sn.cif', omegastep=omegastep)
#reconstructors[savedir+'/ASR'] = ASR( param_file, omegastep, gradient_constraint=gradient_constraint, maxiter=maxiter, number_cpus=number_cpus )

# Perform reconstruction
for key in reconstructors:
    print(key)
    v = Volume(reconstructors[key], omegastep, hkltol, nmedian, rcut, ymin, ymax, ystep, yunit, save_dir=key)
    v.reconstruct( flt_files, grain_files, labels, z_positions=zpos, interpolate=False )
