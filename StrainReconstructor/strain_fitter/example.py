from strain_fitter.reconstructors.single_crystal_refinement.SCR import SCR
from strain_fitter.reconstructors.poly_crystal_refinement.PCR import PCR
from strain_fitter.reconstructors.algebraic_strain_refinement.ASR import ASR
from volume import Volume


gradient_tol = 5*(10**-4)
cif_file = 'test_data/Sn.cif'


grain_files = ['test_data/sixGrads.ubi']
param_file = 'test_data/sixGrads.par'
flt_files = ['test_data/sixGrads.flt']
labels = ['0']

# grain_files = ['test_data/relaxed.ubi']
# param_file = 'test_data/relaxed.par'
# flt_files = ['test_data/relaxed.flt']
# labels = ['0']

# grain_files = ['test_data/grain_average_Sn_ssch.ubi']
# param_file = 'test_data/Sn_ssch.par'
# flt_files = ['test_data/merged_peaks_Sn_ssch.flt']
# labels = ['0']

# base = 'test_data/whisker/'
# grain_files = [base+'whisker_simfit.map']
# param_file = base+'tin_fitted.par'
# flt_files = [base+'whisker_simfit.flt.new']
# labels = ['0']

# base = '/home/axel/workspace/scanning-3dxrd-simulations/ModelScanning3DXRD/scripts/simul/whiskerData/'
# grain_files = [base+'whisker_locref.map']
# param_file = base+'tin_fitted.par'
# flt_files = [base+'whisker_locref.flt.new']
# labels = ['0']


# Recon real data:
#----------------------------------
# base = '/home/axel/workspace/whisker_data/slices/z000/'
# param_file = '/home/axel/workspace/whisker_data/slices/tin_fitted.par'
# labels = ['z000']
# z_positions=[0]
# selected_grain = {}
# selected_grain['label'] = 'z000'
# selected_grain['index'] = 38
# flt_files = [base+'tin.flt.new']
# grain_files = [base+'tin_z000.map']
#------------------------------------

omegastep = 1.0
hkltol = 0.05
nmedian = 50000
rcut = 0.35

reconstructors = {}
reconstructors['test_saves/SCR'] =  SCR( param_file )
reconstructors['test_saves/PCR'] = PCR( param_file, cif_file, omegastep)
reconstructors['test_saves/ASR'] = ASR( param_file, omegastep, gradient_tol )

for key in reconstructors:
    print(key)
    v = Volume(reconstructors[key], omegastep, hkltol, nmedian, rcut, save_dir=key)
    #v.sinogram( 38, flt_files[0], grain_files[0])
    #v.reconstruct( flt_files, grain_files, labels,z_positions=z_positions, selected_grain=selected_grain )
    v.reconstruct( flt_files, grain_files, labels )

#v.reconstruct( flt_files, grain_files, labels, selected_grain=37 )
# 28 and 37
