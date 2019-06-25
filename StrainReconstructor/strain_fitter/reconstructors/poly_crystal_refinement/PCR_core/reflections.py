'''
The reflections.py script is used to generate miller indices
and structure factors for intensity computations.
'''
from __future__ import absolute_import
import numpy as n
from xfab import tools,structure,sg


def gen_miller(param,phase):
    """
    Generate set of miller indices.
    Changed from old tools.genhkl to new tools.genhkl_all
    """

    sintlmin = n.sin(param['theta_min']*n.pi/180)/param['wavelength']
    sintlmax = n.sin(param['theta_max']*n.pi/180)/param['wavelength']

    hkl  = tools.genhkl_all(param['unit_cell_phase_%i' %phase],
                sintlmin,
                sintlmax,
                sgname=param['sgname_phase_%i' %phase],
                cell_choice=param['cell_choice_phase_%i' %phase],
                )

    return hkl

def open_structure(param,phase):
    file = param['structure_phase_%i' %phase]
    if file[-3:] == 'cif':
        if ('structure_datablock_phase_%i' %phase) in param:
            datablock = param['structure_datablock_phase_%i' %phase]
        else:
            datablock = None
        struct = structure.build_atomlist()
        struct.CIFread(ciffile=file,cifblkname=datablock)
    elif file[-3:] == 'pdb':
        struct = structure.build_atomlist()
        struct.PDBread(pdbfile=file)
    else:
        raise IOError('Unknown structure file format')
    param['sgno_phase_%i' %phase] = sg.sg(sgname=struct.atomlist.sgname).no
    param['sgname_phase_%i' %phase] = struct.atomlist.sgname
    param['cell_choice_phase_%i' %phase] = sg.sg(sgname=struct.atomlist.sgname).cell_choice
    param['unit_cell_phase_%i' %phase] =  struct.atomlist.cell
    return struct

def calc_intensity(hkl,struct):
    """
    Calculate the reflection intensities
    """
    int = n.zeros((len(hkl),1))
    for i in range(len(hkl)):

        (Fr, Fi) = structure.StructureFactor(hkl[i],
                             struct.atomlist.cell,
                             struct.atomlist.sgname,
                             struct.atomlist.atom,
                             struct.atomlist.dispersion)
        int[i] = Fr**2 + Fi**2
    hkl = n.concatenate((hkl,int),1)
    return hkl

def add_intensity(hkl,param):
    """
    Calculate the reflection intensities
    """
    if 'structure_int' in param:
        myint = param['structure_int']
    else:
        myint = 2**15

    myint = n.ones((len(hkl),1))*myint
    hkl = n.concatenate((hkl,myint),1)
    return hkl

