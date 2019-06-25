#!/usr/bin/env python

'''
check_input is used to parse the input file to python
strings and floats which are saved in dictionaries. Here we can perform checks to see
if the input makes sense. Should you choose to add some new input option
to the input file of ModelScanning3DXRD this is the place to parse that input.
'''

from __future__ import absolute_import
from __future__ import print_function
from copy import copy
import sys, os
from . import variables
from xfab import tools,sg
import numpy as n

def interrupt(killfile):
    if killfile is not None and os.path.exists(killfile):
        raise KeyboardInterrupt

class parse_input:
    def __init__(self,input_file = None):
        '''
        The parse_input object contains dictionaries carrying all the
        possible input keywords as keys and eventually their inputed values as
        values.
        '''

        self.filename = input_file
        self.param = {}

        # Experimental setup
        self.needed_items = {
                    'wavelength' : 'Missing input: wavelength [wavelength in angstrom]',
                    'distance'   : 'Missing input: distance [sample-detector distance in mm)]',
                    'dety_center': 'Missing input: dety_center [beamcenter, y in pixel coordinatees]',
                    'detz_center': 'Missing input: detz_center [beamcenter, z in pixel coordinatees]',
                    'y_size'     : 'Missing input: y_size [Pixel size y in mm]',
                    'z_size'     : 'Missing input: z_size [Pixel size z in mm]',
                    'dety_size'  : 'Missing input: dety_size [detector y size in pixels]',
                    'detz_size'  : 'Missing input: detz_size [detector z size in pixels]',
                    'omega_start': 'Missing input: omega_start [Omega start in degrees]',
                    'omega_end'  : 'Missing input: omega_end [Omega end in degrees]',
                    'omega_step' : 'Missing input: omega_step [Omega step size in degrees]',
                    'no_voxels'  : 'Missing input: no_voxels [number of voxels]',
                    'direc'      : 'Missing input: direc [directory to save output]',
                                        }

        self.optional_items = {
            'sgno'  : None,
            'sgname': None,
            'tilt_x': 0,
            'tilt_y': 0,
            'tilt_z': 0,
            'wedge': 0.0,
            'beam_width': None,
            'beampol_apply' : 1,
            'beampol_factor': 1,
            'beampol_direct' : 0.0,
            'lorentz_apply' : 1,
            'start_frame': 0,
            'omega_sign': 1,
            'noise' : 0,
            'psf': 0,
            'make_image': 1,
            'o11': 1,
            'o12': 0,
            'o21': 0,
            'o22': 1,
            'bg' : 0,
            'peakshape': [0,0],
            'spatial' : None,
            'flood' : None,
            'dark' : None,
            'darkoffset' : None,
            'gen_phase': [0],
            'gen_U' : 0,
            'gen_pos' : [0,0],
            'gen_eps' : [0,0,0,0,0],
            'gen_size' : [0,0,0,0],
            'sample_xyz': None,
            'sample_cyl': None,
            'direc' : '.',
            'stem': 'test',
            'odf_type' : 1,
            'odf_scale' : 0.02,
            'odf_cut' : None,
            'odf_sub_sample': 1,
            'mosaicity' : 0.2,
            'theta_min' : 0.0,
            'theta_max' : None,
            'unit_cell' : None,
            'structure_file': None,
            'odf_file': None,
            'output': None,
            'structure_factors': 1,
            'no_phases': 1,
            'intensity_const': 0,
            'voxel_grain_map' : None,
            'two_theta_upper_bound' : 90,
            'two_theta_lower_bound' : 0
            }

        self.output_types = ['.edf',
                             '.tif',
                             '.ref',
                             '.par',
                             'delta.flt',
                             'merged.flt',
                             'delta.gve',
                             '.ini',
                             '.ubi']

    def read(self):
        '''
        Read the input file to a string and parse it.
        This method fills the self.param dictionary with
        key value pairs.
        '''

        try:
            f = open(self.filename,'r')
        except IOError:
            raise IOError('No file named %s' %self.filename)

        self.input = f.readlines()
        f.close()

        for lines in self.input:
            if lines.find('#') != 0:
                if lines.find('#') > 0:
                    lines = lines.split('#')[0]
                line = lines.split()
                if len(line) != 0:
                    key = line[0]
                    val = line[1:]
                    # This ensures that file names with space can be handled
                    if key == 'direc' or key == 'stem' or key == 'spatial' or 'structure' in key:
                        valtmp = ''
                        valend = ''
                        sepa = ' '
                    else:
                        valtmp = '['
                        valend = ']'
                        sepa = ','

                    if len(val) > 1 or key == "output":
                        for i in val:
                            valtmp = valtmp + i + sepa
                        # remove last separator
                        val = valtmp[:-len(sepa)] + valend
                    else:
                        #print key,val
                        val = val[0]

                    # Problems using the Windows path separator as they can
                    # cause part of a string to be interpreted as a special charactor
                    # like \n - newline.
                    if key == 'direc' or key == 'stem' or key == 'spatial' or ('structure' in key and 'factors' not in key):
                        # Hack to remove a final backslash in the directory path
                        if str(val).endswith('\\\''):
                            val = val[:-2]+"'"
                        # Unfortunately the escape character can be changed like this
                        val = val.replace('\\x','/x')
                        # before taking care of the rest
                        val = eval(val)
                        val = val.replace('\t','\\t')
                        val = val.replace('\n','\\n')
                        val = val.replace('\c','\\c')
                        val = val.replace('\b','\\b')
                        val = val.replace('\f','\\f')
                        val = val.replace('\r','\\r')
                        val = val.replace('\v','\\v')
                        # Added value to key
                        self.param[key] = val
                    else:
                        self.param[key] = eval(val)

    def check(self):
        '''
        This method will perform hughe amounts of tedious checks to try and
        catch bad input.
        '''
        self.missing = False
        self.errors = {}

        # check that all needed items are present
        for item in self.needed_items:
            if item not in self.param:
                #print self.needed_items[item]
                #self.missing = True
                self.errors[item] =  self.needed_items[item]

        # set all non-read items to defaults
        for item in self.optional_items:
            if (item not in self.param):
                self.param[item] = self.optional_items[item]
            if (self.param[item] == []):
                self.param[item] = self.optional_items[item]*self.param['no_voxels']

        # assert that the correct number of arguments are given
        for key in self.param:
            val = self.param[key]
            if val != None and key != 'output' and key != 'voxel_list':
                if key == 'peakshape':
                    if type(val) == type(1):
                        if val != 0:
                            self.errors[key] = 'Wrong number of arguments'
                        else:
                            val = [0, 0]
                            self.param[key] = val
                    else:
                        if len(val) > 5:
                            self.errors[key] = 'Wrong number of arguments'
                elif key == 'sample_cyl' or key == 'gen_pos':
                    if len(val) != 2:
                        self.errors[key] = 'Wrong number of arguments'
                elif key == 'sample_xyz' or 'pos_voxels' in key:
                    if len(val) != 3:
                        self.errors[key] = 'Wrong number of arguments'
                elif 'gen_size' in key:
                    if len(val) != 4:
                        self.errors[key] = 'Wrong number of arguments'
                elif 'gen_eps' in key:
                    if type(val) == type(1):
                        if val != 0:
                            self.errors[key] = 'Wrong number of arguments'
                    else:
                        if len(val) != 5:
                            self.errors[key] = 'Wrong number of arguments'
                elif key == 'gen_phase':
                    try:
                        dummy = len(val)
                    except:
                        val = [val]
                        self.param[key] = val
                    if len(val) < 1:
                        self.errors[key] = 'Wrong number of arguments'
                elif key == 'unit_cell' or 'eps_voxels' in key:
                    if len(val) != 6:
                        self.errors[key] = 'Wrong number of arguments'
                elif 'U_voxels' in key:
                    if len(val) != 3:
                        if len(val) != 9:
                            self.errors[key] = 'Wrong number of arguments'
                        else:
                            self.param[key] = n.array(self.param[key])
                            self.param[key].shape = (3,3)
                    else:
                        if  val.shape != (3,3):
                            self.errors[key] = 'Wrong number of arguments'
                    # reshape U-matrices
            elif key == 'output':
                for item in val:
                    if item not in self.output_types:
                        self.errors[key] = 'Output type given %s is not an option' %item

        # Check no of phases
        no_phases = self.param['no_phases']

        phase_list_structure = []
        phase_list_unit_cell = []
        phase_list_sgno = []
        phase_list_sgname = []
        phase_list_gen_size = []
        phase_list_gen_eps = []
        phase_list = []

        for item in self.param:
            if '_phase_' in item:
                if 'structure' in item:
                    phase_list_structure.append(eval(item.split('_phase_')[1]))
                elif 'unit_cell' in item:
                    phase_list_unit_cell.append(eval(item.split('_phase_')[1]))
                elif 'sgno' in item:
                    phase_list_sgno.append(eval(item.split('_phase_')[1]))
                elif 'sgname' in item:
                    phase_list_sgname.append(eval(item.split('_phase_')[1]))
                elif 'gen_size' in item:
                    phase_list_gen_size.append(eval(item.split('_phase_')[1]))
                elif 'gen_eps' in item:
                    phase_list_gen_eps.append(eval(item.split('_phase_')[1]))


        phase_list_structure.sort()
        phase_list_unit_cell.sort()
        phase_list_sgno.sort()
        phase_list_sgname.sort()
        phase_list_gen_size.sort()
        phase_list_gen_eps.sort()

        if len(phase_list_structure) != 0:
            if len(phase_list_structure) != no_phases:
                self.errors['phase_list_structure'] = \
                    'Input number of structural phases does not agree with number\n' +\
                    ' of structure_phase, check for multiple names or missing files.'
            else:
                phase_list = phase_list_structure
        elif  len(phase_list_unit_cell) != 0:
            if len(phase_list_unit_cell) != no_phases:
                self.errors['phase_list_unit_cell'] = \
                    'Input number of structural phases does not agree with number\n' +\
                    ' of unit_cell, check for multiple names or missing linies.'
            else:
                phase_list = phase_list_unit_cell
            if len(phase_list_sgno) == 0:
                if len(phase_list_sgname) != no_phases:
                    self.errors['phase_list_sgname'] = \
                        'Input number of structural phases does not agree with number\n' +\
                        'of space group information given (sgno or sgname),\n' +\
                        'check for multiple names or missing linies.'
                if phase_list_sgname != phase_list_unit_cell:
                    self.errors['phase_list_sgname_2'] = \
                        'The phase numbers given to unit_cell does not match those in sgname'

                # add sgno for phase to parameter list
                for phase in phase_list_sgname:
                    self.param['sgno_phase_%i' %phase] = sg.sg(sgname = self.param['sgname_phase_%i' %phase]).no
                    self.param['cell_choice_phase_%i' %phase] = sg.sg(sgname = self.param['sgname_phase_%i' %phase]).cell_choice

            elif len(phase_list_sgname) == 0:
                if len(phase_list_sgno) != no_phases:
                    self.errors['phase_list_sgno'] = \
                        'Input number of structural phases does not agree with number\n' +\
                        'of space group information given (sgno or sgname),\n' +\
                        'check for multiple names or missing linies.'
                if phase_list_sgno != phase_list_unit_cell:
                    self.errors['phase_list_sgno_2'] = \
                        'The phase numbers given to unit_cell does not match those in sgno.'

                # add sgname for phase to parameter list
                for phase in phase_list_sgno:
                    try:
                        self.param['cell_choice_phase_%i' %phase] = sg.sg(sgno = self.param['sgno_phase_%i' %phase],cell_choice=self.param['cell_choice_phase_%i' %phase]).cell_choice
                    except:
                        self.param['cell_choice_phase_%i' %phase] = sg.sg(sgno = self.param['sgno_phase_%i' %phase]).cell_choice
                    self.param['sgname_phase_%i' %phase] = sg.sg(sgno = self.param['sgno_phase_%i' %phase],cell_choice=self.param['cell_choice_phase_%i' %phase]).name
            else:
                # both sg numbers and names in input check if they point at the same space group
                for phase in phase_list_sgno:
                    if self.param['sgname_phase_%i' %phase] != sg.sg(sgno = self.param['sgno_phase_%i' %phase]).name:
                        self.errors['sgname_phase_list_sgno_2'] = \
                            '\nSpace group is specified both as space group name and number - \n' + \
                            'and they do not correspond to the same space group. \n' + \
                            'Please sort this out in the input file for phase %i.' %phase


        if len(phase_list_gen_size) != 0:
            if len(phase_list_gen_size) != no_phases:
                self.errors['phase_list_gen_size_1'] = \
                    'Input number of structural phases does not agree with number\n' +\
                    'of gen_size_phase_ keywords given'

            if phase_list_gen_size != phase_list:
                self.errors['phase_list_gen_size_2'] = \
                    'The phase numbers given to gen_size_phase does not match those\n' +\
                    'in crystallographic part - structure_phase_X or unit_cell_phase_X.'
            self.param['gen_size'][0] = 1
        else:
            if len(phase_list) > 0:
                for phase in phase_list:
                    self.param['gen_size_phase_%i' %phase] = copy(self.param['gen_size'])
            else:
                phase = 0
                self.param['gen_size_phase_%i' %phase] = copy(self.param['gen_size'])

        if len(phase_list_gen_eps) != 0:
            if len(phase_list_gen_eps) != no_phases:
                self.errors['phase_list_gen_eps_1'] = \
                    'Input number of structural phases does not agree with number\n' +\
                    'of gen_size_phase_ keywords given'
            if phase_list_gen_eps != phase_list:
                self.errors['phase_list_gen_eps_2'] = \
                    'The phase numbers given to gen_size_phase does not match those\n' +\
                    'in crystallographic part - structure_phase_X or unit_cell_phase_X.'
            self.param['gen_eps'][0] = 1
        else:
            if len(phase_list) > 0:
                for phase in phase_list:
                    self.param['gen_eps_phase_%i' %phase] = copy(self.param['gen_eps'])
            else:
                phase = 0
                # If strain is not provided and no generation of strains have be asked for
                # "set" all eps to zero.
                if self.param['gen_eps'][0] == 0:
                    self.param['gen_eps'] = [1, 0.0, 0.0, 0.0, 0.0]
                self.param['gen_eps_phase_%i' %phase] = copy(self.param['gen_eps'])
        if self.param['gen_phase'][0] != 0:
            if len(self.param['gen_phase'][1:]) != no_phases*2:
                self.errors['gen_phase'] = 'Missing info for  -  gen_phase'

        # Make sure both sgname and sgno exist
        if len(phase_list_sgno) == 0:
            for phase in phase_list_sgno:
                    self.param['sgno_phase_%i' %phase] = sg.sg(sgname = self.param['sgname_phase_%i' %phase]).no
        if len(phase_list_sgname) == 0:
            for phase in phase_list_sgname:
                    self.param['sgname_phase_%i' %phase] = sg.sg(sgno = self.param['sgno_phase_%i' %phase], cell_choice = self.param['cell_choice_phase_%i' %phase]).name


        # Init no of voxels belonging to phase X if not generated
        if self.param['gen_phase'][0] != 1:
            if len(phase_list) == 0:
                self.param['no_voxels_phase_0'] = self.param['no_voxels']
            else:
                for phase in phase_list:
                    self.param['no_voxels_phase_%i' %phase] = 0
        else:
            for i in range(self.param['no_phases']):
                phase = self.param['gen_phase'][i*2+1]
                no_voxels_phase = int(self.param['gen_phase'][i*2+2])
                self.param['no_voxels_phase_%i' %phase] = no_voxels_phase

        # read U, pos, eps and size for all voxels
        voxel_list_U = []
        voxel_list_pos = []
        voxel_list_eps = []
        voxel_list_size = []
        voxel_list_phase = []
        no_voxels = self.param['no_voxels']

        for item in self.param:
            if '_voxels_' in item:
                if 'U' in item:
                    voxel_list_U.append(eval(item.split('_voxels_')[1]))
                elif 'pos' in item:
                    voxel_list_pos.append(eval(item.split('_voxels_')[1]))
                elif 'eps' in item:
                    voxel_list_eps.append(eval(item.split('_voxels_')[1]))
                elif 'size' in item:
                    voxel_list_size.append(eval(item.split('_voxels_')[1]))
                elif 'phase' in item[:5]:
                    voxel_list_phase.append(eval(item.split('_voxels_')[1]))
                    self.param['no_voxels_phase_%i' %self.param[item]] += 1


        sum_of_voxels = 0
        if self.param['no_phases'] > 1:
            for phase in phase_list:
                sum_of_voxels += self.param['no_voxels_phase_%i' %phase]
            if sum_of_voxels != no_voxels:
                self.errors['no_voxels_2'] = \
                    'Input number of voxels (%i) does not agree ' %no_voxels +\
                    'with number of phase_voxels_ keywords (%i)' %sum_of_voxels
        else:
            self.param['no_voxels_phase_0'] = no_voxels

        # assert that input U, pos, eps size are correct in format
        # (same number of voxels and same specifiers or else not input)
        voxel_list_U.sort()
        voxel_list_pos.sort()
        voxel_list_eps.sort()
        voxel_list_size.sort()
        voxel_list_phase.sort()
        if len(voxel_list_U) != 0 and self.param['gen_U'] == 0:
            if len(voxel_list_U) != no_voxels:
                self.errors['voxel_list_U'] = \
                    'Input number of voxels (%i) does not agree with number\n'  %no_voxels +\
                    ' of U_voxels (%i), check for multiple names' %len(voxel_list_U)
            self.param['voxel_list'] = voxel_list_U
            if len(voxel_list_pos) != 0 and self.param['gen_pos'][0] == 0:
                if voxel_list_U != voxel_list_pos:
                    self.errors['voxel_list_U_pos'] = \
                        'Specified voxel numbers for U_voxels and pos_voxels disagree'
                if len(voxel_list_eps) != 0 and self.param['gen_eps'][0] == 0:
                    if voxel_list_U != voxel_list_eps:
                        self.errors['voxel_list_U_eps'] = \
                            'Specified voxel numbers for U_voxels and eps_voxels disagree'
                if len(voxel_list_size) != 0 and self.param['gen_size'][0] == 0:
                    if voxel_list_U != voxel_list_size:
                        self.errors['voxel_list_U_size'] = \
                            'Specified voxel numbers for U_voxels and size_voxels disagree'
                if len(voxel_list_phase) != 0 and self.param['gen_phase'][0] == 0:
                    if voxel_list_U != voxel_list_phase:
                        self.errors['voxel_list_U_phase'] = \
                            'Specified voxel numbers for U_voxels and phase_voxels disagree'
        else:
            if len(voxel_list_pos) != 0 and self.param['gen_pos'][0] == 0:

                if len(voxel_list_pos) != no_voxels:
                    self.errors['voxel_list_pos_novoxels'] = \
                        'Input number of voxels does not agree with number\n'+\
                        ' of pos_voxels, check for multiple names'
                self.param['voxel_list'] = voxel_list_pos
                if len(voxel_list_eps) != 0 and self.param['gen_eps'][0] == 0:

                    if voxel_list_pos != voxel_list_eps:
                        self.errors['voxel_list_pos_eps'] = \
                            'Specified voxel number for pos_voxels and eps_voxels disagree'
                if len(voxel_list_size) != 0 and self.param['gen_size'][0] == 0:

                    if voxel_list_pos != voxel_list_size:
                        self.errors['voxel_list_pos_size'] = \
                            'Specified voxel number for pos_voxels and size_voxels disagree'
                if len(voxel_list_phase) != 0 and self.param['gen_phase'][0] == 0:

                    if voxel_list_pos != voxel_list_phase:
                        self.errors['voxel_list_pos_phase'] = \
                            'Specified voxel number for pos_voxels and phase_voxels disagree'
            elif len(voxel_list_eps) != 0 and self.param['gen_eps'][0] == 0:

                if len(voxel_list_eps) != no_voxels:
                    self.errors['voxel_list_eps_novoxels'] = \
                        'Input number of voxels does not agree with number'+\
                        ' of eps_voxels, check for multiple names'
                self.param['voxel_list'] = voxel_list_eps
                if len(voxel_list_size) != 0 and self.param['gen_size'][0] == 0:

                    if voxel_list_eps != voxel_list_size:
                        self.errors['voxel_list_eps_size'] = \
                            'Specified voxel number for eps_voxels and size_voxels disagree'
                if len(voxel_list_phase) != 0 and self.param['gen_phase'][0] == 0:

                    if voxel_list_eps != voxel_list_phase:
                        self.errors['voxel_list_eps_phase'] = \
                            'Specified voxel number for eps_voxels and phase_voxels disagree'
            elif len(voxel_list_size) != 0 and self.param['gen_size'][0] == 0:

                if len(voxel_list_size) != no_voxels:
                    self.errors['voxel_list_size_novoxels'] = \
                        'Input number of voxels does not agree with number\n'+\
                        ' of size_voxels, check for multiple names'
                self.param['voxel_list'] = voxel_list_size
                if len(voxel_list_phase) != 0 and self.param['gen_phase'][0] == 0:

                    if voxel_list_size != voxel_list_phase:
                        self.errors['voxel_list_size_phase'] = \
                            'Specified voxel numbers for size_voxels and' +\
                            'phase_voxels disagree'
            elif len(voxel_list_phase) != 0 and self.param['gen_phase'][0] == 0:

                if len(voxel_list_phase) != no_voxels:
                    self.errors['voxel_list_phase_novoxels'] = \
                        'Input number of voxels does not agree with number\n'+\
                        ' of phase_voxels, check for multiple names'
                self.param['voxel_list'] = voxel_list_phase
            else:
                self.param['voxel_list'] = list(range(no_voxels))


        if len(voxel_list_U) == 0 and self.param['gen_U'] == 0:
            self.errors['voxel_list_gen_U'] = \
                'Information on U generations missing'
        if len(voxel_list_pos) == 0 and self.param['gen_pos'][0] == 0:
            self.errors['voxel_list_gen_pos'] = \
                'Information on position generation missing'

        if len(voxel_list_size) == 0 and self.param['gen_size'][0] == 0:
            self.errors['voxel_list_gen_size'] = \
                'Information on voxel size generation missing'

        #If no structure file is given - unit_cel should be
        if len(phase_list) == 0:
            # This is a monophase simulation probably using the "old" keywords
            if self.param['structure_file'] == None:
                if self.param['unit_cell'] == None:
                    self.errors['unit_cell'] = \
                        'Missing input: either structure_file or unit_cell has to be specified'

                # rename keyword
                self.param['unit_cell_phase_0'] = self.param['unit_cell']
                # and delete old one
                del self.param['unit_cell']
                if self.param['sgno'] == None and self.param['sgname'] == None:
                    self.errors['sgno'] = \
                        'Missing input: no space group information, '+\
                        'please input either sgno or sgname'
                if self.param['sgno'] == None:
                    self.param['sgno_phase_0'] = sg.sg(sgname = self.param['sgname']).no
                    self.param['cell_choice_phase_0'] = sg.sg(sgname = self.param['sgname']).cell_choice
                    # rename keyword
                    self.param['sgname_phase_0'] = self.param['sgname']
                    # and delete old one
                    del self.param['sgname']
                else:
                    try:
                        self.param['cell_choice_phase_0'] = sg.sg(sgno = self.param['sgno'],cell_choice=self.param['cell_choice']).cell_choice
                        del self.param['cell_choice']
                    except:
                        self.param['cell_choice_phase_0'] = sg.sg(sgno = self.param['sgno']).cell_choice
                    self.param['sgname_phase_0'] = sg.sg(sgno = self.param['sgno'], cell_choice = self.param['cell_choice_phase_0']).name
                    # rename keyword
                    self.param['sgno_phase_0'] = self.param['sgno']
                    # and delete old one
                    del self.param['sgno']
            else:
                # rename keyword
                self.param['structure_phase_0'] = self.param['structure_file']
                # and delete old one
                del self.param['structure_file']
            phase_list = [0]
        self.param['phase_list'] = phase_list

        # make old inp files work
        if len(voxel_list_phase) == 0 and self.param['no_phases'] == 1:
            self.param['voxel_list_phase_%i' %self.param['phase_list'][0]] = self.param['voxel_list']




        if self.param['sample_xyz'] != None and self.param['sample_cyl'] != None:
            self.errors['sample_dim'] = \
                'Both sample_xyz and sample_cyl are given'


        if self.param['gen_size'][0] != 0:
            for phase in  phase_list:
                phase_key = 'gen_size_phase_%i' %phase

                if self.param[phase_key][1] == 0:
                    self.errors['phase_key_1'] = 'Invalid gen_size command, mean size 0'
                if self.param[phase_key][1] > 0:

                    if self.param[phase_key][2] > self.param[phase_key][1]:
                        self.errors['phase_key_2'] = \
                            'gen_size (phase %i): the minimum voxel size is made larger than the mean voxel size - it should be smaller' %phase
                    if self.param[phase_key][3] < self.param[phase_key][1]:
                        self.errors['phase_key_3'] = \
                            'gen_size (phase %i): the maximum voxel size is made smaller than the mean voxel size - it should be larger' %phase
                    if self.param[phase_key][2] < 0:
                        self.param[phase_key][2] = 0

        #check that the given voxel_size and no_voxels are consistent with sample_vol, adjust max to sample size
        if self.param['sample_xyz'] != None:

            if self.param['sample_xyz'][0] < 0 or \
                    self.param['sample_xyz'][1] < 0 or \
                    self.param['sample_xyz'][2] < 0:
                self.errors['sample_xyz'] = 'Invalid sample_xyz all values should be positive'
                self.param['sample_vol'] = None
            else:
                self.param['sample_vol'] = self.param['sample_xyz'][0]*\
                    self.param['sample_xyz'][1]*\
                    self.param['sample_xyz'][2]
                sample_min_dim = min(self.param['sample_xyz'])
        elif self.param['sample_cyl'] != None:

            if self.param['sample_cyl'][0] <= 0 or \
                    self.param['sample_cyl'][1] <= 0:
                self.errors['sample_cyl'] = \
                    'Invalid sample_cyl <= 0'
                self.param['sample_vol'] = None
            else:
                self.param['sample_vol'] = n.pi*self.param['sample_cyl'][0]*\
                    self.param['sample_cyl'][0]*self.param['sample_cyl'][1]/4.
            sample_min_dim = min(self.param['sample_cyl'])
        else:
            self.param['sample_vol'] = n.inf

        if self.param['sample_vol'] != None and self.param['gen_size'][0] != 0:
            diam_limit = (6*self.param['sample_vol']/\
                         (n.exp(.5)*n.pi*self.param['no_voxels']))**(1/3.)
            mean_diam = 0
            vol = []
            for phase in phase_list:

                weight = self.param['no_voxels_phase_%i' %phase]/self.param['no_voxels']
                vol.append(abs(self.param['gen_size_phase_%i' %phase][1])**3 *\
                               n.pi/6.* self.param['no_voxels_phase_%i' %phase])
                mean_diam += abs(self.param['gen_size_phase_%i' %phase][1])*weight
            for i in range(self.param['no_phases']):
                self.param['vol_frac_phase_%i' %phase_list[i]] = vol[i]/n.sum(vol)



            if mean_diam > diam_limit:
                self.errors['mean_diam'] = \
                    'The sample volume is too small to contain the '+\
                    'specified number of voxels with the given voxel size'
            if len(voxel_list_size) > 0:
                for i in range(len(voxel_list_size)):

                    if self.param['size_voxels_%s' %(self.param['voxel_list'][i])] >= diam_limit:
                        self.errors['size_voxels_%s' %(self.param['voxel_list'][i])] = \
                            'The sample diameter is too small to contain the size of the voxel ' +\
                            'by size_voxels_%s' %(self.param['voxel_list'][i])


        #check that a file name with the odf file is input is odf_type chosen to be 2.
        if self.param['peakshape'][0] == 2:
            if self.param['odf_type'] == 2:
                assert self.param['odf_file'] != None, 'No filename given for ODF'

    def show_errors(self):
        '''
        show_errors() prints out what input was
        incomplete or bad if any input was so.
        '''
        if len(self.errors) > 0:
            print('List of errors and/or inconsistencies found in input: ')
            print('----------------------------------------------------- ')
            no = 0
            for i in self.errors:
                no += 1
                print('Error %3i : ' %no, self.errors[i])
            print('----------------------------------------------------- \n')


    def initialize(self):
        '''
        Initialize the input. This method should be called
        after the input has been checked for errors.
        '''
        # Frame generation
        if self.param['make_image'] != 0:
            if self.param['output'] == None:
                self.param['output'] = '.edf'
            if ('.edf' not in self.param['output']) and \
                    ('.tif' not in self.param['output']) and \
                    ('.tif16bit' not in self.param['output']):
                self.param['output'].append('.edf')

        # Does output directory exist?
        if not os.path.exists(self.param['direc']):
            os.makedirs(self.param['direc'])

	    # Generate FILENAME of frames
        omega_step = self.param['omega_step']
        omega_start  = self.param['omega_start']
        omega_end  = self.param['omega_end']

        modulus = n.abs(omega_end-omega_start)%omega_step
        if  modulus > 1e-9:
            if omega_step-modulus > 1e-9:
                raise ValueError('The omega range does not match an integer number of omega steps')


        omega_sign = self.param['omega_sign']
        start_frame = self.param['start_frame']
        omegalist = omega_sign*n.arange(omega_start,omega_end+omega_step+1e-19,omega_step)
        nframes = int((omega_end-omega_start)/omega_step)

        omegalist.sort()

        #Initialize frameinfo container
        self.frameinfo = []

        if omega_sign > 0:
            filerange = n.arange(start_frame,start_frame+nframes)
        else:
            filerange = n.arange((start_frame-1)+nframes,(start_frame-1),omega_sign)
            # reverse omega_start/omega_end
            self.param['omega_end'] = omega_start*omega_sign
            self.param['omega_start'] = omega_end*omega_sign


        i = 0
        for no in filerange:
            self.frameinfo.append(variables.frameinfo_cont(no))
            self.frameinfo[i].name = '%s/%s%0.4d' \
                %(self.param['direc'],self.param['stem'],filerange[no])
            self.frameinfo[i].omega = omegalist[no];
            self.frameinfo[i].nrefl = 0 # Initialize number of reflections on frame
            self.frameinfo[i].refs = [] # Initialize number of reflections on frame
            i += 1


        if self.param['theta_max'] == None:
            # Find maximum theta for generation of all possible reflections on
            # the detector from the detector specs
            dety_center_mm = self.param['dety_center'] * self.param['y_size']
            detz_center_mm = self.param['detz_center'] * self.param['z_size']
            dety_size_mm = self.param['dety_size'] * self.param['y_size']
            detz_size_mm = self.param['detz_size'] * self.param['z_size']
            c2c = n.zeros((4))
            c2c[0] = (dety_center_mm-dety_size_mm)**2 + (detz_center_mm-detz_size_mm)**2
            c2c[1] = (dety_center_mm-dety_size_mm)**2 + (detz_center_mm-0)**2
            c2c[2] = (dety_center_mm-0)**2 + (detz_center_mm-detz_size_mm)**2
            c2c[3] = (dety_center_mm-0)**2 + (detz_center_mm-0)**2
            c2c_max = n.max(n.sqrt(c2c))
            theta_max = n.arctan(c2c_max/self.param['distance'])/2.0 * 180./n.pi

            self.param['theta_max'] = theta_max


if __name__=='__main__':

    #import check_input
    try:
        filename = sys.argv[1]
    except:
        print('Usage: check_input.py  <input.inp>')
        sys.exit()

    myinput = parse_input(input_file = filename)
    myinput.read()
    myinput.check()
    if myinput.missing == True:
        print('MISSING ITEMS')
    myinput.evaluate()
    print(myinput.param)
