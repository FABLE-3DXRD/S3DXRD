#!/usr/bin/env python
'''
variables.py is a way to save information about voxels and their
reflections. Each voxel will have a voxel_cont() object which
can hold both reflection information and voxel state information

Possibly this is a confusing data structure.
'''

from __future__ import print_function
class frameinfo_cont:
    '''
    This container is not used currently...
    '''
    def __init__(self, frameno):
        self.no = frameno
        self.generic = None

class voxel_cont:
    def __init__(self, U=None):
        '''
        The voxel container contains all information
        about the reflections originating from that voxel
        aswell as the voxel orientation.
        '''
        self.U = U
        self.refl = []

class refarray:
    def __init__(self):
        '''
        The idea is that the self.refl = [] list
        follows the order specified in A_id.
        Such that A_id[key] gives the index of
        the respective quantety.
        '''
        self.voxel_reflections_id = { 'dety'         :0,
                                      'detz'         :1,
                                      'Int'          :2,
                                      'omega'        :3,
                                      'dty'          :4,
                                      'h'            :5,
                                      'k'            :6,
                                      'l'            :7,
                                      'voxel_id'     :8,
                                      'dty_as_index' :9
                                      }
