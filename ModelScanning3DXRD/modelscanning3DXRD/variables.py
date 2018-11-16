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
        self.A_id = { 'voxel_id'       :0,
                      'ref_id'         :1,
                      'spot_id'        :2,
                      'h'              :3,
                      'k'              :4,
                      'l'              :5,
                      'tth'            :6,
                      'omega'          :7,
                      'eta'            :8,
                      'dety'           :9,
                      'detz'           :10,
                      'detyd'          :11,
                      'detzd'          :12,
                      'gv1'            :13,
                      'gv2'            :14,
                      'gv3'            :15,
                      'L'              :16,
                      'P'              :17,
                      'F2'    	       :18,
                      'Int'            :19,
                      'overlaps'       :20,
                      'dty'            :21
                      }
