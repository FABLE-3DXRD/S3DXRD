import numpy as np
from xfab import tools
from ImageD11 import indexing

class FieldConverter(object):

    def __init__(self):
        self.field_keys = \
            [
            'E11','E22','E33','E23','E13','E12',            \
            'a','b','c','alpha','beta','gamma',             \
            'rod1','rod2','rod3','UBI11','UBI12','UBI13',   \
            'UBI21','UBI22','UBI23','UBI31','UBI32','UBI33' \
            ]

    def initiate_field_dict(self, rows, cols):
        field_recons = {}
        for key in self.field_keys:
            field_recons[key] = np.full([rows,cols], np.nan)
        return field_recons

    def add_voxel_to_field(self, voxel, field_recons, row , col, params):

        cell = indexing.ubitocellpars( voxel.ubi )
        field_recons['a'][row, col] = cell[0]
        field_recons['b'][row, col] = cell[1]
        field_recons['c'][row, col] = cell[2]
        field_recons['alpha'][row, col] = cell[3]
        field_recons['beta'][row, col] = cell[4]
        field_recons['gamma'][row, col] = cell[5]

        rod = voxel.Rod
        field_recons['rod1'][row, col] = rod[0]
        field_recons['rod2'][row, col] = rod[1]
        field_recons['rod3'][row, col] = rod[2]

        ubi = np.ravel(voxel.ubi,order='C')
        field_recons['UBI11'][row, col] = ubi[0]
        field_recons['UBI12'][row, col] = ubi[1]
        field_recons['UBI13'][row, col] = ubi[2]
        field_recons['UBI21'][row, col] = ubi[3]
        field_recons['UBI22'][row, col] = ubi[4]
        field_recons['UBI23'][row, col] = ubi[5]
        field_recons['UBI31'][row, col] = ubi[6]
        field_recons['UBI32'][row, col] = ubi[7]
        field_recons['UBI33'][row, col] = ubi[8]

        U, eps_cry = tools.ubi_to_u_and_eps(voxel.ubi, self.extract_cell(params))
        eps_cry_33 = np.array([[eps_cry[0],eps_cry[1],eps_cry[2]],
                                [eps_cry[1],eps_cry[3],eps_cry[4]],
                                [eps_cry[2],eps_cry[4],eps_cry[5]]])
        sample_strain = np.dot(U,np.dot(eps_cry_33,np.transpose(U)))

        field_recons['E11'][row, col] = sample_strain[0,0]
        field_recons['E12'][row, col] = sample_strain[0,1]
        field_recons['E13'][row, col] = sample_strain[0,2]
        field_recons['E22'][row, col] = sample_strain[1,1]
        field_recons['E23'][row, col] = sample_strain[1,2]
        field_recons['E33'][row, col] = sample_strain[2,2]
    
    def extract_cell(self, params):
        a = params.parameters['cell__a']
        b = params.parameters['cell__b']
        c = params.parameters['cell__c']
        alpha = params.parameters['cell_alpha']
        beta = params.parameters['cell_beta']
        gamma = params.parameters['cell_gamma']
        return np.array([a,b,c,alpha,beta,gamma])