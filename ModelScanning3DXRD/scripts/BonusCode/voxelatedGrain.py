'''
Produce input files for ModelScanning3DXRD

This is a little bit faster than doing so by hand (or a lot faster ;) ).
Basically we have a sample or Slice object that we fill up with voxel objects which all have
a series of attributes that properly represent their state.
We then simply call the SliceFactory addSliceToAscii() to parse the
python objects to a input file which is readable for ModelScanning3DXRD.

If you don't like this script, it should be straightforward to build your own.
It's all just a mather of generating input files in a smooth manner.
Sometimes it is faster to write your own parser rather than trying to understand the
logics of somebody elses.
'''

import numpy as n
from modelscanning3DXRD import generate_voxels as gg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from xfab import tools
from xfab import symmetry
import random
import sys


BASE=r'''
direc 'Sn'
stem 'Sn_ssch'
make_image 0
output  'merged.flt' 'delta.flt' 'delta.gve' '.par' '.ubi'
#output  'merged.flt' '.par' '.ubi'
structure_phase_0 '/home/axel/workspace/scanning-3dxrd-simulations/ModelScanning3DXRD/scripts/simul/Sn.cif'
y_size 0.0500
z_size 0.0500
dety_size 2048
detz_size 2048
distance 162.888383321
tilt_x -0.00260053595757
tilt_y -0.00366923010272
tilt_z 0.00463564130438
o11 1
o12 0
o21 0
o22 -1
noise 0
intensity_const 1
lorentz_apply 1
beampol_apply 0
peakshape 0
wavelength 0.21878
beamflux 1e-12
beampol_factor 1
beampol_direct 0
dety_center 998.798998
detz_center 1041.55049636
omega_start 0
omega_end 180
omega_step 1.0
omega_sign 1
theta_min 0
theta_max 8
wedge -0.0142439915706
#spatial '/home/axel/workspace/fable/ModelScanning3DXRD/scripts/simul/frelon4m.spline'
'''

class Voxel(object):
    '''
    A Voxel is acosiated with a single crystalographic orientation in space
    defined by the orientation matrix U.
    '''
    def __init__(self, voxelId, grainNo, size, x,y,z,U=None,strain=None):

        self.voxelId = voxelId
        self.grainNo = grainNo
        self.size = size
        self.x = x
        self.y = y
        self.z = z
        self.U = U
        self.strain = strain

class Slice(object):
    '''
    A Slice is constructed by an arbitrary number of Voxels.
    '''
    def __init__(self, voxels, shape, resolution=1):
        self.voxels = voxels
        self.resolution = resolution
        self.shape = shape

    def addVoxel(self, voxel):
        self.voxels.append(voxel)

    def get_dim(self,axis):
        voxels_coord = n.zeros((self.noVoxels(),1))
        if axis=="x":
            voxels_coord = [v.x for v in self.voxels]
        elif axis=="y":
            voxels_coord = [v.y for v in self.voxels]
        elif axis=="z":
            return
            #needs fix as +-d has been introduced
            #voxels_coord = [v.y for v in self.voxels]
        else:
            print("axis need to be set to x, y or z")
            return
        if self.resolution is not 1:
            d = self.voxels[0].size/2.
        else:
            d=0
        return [n.min(voxels_coord) + d, n.max(voxels_coord) - d]

    def noVoxels(self):
        return len(self.voxels)

    def getVoxelGrainMap(self):
        '''produce a: voxel <=> grain ,dictionary (passed it to PXS via .inp file)
        we need to track which voxels belong to which grains in order to produce
        average orientations for the grains to be inputed to makemap/toyota scripts
        '''
        voxelgrainMap = {}
        for v in self.voxels:
            voxelgrainMap[v.voxelId]=v.grainNo
        return str(voxelgrainMap).replace(" ","")

    def boxWrappSides(self):
        Xrange = self.get_dim("x")
        Yrange = self.get_dim("y")
        #Zrange = self.get_dim("z")
        Zrange = [-self.voxels[0].size/2., self.voxels[0].size/2.]
        d = 2*self.voxels[0].size #make sure there is space enough (hence the 2 instead of 1)
        return (Xrange[1]-Xrange[0]+d, Yrange[1]-Yrange[0]+d, Zrange[1]-Zrange[0]+d)

    # Rotate the entire slice (all voxels) 90 degrees counterclockwise
    def rotateSlice90(self):
        c, s = n.cos(n.pi/2.), n.sin(n.pi/2.)
        Rz = n.array([[c,-s,0],[s,c,0],[0,0,1]])
        for v in self.voxels:
            loc_vect = n.array([[v.x],
                                [v.y],
                                [v.z]])
            rot_vect = Rz.dot(loc_vect)
            v.x = rot_vect[0]
            v.y = rot_vect[1]
            v.z = rot_vect[2]


class SliceFactory(object):
    # The SliceFactory manufactures Slices, these methods are helpful
    # for constructing random Slices etc..
    def __init__(self):
        pass

    # x, y, z are the integer sides of the romboid given in voxel counts.
    # The romboid will be centered at the omega rotation axis
    def randomoOrientRomboidSlice(self, x, y, z, voxelSize,crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        U = n.array([[1,0,0],[0,1,0],[0,0,1]])
        #U = self.generate_U(crystal_system)
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr,U)
                    Id=Id+1
                    #U = gg.generate_U(1,sgi)
                    #strain = gg.generate_eps
                    a_slice.addVoxel(voxel)
        return a_slice

    def TwoGrainSlice(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        U1 = n.array([[-0.158282, -0.986955, 0.029458],
                        [-0.929214, 0.158978, 0.333597],
                        [-0.333929, 0.025430, -0.942255]])
        # U2 = n.array([[0.861031141, -0.350285857, 0.368680340],
        #               [0.504660772, 0.499018829, -0.704484005],
        #               [0.062792352, 0.792641171, 0.606446283]])
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])

        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if (i<=1 and j<=1):
                        voxel = Voxel(Id,1, voxelSize, xCurr, yCurr, zCurr, U1)
                    else:
                        voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def offsetSquare(self, x, y, z, voxelSize, crystal_system):

        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize

        dx=dy=voxelSize*110
        xOrig = xOrig - dx
        yOrig = yOrig - dy

        a_slice = Slice([],shape="square")
        U1 = n.array([[-0.158282, -0.986955, 0.029458],
                        [-0.929214, 0.158978, 0.333597],
                        [-0.333929, 0.025430, -0.942255]])
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def gridOfgrains(self, x, y, z, voxelSize, crystal_system):

        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize

        #make a line grid of grains with size equal to y*2
        grain_size = voxelSize*x

        U = n.array([[-0.158282, -0.986955, 0.029458],
                        [-0.929214, 0.158978, 0.333597],
                        [-0.333929, 0.025430, -0.942255]])
        Id=0
        a_slice = Slice([],shape="square")
        s = SliceFactory()
        grain_nbr = 0
        #for q in range(0,x,1):
        for p in range(0,5,1):
            grain_nbr += 1
            #dx = -2*grain_size + 2*q*grain_size
            dx = 0
            dy = p*grain_size + p*1*voxelSize
            dz = 0
            U = s.randRot(U,n.pi/6.,n.pi/6)
            c_vox = int(n.floor(x/2.))
            for i in range(0,x,1):
                xCurr = xOrig + i*voxelSize + dx
                for j in range(0,y,1):
                    yCurr = yOrig + j*voxelSize + dy
                    for k in range(0,z,1):
                        zCurr = zOrig + k*voxelSize + dz
                        voxel = Voxel(Id,grain_nbr, voxelSize, xCurr, yCurr, zCurr, U)
                        print(n.round(xCurr*1000), n.round(yCurr*1000), n.round(zCurr*1000))
                        Id=Id+1
                        a_slice.addVoxel(voxel)
        return a_slice

    def localStrain(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        #U0 = n.array([[-0.158282, -0.986955, 0.029458],
                    #   [-0.929214, 0.158978, 0.333597],
                    #   [-0.333929, 0.025430, -0.942255]])
        # U = n.array([[0.861031141, -0.350285857, 0.368680340],
                    #  [0.504660772, 0.499018829, -0.704484005],
                    #  [0.062792352, 0.792641171, 0.606446283]])
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6)) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        eps[0,0]=0.01
        epsZero = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if i>1 and j>1 and i<4 and j<4:
                        #strain in central voxel only
                        voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps)
                    else:
                        voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=epsZero)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice



    def roundGrain(self, x, y, z, voxelSize, crystal_system):

        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6)) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        epsZero = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))


        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize

                    # spread = 0.5
                    # U = self.randRot(Ualign,0,spread)

                    # U = Ualign.copy()
                    # U = ((-0.2)+(2*0.2*float(j/(y-1.0))))/100.0
                    # 1 degree rotation around each axis in x gradient
                    phi1 = n.radians( 0.25*float(i/(y-1.0)) )
                    PHI = n.radians( 0.25*float(i/(y-1.0))  )
                    phi2 = n.radians( 0.25*float(i/(y-1.0)) )
                    U = tools.euler_to_u( phi1, PHI, phi2 )

                    print( float(j/(y-1.0)) )
                    print(U)
                    print()

                    if n.sqrt( xCurr*xCurr+yCurr*yCurr ) < abs(xOrig):
                        
                        eps[0,0] = ((-0.2)+(2*0.2*float(j/(y-1.0))))/100.0

                        # We need to transform to still have a gradient in lab frame..
                        eps33 = n.zeros((3,3))
                        eps33[0,0] = eps[0,0]
                        eps33[0,1] = eps[0,1]
                        eps33[0,2] = eps[0,2]
                        eps33[1,0] = eps33[0,1]
                        eps33[1,1] = eps[0,3]
                        eps33[2,1] = eps[0,4]
                        eps33[2,2] = eps[0,5]

                        eps33[1,2] = eps33[2,1]
                        eps33[2,0] = eps33[0,2]

                        epsCrystal = n.dot( n.dot( n.transpose(U), eps33 ), U )

                        # eps11 eps12 eps13 eps22 eps23 eps33 (X is an integer number)
                        epsC = n.zeros((1,6))
                        epsC[0,0] = epsCrystal[0,0]
                        epsC[0,1] = epsCrystal[0,1]
                        epsC[0,2] = epsCrystal[0,2]
                        epsC[0,3] = epsCrystal[1,1]
                        epsC[0,4] = epsCrystal[1,2]
                        epsC[0,5] = epsCrystal[2,2]

                        voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, U.copy(), strain=eps.copy())
                        Id=Id+1
                        a_slice.addVoxel(voxel)
        
        return a_slice

    def x_rot(self, U, ang):
        c = n.cos(n.radians(ang))
        s = n.cos(n.radians(ang))
        return n.dot( n.array([[1,0,0],[0,c,-s],[0,s,c]]), U )

    def y_rot(self, U, ang):
        c = n.cos(n.radians(ang))
        s = n.cos(n.radians(ang))
        return n.dot( n.array([[c,0,s],[0,1,0],[-s,0,c]]), U )

    def z_rot(self, U, ang):
        c = n.cos(n.radians(ang))
        s = n.cos(n.radians(ang))
        return n.dot( n.array([[c,-s,0],[s,c,0],[0,0,1]]), U )




    def testFunction(self, x, y, z, voxelSize, crystal_system):
        x = 2*x
        xOrig = -0.5*x*voxelSize+0.5*voxelSize - 4*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize + 8*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6)) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        epsZero = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))

        forbidden = n.array([[0,0],[0,1],[1,0]])
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize

                    skip=False
                    for a,b in forbidden:
                        if i==a and j==b:
                            skip=True
                    if skip: continue
                    #eps_grains__X_ eps11 eps12 eps13 eps22 eps23 eps33 (X is an integer number)

                    eps[0,0] = ((-0.2)+(2*0.2*float(i/(x-1.0))))/100.0 #eps_xx
                    eps[0,1] = ((-0.15)+(2*0.15*float(i/(x-1.0))))/100.0 #eps_xy
                    eps[0,2] = ((-0.17)+(2*0.17*float(i/(x-1.0))))/100.0 #eps_xz
                    eps[0,3] = ((-0.3)+(2*0.3*float(i/(x-1.0))))/100.0 #eps_yy
                    eps[0,4] = ((-0.35)+(2*0.35*float(i/(x-1.0))))/100.0 #eps_yz
                    eps[0,5] = ((-0.1)+(2*0.1*float(j/(y-1.0))))/100.0 #eps_zz
                    #print(eps[0,5])
                    # grad_neg_start=0.2
                    # eps[0,0] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_xx
                    # eps[0,1] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_xy
                    # eps[0,2] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_xz
                    # eps[0,3] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_yy
                    # eps[0,4] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_yz
                    # eps[0,5] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_zz

                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientYstrainXX(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-2.0)+(4.0*float(j/(y-1.0))))/100.0
                    #print(eps[0,0])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def ContGradientYstrainXX(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        grad_neg_start = 0.1
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-grad_neg_start)+(2*grad_neg_start*float(j/(y-1.0))))/100.0
                    #print(eps[0,0])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def ContGradientXstrainXX(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        grad_neg_start = 0.1
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    #print(eps[0,0])
                    #eps[0,0]=0
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientXstrainXX(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-2.0)+(4.0*float(i/(x-1.0))))/100.0
                    #print(eps[0,0])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientYstrainYY(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,3] = ((-2.0)+(4.0*float(j/(y-1.0))))/100.0
                    #print(eps[0,3])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientXstrainYY(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,3] = ((-2.0)+(4.0*float(i/(x-1.0))))/100.0
                    #print(eps[0,3])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice


    def GradientYstrainXXGrainrotation(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")

        #rotate 45 degree positive
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        c, s = n.cos(n.pi/4.), n.sin(n.pi/4.)
        Rz = n.array([[c,-s,0],[s,c,0],[0,0,1]])
        Urot = n.dot(Rz,Ualign)

        eps = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-2.0)+(4.0*float(j/(y-1.0))))/100.0
                    #print(eps[0,0])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Urot, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientXstrainXXGrainrotation(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")

        #rotate 45 degree positive
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        c, s = n.cos(n.pi/4.), n.sin(n.pi/4.)
        Rz = n.array([[c,-s,0],[s,c,0],[0,0,1]])
        Urot = n.dot(Rz,Ualign)

        eps = n.zeros((1,6))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-2.0)+(4.0*float(i/(x-1.0))))/100.0
                    #print(eps[0,0])
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Urot, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientYstrainXXSamplerotation(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")

        #rotate 45 degree positive
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        c, s = n.cos(n.pi/4.), n.sin(n.pi/4.)
        Rz = n.array([[c,-s,0],[s,c,0],[0,0,1]])
        Urot = n.dot(Rz,Ualign)

        epsSamp = n.zeros((3,3))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    #Sample desired strain
                    epsSamp[0,0] = (1/n.sqrt(2.))*((-2.0)+(4.0*float(j/(y-1.0))))/100.0

                    #convertion to grain strain
                    epsGrain = n.dot(n.dot(n.linalg.inv(Urot),epsSamp),n.linalg.inv(n.transpose(Urot)))
                    eps = n.array([[epsGrain[0,0],epsGrain[0,1],epsGrain[0,2],epsGrain[1,1],epsGrain[1,2],epsGrain[2,2]]])
                    #print(eps)

                    #print(n.sqrt(eps[0,0]*eps[0,0]+eps[0,3]*eps[0,3]))
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Urot, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def GradientXstrainXXSamplerotation(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")

        #rotate 45 degree positive
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        c, s = n.cos(n.pi/4.), n.sin(n.pi/4.)
        Rz = n.array([[c,-s,0],[s,c,0],[0,0,1]])
        Urot = n.dot(Rz,Ualign)

        epsSamp = n.zeros((3,3))
        Id = 0
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    #Sample desired strain
                    epsSamp[0,0] = ((-2.0)+(4.0*float(i/(x-1.0))))/100.0

                    #convertion to grain strain
                    epsGrain = n.dot(n.dot(n.linalg.inv(Urot),epsSamp),n.linalg.inv(n.transpose(Urot)))

                    eps = n.array([[epsGrain[0,0],epsGrain[0,1],epsGrain[0,2],epsGrain[1,1],epsGrain[1,2],epsGrain[2,2]]])
                    #print(eps)
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, Urot, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def unstrainedCircle(self, voxelSize, R):

        a_slice = Slice([],shape="circle")
        U = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6)) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        epsZero = n.zeros((1,6))
        Id=0
        X = int(R) # R is the radius
        for x in range(-X,X+1):
            Y = int((R*R-x*x)**0.5) # bound for y given x
            for y in range(-Y,Y+1):
                xCurr = x*voxelSize
                yCurr = y*voxelSize
                zCurr = 0
                voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr, U, strain=epsZero)
                Id=Id+1
                a_slice.addVoxel(voxel)

        #Sanity check, we reqire pixel at origin.
        #---------------------------
        nonCentered=True
        for v in a_slice.voxels:
            if v.x==0 and v.y==0:
                nonCentered = False
        if nonCentered:
            raise KeyboardInterrupt
        #---------------------------

        return a_slice


    def randomoOrientRomboidSliceWithStrain(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        U = self.generate_U(crystal_system)
        eps = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    voxel = Voxel(Id, 0, voxelSize, xCurr, yCurr, zCurr, U, strain=eps)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def randomoOrientRomboidSliceWithStrainGradient(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        U = self.generate_U(crystal_system)
        eps = gg.generate_eps(1,[0, 0.02, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        # epsZero = n.zeros((1,6))
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    # if (i>=8 and i<=10) and (j>=8 and j<=10):
                    #     voxel = Voxel(Id, voxelSize, xCurr, yCurr, zCurr, U, strain=eps)
                    # else:
                    #     voxel = Voxel(Id, voxelSize, xCurr, yCurr, zCurr, U, strain=epsZero)
                    voxel = Voxel(Id, 0, voxelSize, xCurr, yCurr, zCurr, U, strain=(eps*(j/float(y))))
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def randomoOrientRomboidSliceWithMosaicSpread(self, x, y, z, voxelSize,crystal_system,spread=1):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        U = self.generate_U(crystal_system)
        RotatedUs=[]
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    Ucurr = self.randRot(U,0,spread)
                    RotatedUs.append(Ucurr)
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr,Ucurr)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice



    def checkerOrientRomboidSlice(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        U1 = self.generate_U(crystal_system)
        U2 = self.generate_U(crystal_system)
        a_slice = Slice([],shape="square")
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if (i%2==0 and j%2==1) or (i%2==1 and j%2==0) :
                        U = U1
                    else:
                        U = U2
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, zCurr,U)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def TwoOrientRomboidSlice(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        U1 = n.array([[1,0,0],[0,1,0],[0,0,1]])
        #U2 = self.generate_U(crystal_system)
        thetaZ = n.radians(20)
        thetaX = n.radians(20)
        cZ, sZ = n.cos(thetaZ), n.sin(thetaZ)
        cX, sX = n.cos(thetaX), n.sin(thetaX)
        Rz = n.array([[cZ,-sZ,0],[sZ,cZ,0],[0,0,1]])
        Rx = n.array([[1,0,0],[0,cX,-sX],[0,sX,cX]])
        Urotated = Rz.dot(Rx.dot(U1))
        U2 = Urotated
        grainNo=0
        a_slice = Slice([],shape="square")
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if j>=y/2.:
                        U = U1
                        grainNo=0
                    else:
                        U = U2
                        grainNo=1
                    voxel = Voxel(Id,grainNo, voxelSize, xCurr, yCurr, zCurr,U)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def ThreeOrientRomboidSlice(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        U1 = self.generate_U(crystal_system)
        U2 = self.generate_U(crystal_system)
        U3 = self.generate_U(crystal_system)
        grainNo=0
        a_slice = Slice([],shape="square")
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if j>=y/2.:
                        U = U1
                        print("U1 at: ", xCurr, yCurr)
                        grainNo=0
                    elif i>=x/2.:
                        print("U2 at: ", xCurr, yCurr)
                        U = U2
                        grainNo=1
                    else:
                        print("U3 at: ", xCurr, yCurr)
                        U = U3
                        grainNo=2
                    voxel = Voxel(Id,grainNo, voxelSize, xCurr, yCurr, zCurr,U)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice

    def ThreeOrientRomboidSliceWithStrain(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        U1 = self.generate_U(crystal_system)
        U2 = self.generate_U(crystal_system)
        U3 = self.generate_U(crystal_system)
        grainNo=0
        eps1 = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        eps2 = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        eps3 = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        a_slice = Slice([],shape="square")
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if j>=y/2.:
                        U = U1
                        eps = eps1
                        grainNo=0
                    elif i>=x/2.:
                        U = U2
                        eps = eps2
                        grainNo=1
                    else:
                        U = U3
                        eps = eps3
                        grainNo=2
                    voxel = Voxel(Id,grainNo, voxelSize, xCurr, yCurr, zCurr,U,strain=eps)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice


    def OneGrain2Strains(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        U = n.eye(3,3)
        grainNo=0
        eps1 = n.zeros((1,6))
        eps1[0,0]=0.001
        eps2 = n.zeros((1,6))
        eps2[0,0]=-0.001
        a_slice = Slice([],shape="square")
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if j>=y/2.:
                        eps = eps1
                    else:
                        eps = eps2
                    voxel = Voxel(Id,grainNo, voxelSize, xCurr, yCurr, zCurr,U,strain=eps)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice




    def Aligned_xxStrainGrads(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize + 5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize + 10*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        grad_neg_start = 0.2
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, 0, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice


    def Aligned_NoStrainGrads(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize #- 10*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize #- 5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        grad_neg_start = 0.2
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, 0, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice



    def Aligned_StrainGrads(self, x, y, z, voxelSize, crystal_system):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        grad_neg_start = 0.2
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize

                    #eps_grains__X_ eps11 eps12 eps13 eps22 eps23 eps33 (X is an integer number)

                    eps[0,0] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_xx
                    eps[0,1] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_xy
                    eps[0,2] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_xz
                    eps[0,3] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_yy
                    eps[0,4] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_yz
                    eps[0,5] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0 #eps_zz

                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, 0, Ualign, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice


    def Aligned_StrainGrads_MosaicSpread(self, x, y, z, voxelSize, crystal_system):

        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        grad_neg_start = 0.2
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    eps[0,0] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    eps[0,1] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    eps[0,2] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    eps[0,3] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    eps[0,4] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    eps[0,5] = ((-grad_neg_start)+(2*grad_neg_start*float(i/(x-1.0))))/100.0
                    spread = 0.05
                    U = self.randRot(Ualign,0,spread)

                    # We need to transform to still have a gradient in lab frame..
                    eps33 = n.zeros((3,3))
                    eps33[0,0] = eps[0,0]
                    eps33[0,1] = eps[0,1]
                    eps33[0,2] = eps[0,2]
                    eps33[1,0] = eps33[0,1]
                    eps33[1,1] = eps[0,3]
                    eps33[2,1] = eps[0,4]
                    eps33[2,2] = eps[0,5]

                    eps33[1,2] = eps33[2,1]
                    eps33[2,0] = eps33[0,2]

                    epsCrystal = n.dot( n.dot( n.transpose(U), eps33 ), U )

                    # eps11 eps12 eps13 eps22 eps23 eps33 (X is an integer number)
                    epsC = n.zeros((1,6))
                    epsC[0,0] = epsCrystal[0,0]
                    epsC[0,1] = epsCrystal[0,1]
                    epsC[0,2] = epsCrystal[0,2]
                    epsC[0,3] = epsCrystal[1,1]
                    epsC[0,4] = epsCrystal[1,2]
                    epsC[0,5] = epsCrystal[2,2]
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, 0, U, strain=epsC.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice




    def Aligned_MosaicSpread(self, x, y, z, voxelSize, crystal_system):

        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        a_slice = Slice([],shape="square")
        Ualign = n.array([[1,0,0],[0,1,0],[0,0,1]])
        eps = n.zeros((1,6))
        Id = 0
        grad_neg_start = 0.1
        c_vox = int(n.floor(x/2.))
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    spread = 0.05
                    U = self.randRot(Ualign,0,spread)
                    voxel = Voxel(Id,0, voxelSize, xCurr, yCurr, 0, U, strain=eps.copy())
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice


    def ThreeOrientRomboidSliceWithMosaicSpreadAndStrainGradients(self, x, y, z, voxelSize, crystal_system,spread=1):
        xOrig = -0.5*x*voxelSize+0.5*voxelSize
        yOrig = -0.5*y*voxelSize+0.5*voxelSize
        zOrig = -0.5*z*voxelSize+0.5*voxelSize
        U1 = self.generate_U(crystal_system)
        U2 = self.generate_U(crystal_system)
        U3 = self.generate_U(crystal_system)
        grainNo=0
        eps1 = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        eps2 = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        eps3 = gg.generate_eps(1,[0, 0.01, 0, 0]) #[mean_diag, spread_diag, mean_ofdiag, spread_ofdiag]
        a_slice = Slice([],shape="square")
        Id = 0
        for i in range(0,x,1):
            xCurr = xOrig + i*voxelSize
            for j in range(0,y,1):
                yCurr = yOrig + j*voxelSize
                for k in range(0,z,1):
                    zCurr = zOrig + k*voxelSize
                    if j>=y/2.:
                        U = self.randRot(U1,0,spread)
                        eps = eps1*(j/float(y))
                        grainNo=0
                    elif i>=x/2.:
                        U = self.randRot(U2,0,spread)
                        eps = eps2*(j/float(y))
                        grainNo=1
                    else:
                        U = self.randRot(U3,0,spread)
                        eps = eps3*(j/float(y))
                        grainNo=2
                    voxel = Voxel(Id,grainNo, voxelSize, xCurr, yCurr, zCurr,U,strain=eps)
                    Id=Id+1
                    a_slice.addVoxel(voxel)
        return a_slice


# seeems broken?
    # def generate_U(self,crystal_system):
	#     # copy paste
    #     # crystal_system can be one of the following values
    #     # 1: Triclinic
    #     # 2: Monoclinic
    #     # 3: Orthorhombic
    #     # 4: Tetragonal
    #     # 5: Trigonal
    #     # 6: Hexagonal
    #     # 7: Cubic
    #     U = n.zeros((3,3))
    #     Urot = n.zeros((3,3))
    #     tilt_z = n.random.rand()*2*n.pi
    #     tilt_y = n.arcsin(n.random.rand())
    #     tilt_x  = n.pi*(2*n.random.rand()*n.pi-1)
    #     U = tools.detect_tilt(tilt_x, tilt_y, tilt_z)
    #     t = 0
    #     Ut = U.copy()
    #     rot = symmetry.rotations(crystal_system)
    #     for j in range(len(rot)):
    #         Urot = n.dot(U,rot[j])
    #         trace = Urot.trace()
    #         if trace > t:
    #             t = trace
    #             Ut = Urot
    #     U = Ut
    #     return U



    def randRot(self,U,mu,sigma):
        randZ = n.random.normal(mu,sigma,1)
        randX = n.random.normal(mu,sigma,1)
        thetaZ = n.radians(randZ)
        thetaX = n.radians(randX)
        cZ, sZ = n.cos(thetaZ), n.sin(thetaZ)
        cX, sX = n.cos(thetaX), n.sin(thetaX)
        Rz = n.array([[cZ,-sZ,0],[sZ,cZ,0],[0,0,1]])
        Rx = n.array([[1,0,0],[0,cX,-sX],[0,sX,cX]])
        Urotated = Rz.dot(Rx.dot(U))
        for colonvector in [Urotated[:,0],Urotated[:,1],Urotated[:,2]]:
            if n.abs(1.0-n.linalg.norm(colonvector))>1e-5:
                print("The rotated orientation matrix: ", Urotated)
                print("is not normalised, something went wrong..")
                raise KeyboardInterrupt
        return Urotated

    def addSliceToAscii(self, a_slice, ascii=None):

        if ascii==None:
            fname = "inputfile"
            mode = "w+"
        else:
            fname = ascii
            mode = "r+"

        with open(fname,mode) as f:
            f.truncate(0)
            f.write(BASE)
            beam_width = (a_slice.voxels[0].size*n.sqrt(a_slice.resolution))
            f.write("\nno_voxels %d\n" % a_slice.noVoxels())
            f.write("voxel_grain_map %s\n" % a_slice.getVoxelGrainMap())
            f.write("gen_size 1 %.9f %.9f %.9f\n" % ((-2.0*a_slice.voxels[0].size*((3.0/(4.0*3.1415))**(1/3.0))), 0., 100.))
            f.write("sample_xyz %.9f %.9f %.9f\n" % a_slice.boxWrappSides())
            f.write("beam_width %.9f\n" % beam_width )

            if a_slice.voxels[0].U is not None:
                f.write("gen_U %d\n" % 0)
            else:
                f.write("gen_U %d\n" % 1)

            #If the voxels has no strains prescribed, set all strain tensors to zero
            if a_slice.voxels[0].strain is None:
                f.write("gen_eps %d %d %d %d %d\n" % (1,0,0,0,0))

            for voxel in a_slice.voxels:
                if voxel.x!=None:
                    f.write("pos_voxels_%d %.9f %.9f %.9f\n" % (voxel.voxelId, voxel.x, voxel.y, voxel.z))
                if voxel.U is not None:
                    U=voxel.U
                    f.write("U_voxels_%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (voxel.voxelId, U[0,0], U[0,1], U[0,2], U[1,0], U[1,1], U[1,2], U[2,0], U[2,1], U[2,2]))
                if voxel.strain is not None:
                    e = voxel.strain
                    f.write("eps_voxels_%d %.9f %.9f %.9f %.9f %.9f %.9f\n" % (voxel.voxelId, e[0,0], e[0,1], e[0,2], e[0,3], e[0,4], e[0,5]))


#to make the script easy to call from python script
class profiler():

    def __init__(self):
        pass

    def makeAsample(self,no_voxels,voxel_size):
        slice_factory = SliceFactory()
        voxel_size = voxel_size*(10**-3)
        a_slice = slice_factory.GradientYstrainXX(no_voxels,no_voxels,1,voxel_size,4)
        slice_factory.addSliceToAscii(a_slice,"Sn.inp")

if __name__ == "__main__":
    slice_factory = SliceFactory()
    no_voxels = int(sys.argv[1])
    #TODO: implement even numbers of voxels
    # if no_voxels%2==0:
    #     print("slices with center not at a voxel center is not yet implemented")
    #     import sys
    #     sys.exit()

    voxel_size =float(sys.argv[2])*(10**-3) #give this in microns and it is rescaled to mm
    #a_slice = slice_factory.randomoOrientRomboidSlice(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.randomoOrientRomboidSliceWithStrain(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.randomoOrientRomboidSliceWithStrainGradient(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.checkerOrientRomboidSlice(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.TwoOrientRomboidSlice(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.ThreeOrientRomboidSlice(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.ThreeOrientRomboidSliceWithStrain(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.HighResTwoOrientRomboidSlice(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.randomoOrientRomboidSliceWithMosaicSpread(no_voxels,no_voxels,1,voxel_size,4,spread=(1/3.))
    #a_slice = slice_factory.ThreeOrientRomboidSliceWithMosaicSpreadAndStrainGradients(no_voxels,no_voxels,1,voxel_size,4,spread=(1/3.))

    #Simultaneous fitting
    #a_slice = slice_factory.Aligned_StrainGrads_MosaicSpread(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.Aligned_MosaicSpread(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.Aligned_StrainGrads(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.Aligned_NoStrainGrads(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.Aligned_xxStrainGrads(no_voxels,no_voxels,1,voxel_size,4)

    #a_slice= slice_factory.localStrain(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.OneGrain2Strains(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.ContGradientXstrainXX(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.ContGradientYstrainXX(no_voxels,no_voxels,1,voxel_size,4)

    #Investigate CMS corrections
    #a_slice = slice_factory.TwoGrainSlice(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.offsetSquare(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.gridOfgrains(no_voxels,no_voxels,1,voxel_size,4)

    #Put tests in this function
    # a_slice = slice_factory.testFunction(no_voxels,no_voxels,1,voxel_size,4)

    #Round "contact strain grain"
    a_slice = slice_factory.roundGrain(no_voxels,no_voxels,1,voxel_size,4)


    #Gradient investigations
    #------------------------------------------------------------------------------
    # strain in xx direction
    #a_slice = slice_factory.GradientYstrainXX(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.GradientXstrainXX(no_voxels,no_voxels,1,voxel_size,4)

    # strain in yy direction
    #a_slice = slice_factory.GradientXstrainYY(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.GradientYstrainYY(no_voxels,no_voxels,1,voxel_size,4)

    # strain in xx crystal system and rotation 45 degrees The tsrain is 45 to the gradient
    #a_slice = slice_factory.GradientYstrainXXGrainrotation(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.GradientXstrainXXGrainrotation(no_voxels,no_voxels,1,voxel_size,4)

    # strain in xx sample system and rotation 45 degrees The strain is 0 or 90 to the gradient
    #a_slice = slice_factory.GradientYstrainXXSamplerotation(no_voxels,no_voxels,1,voxel_size,4)
    #a_slice = slice_factory.GradientXstrainXXSamplerotation(no_voxels,no_voxels,1,voxel_size,4)
    #------------------------------------------------------------------------------


    #a_slice = slice_factory.unstrainedCircle(voxel_size, no_voxels) #Use no_voxels input as radius


    slice_factory.addSliceToAscii(a_slice,"Sn.inp")
    print("Created ",len(a_slice.voxels)," voxels")
    if str(sys.argv[3])=='plot':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for voxel in a_slice.voxels:
            ax.scatter(voxel.x, voxel.y, voxel.z)
        plt.show()

