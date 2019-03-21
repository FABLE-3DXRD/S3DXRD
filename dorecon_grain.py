
from __future__ import print_function, division

from ImageD11.columnfile import columnfile
from ImageD11 import grain
import numpy as np
import sys, time, os.path
import pylab as pl
from skimage.transform import iradon, radon
import h5py

# Optimisation : cache existing columnfiles in memory
#   store the NY step info here
class colfilecache( object ):
    
    def __init__(self, ymin=-15, ymax= 15.01, ystep=0.5, parfile=None):
        self.ymin = ymin
        self.ymax = ymax
        self.ystep = ystep
        self.NY = len(np.arange( ymin, ymax, ystep ) )
        self.parfile = parfile
        self.cache = {}
        
    def get( self, fname ):
        if fname not in self.cache:
            # integer dty positions for indexing arrays
            colfile = columnfile( fname )
            iy = np.round( (colfile.dty - self.ymin)/self.ystep ).astype(int)
            colfile.addcolumn( iy, "iy")
            colfile.NY = self.NY
            # ensure tth, eta, gx, gy, gz are up-to-date
            if self.parfile is not None:
                colfile.parameters.loadparameters( self.parfile )
                colfile.updateGeometry()
            self.cache[fname] = colfile
        return self.cache[fname]


"""
Script with functions to:

   load/save grain reconstruction file

   create new slices from hdf file with peaks

   map one hdf group as one peaksfile and discrete orientation

   reconfile (3D)
      slice1grain1(2D)  == hdfgroup 
         peakfile         <- string filename (z001_all.hdf)
         peakspath        <- string full path (/data/.../z001_all.hdf)
          (optional checksum on the peakfile?)
         id,iy,io,h,k,l,sign(eta)  <- n*[int,int,int,int,int]
            id    = index position in peakfile
            iy    = index position in dty
            io    = index position in omega [h,k,l,sign(eta)]
            h,k,l = assigned integers for this orientation
            sign(eta) : which side of the detector
         sinogram         <- nangles * npoints(dty)
         recon            <- npoints * npoints(dty)


   Might be in hdf file as:
     /z000/grain0/
     /z000/grain1/
     /z000/grain2/
     /z000/grain3/
     /z001/grain0/

   or instead:
     /grain0/z0/
     /grain0/z1/
     /grain0/z2/
       ....
     /grain1/z0/

load / save 2D slices to hdf groups

"""


def hkl_err( ubi, gve, errwt=(1,0,1) ):
    """
    Computes integer hkl for each peak
    Computes error in some g-vector based units
    errwt = tth_direction, omega_direction, eta_direction

    returns 
      integer h,k,l
      err = errwt * (e0,e1,e2)
    """
    assert ubi.shape == (3,3)
    assert gve.shape[0] == 3
    hkl = np.dot( ubi, gve )
    ihkl = np.round( hkl )
    gcalc = np.dot( np.linalg.inv( ubi ), ihkl )
    # error in g-vector ...
    gerr =  gcalc - gve
    #
    # Decompose this into 3 directions ...
    #   along gve = gerr . g / |g|
    #   perp to gve and z
    #   perp to both
    # 1/|g|
    x,y,z = gve
    # radial error is gve . gerr
    ng0 = gve
    # omega error is (gve x axis=001) . gerr
    ng1 = ( -y, x, np.zeros(z.shape))
    # eta error makes right handed set
    ng2 = ( x*z, y*z, -x*x-y*y)
    e0 = (gerr * ng0).sum(axis=0)**2
    e1 = (gerr * ng1).sum(axis=0)**2
    e2 = (gerr * ng2).sum(axis=0)**2
    # We mainly care about 2theta / eta error
    err = np.sqrt( errwt[0]*e0 + errwt[1]*e1 + errwt[2]*e2 )
    return ihkl, err


def loadslice( grp ):
    """
    loads reconstruction from 3D grain file (many slices, 1 grain)
    grp = hdf5 group
     NOT TESTED YET ! 
    """
    try:
        name = grp.attrs['pksfile'] 
        allpks = getcolumnfile( name )
    except:
        print("Could not get your columnfile")
        print("pksfile", grp.attrs['pksfile'] )
        print("pksfilepath", grp.attrs['pksfilepath'] )
        raise
    ubi = grp.ubi[:]
    items = "pkid", "iy", "io", "hkle", "sinogram", "angles", "recon"
    argdict = {}
    for arrayname in items:
        if arrayname in grp:
            argdict[ arrayname ] = grp[ arrayname ][:]
    return grain_recon_slice( allpks, ubi, **argdict )
                       
        
class grain_recon_slice( object ):
    """
    A 2D sinogram reconstruction
       must hold the minimum info
    """
    def __init__(self, allpks, ubi,
                 pkid = None,
                 iy   = None,
                 io   = None,
                 hkle = None,
                 sinogram = None,
                 angles= None,
                 recon = None ):
        """
        allpks is a reference to the columnfile holding *all* the peaks
        ubi  = (3x3) orientation [ [a], [b], [c] ] lattice vectors
        pkid = assigned peaks in allpks [ id = allpks[id] ] (npks)
        iy, io  = position on sinogram [ iy, iomega ]       (npks)
        hkle = h,k,l,sign(eta) for peaks in pkid 
               (4, npks)
        sinogram = float, (nuniq, ny)
        angles   = float, (nuniq,)
        recon    = float, (ny, ny)
        """
        self.allpks = allpks  # NOT SAVED, referenced by attrs[]
        self.ubi = ubi           # orientation matrix        
        self.pkid = pkid         # indexing in peaksfile
        self.iy = iy             #  iy
        self.io = io             #  iomega
        self.hkle = hkle         # h,k,l,sign(eta) of used peaks
        self.sinogram = sinogram # sinogram [nuniq x NY]
        self.angles = angles     # angles [nuniq]
        self.recon = recon       # reconstruction of intensity

    def save(self, grp):
        """
        save into a file (should work)
          assuming all items are filled in for now...
        """
        grp.attrs["pksfile"]= self.allpks.filename
        grp.attrs["pksfilepath"] = os.path.abspath( self.allpks.filename )
        # ubi average orientation - always 3x3
        grp.require_dataset( "ubi",
                             shape = (3,3),
                             dtype = np.float,
                             data = self.ubi )
        # peak labels :
        #   (id,)
        grp.require_dataset( "pkid",
                             shape = self.pkid.shape,
                             maxshape = (None,),
                             dtype = np.int32,
                             data = self.pkid)
        # iy, iomega:
        grp.require_dataset( "iy",
                             shape = self.iy.shape,
                             maxshape = (None,),
                             dtype = np.int32,
                             data = self.iy)
        grp.require_dataset( "io",
                             shape = self.io.shape,
                             maxshape = (None,),
                             dtype = np.int32,
                             data = self.io)
        # hkls can grow or shrink if we use or dont use peaks ...
        #  ... depends on gve error cutoff
        grp.require_dataset( "hkle",
                             shape = self.hkle.shape,
                             dtype = np.int32,
                             maxshape = ( None, 4),
                             data = self.hkle ) 
        # again, shrinks and grows       
        grp.require_dataset( "sinogram",
                             shape = self.sinogram.shape,
                             dtype = np.float,
                             maxshape = (None, self.recon.shape[0]),
                             data = self.sinogram )
        grp.require_dataset( "angles",
                             shape = self.angles.shape,
                             dtype = np.float,
                             maxshape = (None,),
                             data = self.angles )
        # again, shrinks and grows
        grp.require_dataset( "recon",
                             shape = self.recon.shape,
                             dtype = np.float,
                             data = self.recon )
        # always NY x NY size [top of this file]

    def check( self ):
        """
        test things look OK
        """
        npks = len(self.pkid)
        assert self.hkle.shape == (npks, 4)
        nangles = self.angles.shape[0]
        ny = self.recon.shape[0]
        assert self.sinogram.shape == (nangles,ny),self.sinogram.shape+(
            nangles,ny)
        assert self.recon.shape == (ny,ny)
        # print("check looks OK")

    def choosepeaks( self, gerrtol=None):
        """
        Decide which peaks from self.allpks that we want to use

        Fills in pkid and hkle
        """
        c = self.allpks
        gve = np.array( (c.gx,c.gy,c.gz) )
        ihkl, err = hkl_err( self.ubi, gve )
        # select peaks within tolerance
        if gerrtol is None:
            ct = (err < 0.01).sum()
            h,b = np.histogram( err, np.linspace(0, 0.01, int(ct/20) ) )
            pl.plot( b[1:],h,"-")
            pl.title("npks versus error")
            pl.show()
            # py2/3 thing:
            gerrtol = float( input( "Enter cut off gerrtol: ") )
        m0 = err < gerrtol
        npks = m0.sum()
        self.pkid = np.arange(0,c.nrows,dtype=np.int32)[m0]
        h,k,l = ihkl[:,m0]
        se    = np.sign(c.eta[m0])
        self.hkle = np.array( ( h,k,l,se), np.int32).T
        
    def makesino( self ):
        """ 
        NY = number of points in Y scan <- global
        iy = y index positions from self.allpks
        omega = omega angles   from self.allpks
        h,k,l,se = labels to get uniq peaks (se == sign(eta))
        intensities  = peak intensities
        
        fills in 
        self.angles = <omega> for projection
        self.sinogram[na,NY] = (max) intensity at each iy/uniq angle
        self.iy, self.io = indexing for peak into sinogram
        """
        npks = len(self.pkid)
        assert self.hkle.shape == (npks, 4), self.hkle.shape
        ## FIXME : sortable thing is [h,k,l,se,iy]
        ##         ...go through in order assigning io
        h,k,l,se = self.hkle.T
        iy = self.allpks.iy[ self.pkid ].astype( np.int32 )
        # numpy lexsort : sorts axis by axis
        # hklsy = np.array( (h,k,l,se,iy,self.pkid) )
        hklsy = np.array( (self.pkid,iy,se,l,k,h) )
        order = np.lexsort( hklsy )
        # find out how many projections we have
        io = np.zeros( npks, np.int32 )
        iproj = 0
        current = hklsy[2:,order[0]]
        # i_omega
        for i in order:
            t = hklsy[2:,i] # h,k,l,se
            if not (t == current).all():
                current = t
                iproj += 1
            io[ i ] = iproj
        self.iy = iy
        self.io = io
        self.nproj = iproj + 1
        self.fill_sinogram()

    def fill_sinogram(self):            
        # now fill in the sinogram
        self.sinogram = np.zeros((self.nproj, self.allpks.NY)) - 1
        self.angles = np.zeros(self.nproj)
        intensities = self.allpks.sum_intensity[ self.pkid ]
        # Not needed unless you are masking
        assert intensities.min() >= 0
        # Now fill in the intensities and angles
        omega =       self.allpks.omega[ self.pkid ]
        io = self.io
        iy = self.iy
        for i in range( len(self.pkid) ):
            t = self.sinogram[ io[i], iy[i]]
            if intensities[i] > t:
                self.sinogram[ io[i], iy[i]] = intensities[i]
                # We do the intensity weighted average for omega
                self.angles[io[i]] += omega[i]
        self.smask = self.sinogram < 0
        self.sinogram = np.where( self.smask, 0, self.sinogram )
        # normalise intensity
        isum = self.sinogram.sum( axis = 1 )
        npp = self.allpks.NY - self.smask.sum(axis=1)
        self.angles /= npp 
        self.sinogram = self.sinogram/self.sinogram.max(axis=1)[:, np.newaxis]
        self.sort_omega()
        
    def sort_omega( self ):
        """
        put the projections in order
        """
        order = np.argsort(self.angles)
        self.angles = self.angles[ order ]
        self.sinogram = self.sinogram[ order ]
        self.io = order[ self.io ]

    def run_iradon(self):
        """
        Fills in self.recon from self.sinogram and self.angles
        """
        self.recon = iradon( self.sinogram.T, self.angles, circle=True )

    def clean(self, cctol=None):
        """
        Apply a tolerance in cor-coeff on projections
        to kill the worst ones
        (should work)
        """
        # scalc is "self-consistent", just reverse transform
        scalc = radon( self.recon, self.angles, circle=True ).T

        # Scor each angle project to see how well it fits
        scors = np.array( [np.corrcoef(self.sinogram[i], scalc[i])[1,0]
                           for i in range(len(self.angles))] )
        # apply a correlation coefficient cutoff
        if cctol is None:
            pl.subplot(121)
            pl.plot( self.angles, scors, "o")
            pl.subplot(122)
            pl.hist( scors, np.linspace(0,1,len(scors)/10))
            pl.show()
            # might change of py3/py2
            cctol = float( input( "Enter cut off for cctol: ") )

        io = self.io # i_omega indices of peaks to sinogram
        msk = np.zeros( len(io), np.bool )
        for i in range(len(self.angles)):
            if scors[i] < cctol:
                # remove
                msk = msk | (io == i)
        # filter peak list according to masking
        self.pkid = self.pkid[msk]
        self.io   = self.io[msk]
        self.iy   = self.iy[msk]
        self.hkle = self.hkle[msk]
                       
        recon_mask = scors  > cctol
        # FIXME = remove pkid or add a used / not used mask
        self.sinogram = self.sinogram[ recon_mask ]
        self.angles = self.angles[ recon_mask ]
        
        self.run_iradon()

                
def create_slice( colfile, ubi,
                  gerrtol = None,
                  cctol = None):
    """
    given a columnfile
    ubi matrix this assign peaks and creates a "slice"
    
    gerrtol = cutoff for assignment of peaks to ubi
    cctol   = cutoff for correlation coefficient on sino - recon
    
    return a grain_recon_slice object
    """
    global ymin, ystep
    # read the datafile with the spots
    #
    slc = grain_recon_slice( colfile, ubi )
    #
    # Compute the hkl error for this UBI matrix
    #  and decide which peaks to use
    slc.choosepeaks( gerrtol )
    #
    # Make an initial sinogram (may contain overlaps)
    slc.makesino()
    #
    # Make a reconstruction
    slc.run_iradon()
    # debugging
    slc.check()
    slc.clean( cctol )
    slc.check()
    return slc

def recon_all_peaks( colfile, mask = None, abins = 180 ):
    """
    Does a reconstruction of all peaks in a columnfile
    ignoring the hkl indexing and intensity normalisation
    """
    if mask is None:
        s = np.histogram2d( colfile.omega,
                            colfile.iy,
                            bins=(abins,colfile.NY) )
    else:
        s = np.histogram2d( colfile.omega[mask],
                            colfile.iy[mask],
                            bins=(abins,colfile.NY) )
    r = iradon( s[0].T, circle=True )
    return r


if __name__=="__main__":

    pksfile = sys.argv[1]
    parfile = sys.argv[2]

    ccache = colfilecache( ymin=-15,
                           ymax=15.01,
                           ystep=0.5,
                           parfile=parfile )
    
    colfile = ccache.get( pksfile )
                             
    grains = grain.read_grain_file( sys.argv[3] )
    mapfilename = sys.argv[4]
    
    mapf = h5py.File( mapfilename, "w" )    
    pl.ion()
    for k in range(len(grains)):
        ubi = grains[k].ubi
        slc = create_slice( colfile, ubi, gerrtol = 0.0025, cctol = 0.8 )
        grp = mapf.require_group("grain_%d"%(k))
        slc.save( grp )
        pl.figure(1)
        pl.imshow(slc.recon)
        pl.title("%d %s"%(k,slc.allpks.filename))
        pl.draw()
        mapf.flush()
    mapf.close()

    sys.exit()

# rnice8-0207:~/id11/merged_peaks % ipython -i dorecon_grain.py z500_all.hdf fit.par no_duplicates/t.ubi  test1.hdf
