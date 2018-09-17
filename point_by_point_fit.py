
from __future__ import print_function, division

# Attempt to fit "strain" by refining unit cell parameters at
# each point on a sinogram.

import sys
import numpy as np, pylab as pl
from scipy.optimize import leastsq
import scipy.sparse
from skimage.transform import iradon, radon
from ImageD11 import columnfile, refinegrains, parameters, grain, transform, \
    indexing, cImageD11



def calc_tth_eta_omega( ub, hkls, pars, etasigns ):
    """
    Predict the tth, eta, omega for each grain
    ub = ub matrix (inverse ubi)
    hkls = peaks to predict
    pars = diffractometer info (wavelength, rotation axis)
    etasigns = which solution for omega/eta to choose (+y or -y)
    """
    g = np.dot( ub, hkls )
    tthcalc, eta2, omega2 = transform.uncompute_g_vectors(
        g,
        pars.get('wavelength'),
        wedge=pars.get('wedge'),
        chi=pars.get('chi') )
    # choose which solution (eta+ or eta-)
    e0 = np.sign(eta2[0]) == etasigns
    etacalc = np.where( e0, eta2[0], eta2[1] )
    omegacalc = np.where( e0, omega2[0], omega2[1] )
    return tthcalc, etacalc, omegacalc


def update_mask( mygr, flt, pars, nmedian ):
    """
    Remove 5*median_error outliers from grains assigned peaks
    This routine fills in mygr.mask and mygr.hkl
    """
    # obs data for this grain
    tthobs = flt.tth[ mygr.mask ] 
    etaobs = flt.eta[ mygr.mask ]
    omegaobs = flt.omega[ mygr.mask ]
    gobs = np.array( (flt.gx[mygr.mask], flt.gy[mygr.mask], flt.gz[mygr.mask]) )
    # hkls for these peaks
    hklr = np.dot( mygr.ubi, gobs )
    hkl  = np.round( hklr )
    # Now get the computed tth, eta, omega
    etasigns = np.sign( etaobs )
    mygr.hkl = hkl.astype(int)
    mygr.etasigns = etasigns
    tthcalc, etacalc, omegacalc = calc_tth_eta_omega(
        mygr.ub, hkl, pars, etasigns )
    # update mask on outliers
    dtth = (tthcalc - tthobs)
    deta = (etacalc - etaobs)
    domega = (omegacalc - omegaobs)
    msk  = abs( dtth ) > np.median( abs( dtth   ) ) * nmedian
    msk |= abs( deta ) > np.median( abs( deta   ) ) * nmedian
    msk |= abs( domega)> np.median( abs( domega ) ) * nmedian
    allinds = np.arange( flt.nrows )
    mygr.mask[ allinds[mygr.mask][msk] ] = False
    return msk.astype(int).sum()


def calc_teo_fit( ub, flt, pars, gr):
    """
    Function for refining ub using tth, eta, omega data
    ub is the parameter array to fit
    flt is all the data
    pars in the diffractometer geometry to get tthcalc, etacalc, omegacalc
    gr is the grain holding the peak assignments
    flt.wtth, weta, wometa = weighting functions for tth vs eta vs omega errors
    """
    UB = np.array(ub)
    UB.shape=3,3
    
    tthcalc, etacalc, omegacalc = calc_tth_eta_omega(
        UB, gr.hkl, pars, gr.etasigns )
    
    dtth   = ( flt.tth[ gr.mask ]   - tthcalc   ) * flt.wtth[ gr.mask ]
    deta   = ( flt.eta[ gr.mask ]   - etacalc   ) * flt.weta[ gr.mask ]
    domega = ( flt.omega[ gr.mask ] - omegacalc ) * flt.womega[ gr.mask ]
    return np.concatenate( (dtth, deta, domega) )



def estimate_weights( pars, flt, OMSLOP ):
    distance  = pars.get('distance')
    pixelsize = ( pars.get('y_size') + pars.get('z_size') ) / 2.0
    # 1 pixel - high energy far detector approximation
    wtth = np.ones(flt.nrows)/ np.degrees( pixelsize / distance )
    weta = wtth * np.tan( np.radians( flt.tth ) )
    womega = np.ones(flt.nrows)/OMSLOP
    print("Weights:")
    print("    tth:",wtth[0] )
    print("    eta:",weta.min(),weta.max(),weta.mean())
    print("  omega:",womega[0] )
    return wtth, weta, womega

    

def fit_one_grain( gr, flt, pars ):
    """
    Uses scipy.optimize to fit a single grain
    """
    # print("Cell before:",("%.6f  "*6)%( indexing.ubitocellpars( gr.ubi )))
    args = flt, pars, gr
    x0 = gr.ub.ravel().copy()
    try:
        xf, cov_v, info, mesg, ier = leastsq( calc_teo_fit, x0,
                                              args, full_output=True )
    except:
        xf = x0
    ub = xf.copy()
    ub.shape = 3,3
    ubi = np.linalg.inv(ub)

    # print("Cell after :",("%.6f  "*6)%( indexing.ubitocellpars( ubi ) ))
    gr.set_ubi( ubi )
    

def update_cols( flt, pars, OMSLOP ):
    """
    update the twotheta, eta, g-vector columns to be sure they are right
    fill in some weighting estimates for fitting
    """
    tth, eta = transform.compute_tth_eta( [flt.sc, flt.fc], **pars.parameters )
    gve      = transform.compute_g_vectors( tth, eta, flt.omega,
                                            pars.get('wavelength'),
                                            wedge=pars.get('wedge'),
                                            chi=pars.get('chi') )
    flt.addcolumn( tth  , "tth" )
    flt.addcolumn( eta  , "eta" )
    # Compute the relative tth, eta, omega errors ...
    wtth, weta, womega = estimate_weights( pars, flt, OMSLOP )
    flt.addcolumn( wtth, "wtth" )
    flt.addcolumn( weta, "weta" )
    flt.addcolumn( womega, "womega" )
    
    flt.addcolumn( gve[0], "gx" )
    flt.addcolumn( gve[1], "gy" )
    flt.addcolumn( gve[2], "gz" )
    return tth, eta, gve


def assign_peaks( grains, gve, flt, pars, nmedian, hkltol ):
    """
    Assign peaks to grains for fitting
    - each peak chooses the spots it likes
    - overlapping spots (chosen by more than 1 grain) are removed
    - fit outliers are removed abs(median err) > nmedian
    Fills in grain.mask for each grain
    """
    for i, g in enumerate(grains):
        # For each grain we compute the hkl integer labels
        hkl = np.dot( g.ubi, gve )
        hkli = np.round( hkl )
        # Error on these:
        drlv = hkli - hkl
        drlv2 = (drlv*drlv).sum(axis=0)
        # Tolerance to assign to a grain is rather poor
        g.mask = drlv2 < hkltol*hkltol
        print( "Grain",i,"npks",(g.mask.astype(int)).sum())
    
    print("Checking for peaks that might overlap")
    overlapping = np.zeros( flt.nrows, dtype=bool )
    for i in range(len(grains)):
        for j in range(i+1,len(grains)):
            overlapping |= grains[i].mask & grains[j].mask
    print("Total peaks",flt.nrows,"overlapping",overlapping.astype(int).sum())
    
    for i, g in enumerate(grains):
        g.mask &= ~overlapping
        print( "Grain",i,"npks",(g.mask.astype(int)).sum(),end=" " )    
        print("update mask",end=" ")
        while 1:
            ret = update_mask( g, flt, pars, nmedian=nmedian )
            print(ret,end=" ")
            if ret == 0:
                break
        print(g.mask.astype(int).sum())

def fit_dty( g, flt ):
    """
    Fit a sinogram to get a grain centroid
    """
    dty = flt.dty[ g.mask ]
    romega = np.radians( flt.omega[ g.mask ] )
    co  = np.cos( romega )
    so  = np.sin( romega )
    # calc = d0 + x*co + y*so
    # dc/dpar : d0 = 1
    #         :  x = co
    #         :  y = so
    # gradients
    g = [ np.ones( dty.shape, float ),  co, so ]
    nv = len(g)
    m = np.zeros((nv,nv),float)
    r = np.zeros( nv, float )
    for i in range(nv):
        r[i] = np.dot( g[i], dty )
        for j in range(i,nv):
            m[i,j] = np.dot( g[i], g[j] )
            m[j,i] = m[i,j]
    sol = np.dot(np.linalg.inv( m ), r)
    return sol

def uniq( vals ):
    d = {}
    newvals = []
    for v in vals:
        if v not in d:
            d[v]=0
            newvals.append(v)
    return newvals

def map_grain( g, flt, ymin, ystep, omegastep ):
    """
    Computes sinogram
    Runs iradon
    Returns angles, sino, recon
    """
    iy = np.round( (flt.dty[ g.mask ] - ymin) / ystep ).astype(int)    
    omega = np.round( flt.omega[ g.mask ] / omegastep ).astype(int)

    keys = [ (hkl[0], hkl[1], hkl[2], int(s))
             for hkl, s in zip(g.hkl.T , g.etasigns)]

    uni = uniq(keys)
    akeys = np.array( keys )

    sum_intensity = flt.sum_intensity[ g.mask ]
    assert (sum_intensity > 0).all(), "peaks are positive" 
    
    NY = int(np.round(abs(ymin)*2/ystep)) + 1
    npks = len( uni )
    sino = np.zeros( ( npks, NY ), np.float )
    angs = np.zeros( ( npks, NY ), np.float )
    
    for refi,u in enumerate(uni):
        
        # h==h, k==k, l==l, sign==sign
        mask = (akeys == u).astype(int).sum(axis=1) == 4
        
        dtypos = iy[mask]
        intensities = sum_intensity[mask]
        angles = omega[mask]
        
        ndup = 0
        for yindex, counts, omegapk in zip( dtypos, intensities, angles ):
            # Take strongest if more than one
            if counts > sino[refi][yindex]:
                sino[refi][yindex] = counts
                angs[refi][yindex] = omegapk
                continue
            if sino[refi][yindex] > 0:
                ndup += 1
#        if ndup > 0:
#            print(ndup,"duplicates!")

    sinoangles = np.sum( angs, axis = 1) / np.sum( sino > 0, axis = 1)
    # Normalise:
    sino = (sino.T/sino.max( axis=1 )).T
    # Sort (cosmetic):
    order = np.argsort( sinoangles )
    sinoangles = sinoangles[order]
    ssino = sino[order].T
    # Reconstruct
    output_size = int( NY*1.5 )
    recon = iradon( ssino, theta=sinoangles, output_size=output_size,
                    circle = False )

    if 0:
        # code to clean up sinogram - didn't help
        calcsino = radon( recon, theta=sinoangles, circle = False )
        px0 = calcsino.shape[0]//2 - ssino.shape[0]//2
        cs = calcsino[ px0 : px0 + ssino.shape[0] ]
        error = (ssino - cs)
        medabserr = np.median( abs( error.ravel() ) )
        cleanedsino =  np.where( abs(error) >5*medabserr, cs, ssino )
        recon = iradon( cleanedsino, theta=sinoangles,
                        output_size=output_size,  circle = False )
    
    return sinoangles, ssino, recon
                
                
            
def fit_one_point( g, flt, pars, ix, iy, ystep ):
    """
    Take each time the nearest point in dty (not the mask!)
    """
    om = np.radians( flt.omega[g.mask] )
    co = np.cos( om )
    so = np.sin( om )
    idtycalc = np.round(-ix * so + iy * co)
    idty = np.round(flt.dty[ g.mask ] / ystep)
    #     m = abs(dty - dtycalc) < ystep*0.75
    m = idtycalc == idty
    if 0:
        pl.figure()
        pl.plot( om, dty, ".")
        pl.plot( om, dtycalc, ".")
        pl.show()
    grfit = grain.grain( g.ubi )
    grfit.hkl = g.hkl[:,m]
    grfit.etasigns = g.etasigns[m]
    inds = np.arange( flt.nrows, dtype=int )
    grfit.mask = np.zeros( flt.nrows, np.bool )
    grfit.mask[ inds[g.mask][m] ] = True
    fit_one_grain( grfit, flt, pars )
    return grfit


def make_sino( g, flt, pars, ymin, ystep):
    """
    Computes sinogram
    Builds up a sparse least square problem
    method = 'nearest' use the nearest pixel
           = 'bilinear' for bilinear interpolation
    """
    # integer values of dty/omega for binning
    iy = np.round( (flt.dty[ g.mask ] - ymin) / ystep ).astype(int)
    omega = np.round( flt.omega[ g.mask ] / omegastep ).astype(int)
    # observed peaks grouped into projections via h,k,l,sign(eta)
    keys = [ (hkl[0], hkl[1], hkl[2], int(s))
             for hkl, s in zip(g.hkl.T , g.etasigns)]
    uni = uniq(keys)
    akeys = np.array( keys )
    # sum of the peak intensity is to be fitted
    # eventually add tth, eta, omega
    sum_intensity = flt.sum_intensity[ g.mask ]
    assert (sum_intensity > 0).all(), "peaks are positive" 
    # Ysteps for the sinogram
    NY = int(np.round(abs(ymin)*2/ystep)) + 1
    npks = len( uni )
    sino = np.zeros( ( npks, NY ), np.float )
    angs = np.zeros( ( npks, NY ), np.float )
    # For reconstructing later : pmat is density. Will also need ub ...
    pmat = np.zeros( ( NY, NY ), np.float )
    # Fill the sinogram
    for refi,u in enumerate(uni):        
        # h==h, k==k, l==l, sign==sign
        mask = (akeys == u).astype(int).sum(axis=1) == 4
        dtypos = iy[mask]
        intensities = sum_intensity[mask]
        angles = omega[mask]
        ndup = 0
        for yindex, counts, omegapk in zip( dtypos, intensities, angles ):
            # Take strongest if more than one
            if counts > sino[refi][yindex]:
                sino[refi][yindex] = counts
                angs[refi][yindex] = omegapk
                continue
            if sino[refi][yindex] > 0:
                ndup += 1
        if ndup > 0:
            print(ndup,"duplicates!")
    # average along y
    sinoangles = np.sum( angs, axis = 1) / np.sum( sino > 0, axis = 1)
    # Normalise (allows for hkl,eta intensity variation)
    sino = (sino.T/sino.max( axis=1 )).T

    # Now construct a least squares problem. For each pixel in the sinogram
    # compute the derivative with respect to pmat[i,j].
    idtycalc = np.round(-ix * so + iy * co)
    
    return sinoangles, ssino, recon
                

def sparse_recon( sinoshape, angles, circle = True, output_size = None,
                  method = 'nearest' ):
    """
    Builds up a sparse least square problem
    method = 'nearest' use the nearest pixel
           = 'bilinear' for bilinear interpolation
    To come : intensity weighted twotheta, eta and omega

    This needs a diagram...
       A . recon           = sinogram
       A . nout,nout       = nangles, nproj
       A has row length (nout*nout) => dots with recon
       A has col height (nangles*nproj)
    
        aaaa . r  =  s
               r
               r
               r

       e.g. a single point in the sinogram is a row of A dot the recon

       Much of A is empty. A recon pixel only hits a sine
       wave path in the output.

    For the 'nearest' case: each recon pixel hits one projection
    pixel for each angle in the sinogram

    For the splitting case: ideally we should integrate areas.
    """
    NY, NANGLES = sinoshape
    assert NANGLES == len(angles)
    NOUT = NY
    if output_size is not None:
        assert output_size >= NOUT
        NOUT = output_size
    i, j = np.mgrid[0:NOUT, 0:NOUT]                  # pixel indices
    x, y = np.mgrid[0:NOUT, 0:NOUT] - NOUT // 2      # vs centre of rotation
    if circle:
        mask = x*x+y*y < NY*NY//4
    else:
        raise Exception("Circle is needed for now")
    #
    # Pixels in the recon to be directed to output by A:
    NMASK = np.where(mask, 1, 0).ravel().sum()
    #
    # We will index as A[i][j] with [i] being row number and [j]
    # being the column.
    #
    #  ... there will be two for bilinear, one for nearest
    nsplit = { 'nearest' : 1,
               'bilinear' : 2 }
    assert method in nsplit
    # j is the angle + projection position, always different
    A_adr_j = np.zeros( (NANGLES , nsplit[method], NMASK ), np.int)
    A_ij =    np.ones( (NANGLES , nsplit[method], NMASK ), np.float32)
    #
    # i is the position in the recon, always the same
    row = (i * NOUT + j )[mask].ravel()
    A_adr_i = np.zeros( (NANGLES , nsplit[method], NMASK ), np.int)
    A_adr_i[:,:] = row
    # 
    # sins and cosines for pixel locations
    so = np.sin(np.radians( angles ) )
    co = np.cos(np.radians( angles ) )
    # Active pixels
    xa = x[mask]
    ya = y[mask]
    if method == 'bilinear':
        bi = True
    else:
        bi = False
    for i in range( len( angles ) ):
        # position in projection, C.O.R centred on zero (NOUT//2 above)
        z  = co[i]*ya - so[i]*xa + NY // 2
        iz = np.round(z).astype(int)
        # Mask to select points which hit sinogram
        if bi:
            dz = z - iz
            # sign(0) = 0, which would be bad for us, so avoid it
            iz2 = iz + np.where( dz >= 0, 1, -1)
            assert iz2.min() >= 0
            assert iz2.max() < NY, (NY, iz2.max(),i,dz)
            A_adr_j[ i, 1 ] = (i * NY + iz2)
            # close to iz2, dz ~= 1
            A_ij[ i, 0 ] = abs(dz)
            # close to iz, dz ~= 0
            A_ij[ i, 1 ] = 1-abs(dz)
    print( 'i min/max',A_adr_i.min(),A_adr_i.max())
    print( 'j min/max',A_adr_j.min(),A_adr_j.max())
    print( NANGLES*NY, NOUT*NOUT)
    amat = scipy.sparse.coo_matrix( ( A_ij.ravel(), (A_adr_j.ravel(), A_adr_i.ravel())),
                                    shape = (NANGLES*NY, NOUT*NOUT),
                                    dtype = np.float32 )
    return amat, NOUT
        
def sparse_recon( sinoshape, angles, circle = True, output_size = None,
                  method = 'nearest' ):
    """
    Builds up a sparse least square problem
    method = 'nearest' use the nearest pixel
           = 'bilinear' for bilinear interpolation
    To come : intensity weighted twotheta, eta and omega

    This needs a diagram...
       A . recon           = sinogram
       A . nout,nout       = nangles, nproj
       A has row length (nout*nout) => dots with recon
       A has col height (nangles*nproj)
    
        aaaa . r  =  s
               r
               r
               r

       e.g. a single point in the sinogram is a row of A dot the recon

       Much of A is empty. A recon pixel only hits a sine
       wave path in the output.

    For the 'nearest' case: each recon pixel hits one projection
    pixel for each angle in the sinogram

    For the splitting case: ideally we should integrate areas.
    """
    NY, NANGLES = sinoshape
    assert NANGLES == len(angles)
    NOUT = NY
    if output_size is not None:
        assert output_size >= NOUT
        NOUT = output_size
    i, j = np.mgrid[0:NOUT, 0:NOUT]                  # pixel indices        
    x, y = np.mgrid[0:NOUT, 0:NOUT] - NOUT // 2      # pos vs centre of rotation
    #  mask circle if asked:
    if circle:
        mask = x*x+y*y < NY*NY//4
    else:
        raise Exception("Circle is needed for now")
    #
    # We will index as A[i][j] with [i] being row number and [j]
    # being the column.
    #
    # So [i] is the angle/dty in the sinogram and j is the
    # position of the pixel in the sample (x/y)
    #
    #  ... there will be two outputs for bilinear, one for nearest
    nsplit = { 'nearest' : 1,
               'bilinear' : 2 }
    assert method in nsplit
    # j is the angle + projection position, always different
    A_i  = []  # index i = sinogram address
    A_j  = []  # index j = sample address
    A_ij = []  # coefficient. 1 for nearest. [0-1] for split.
    #
    # sins and cosines for pixel locations
    so = np.sin(np.radians( angles ) )
    co = np.cos(np.radians( angles ) )
    # Active pixels
    jj = i*NOUT + j
    if method == 'nearest':
        print("Using nearest method")
        for iang in range( len( angles ) ):
            # position in projection, C.O.R centred on zero (NOUT//2 above, NY here)
            z  = co[iang]*y - so[iang]*x + NY // 2
            iz = np.round(z).astype(int)
            # Address in sinogram [i] is therefore i*NOUT + iz
            m = (iz >= 0)&(iz < NY)
            ivals = iz[m] + iang * NY
            jvals = jj[m] 
            A_i.append( ivals )
            A_j.append( jvals )
            A_ij.append( np.ones( len(ivals), np.float32 ))
        A_i = np.concatenate( A_i )
        A_j = np.concatenate( A_j )
        A_ij = np.concatenate( A_ij )
        print( 'i min/max',np.min(A_i),np.max(A_i))
        print( 'j min/max',np.min(A_j),np.max(A_j))
        print( NANGLES*NY, NOUT*NOUT)
        amat = scipy.sparse.coo_matrix( ( A_ij, (A_i, A_j)),
                                        shape = (NANGLES*NY, NOUT*NOUT),
                                        dtype = np.float32 )
        return amat, NOUT
            
    if method == 'bilinear':
        print("Using bilinear method")
        for iang in range( len( angles ) ):
            # position in projection, C.O.R centred on zero (NOUT//2 above,
            # NY here)
            z  = co[iang]*y - so[iang]*x + NY // 2
            iz1 = np.round(z).astype(int)
            dz = z - iz1
            # sign(0) = 0, which would be bad for us, so avoid it
            iz2 = iz1 + np.where( dz >= 0, 1, -1)

            # Address in sinogram [i] is therefore i*NOUT + iz
            m1 = (iz1 >= 0)&(iz1 < NY)

            A_i.append( iz1[m1] + iang * NY )
            A_j.append(  jj[m1] )
            A_ij.append( 1 - abs(dz[m1]) ) # dz == 0

            m2 = (iz2 >= 0)&(iz2 < NY)
            A_i.append( iz2[m2] + iang * NY )
            A_j.append(  jj[m2] )
            A_ij.append( abs(dz[m2]) )
        A_i = np.concatenate( A_i )
        A_j = np.concatenate( A_j )
        A_ij = np.concatenate( A_ij )
        print( 'i min/max',np.min(A_i),np.max(A_i))
        print( 'j min/max',np.min(A_j),np.max(A_j))
        print( NANGLES*NY, NOUT*NOUT)
        amat = scipy.sparse.coo_matrix( ( A_ij, (A_i, A_j)),
                                        shape = (NANGLES*NY, NOUT*NOUT),
                                        dtype = np.float32 )
        return amat, NOUT
    
    
    
def test_sparse_recon(NY=101, NA=180, NOUT=None, method='nearest' ):
    """
    """
    sinoshape = NY, NA
    if NOUT is None:
        NOUT = NY
    angs = np.linspace(0, 179, NA)
    amat, ntest = sparse_recon(sinoshape, angs, output_size=NOUT, method=method)
    assert ntest == NOUT
    print ( repr(amat ) )
    #
    x = np.zeros( (NOUT, NOUT), np.float32 )
    x[NOUT//2, NOUT//2] = 0
    x[NOUT//4:NOUT*3//4,(NOUT+5)//2]=1
    x[(NOUT+5)//2,NOUT//4:NOUT*3//4]=1
    scalc = amat.dot( x.ravel() )
    scalc.shape = NA, NY
    #scipy.linalg.lsmr
    return scalc, amat, x

LSMR_DOC = {0: "means x=0 is a solution.",
            1: "means x is an approximate solution to A*x = B, according to atol and btol.",
            2: "means x approximately solves the least-squares problem according to atol.",
            3: "means COND(A) seems to be greater than CONLIM.",
            4: "is the same as 1 with atol = btol = eps (machine precision)",
            5: "is the same as 2 with atol = eps.",
            6: "is the same as 3 with CONLIM = 1/eps.",
            7: "means ITN reached maxiter before the other stopping conditions were satisfied."
            }

#def test_lsmr_for_recon():
from pylab import *

def diagsolve( A, b):
    diagATA = np.zeros( A.shape[1] )
    c = A.tocsr()
    for i in range(c.shape[1]):
        d = c.getcol(i).data
        diagATA[i] = d.sum()
    ATb = A.T.dot( b )
    x0 = ATb / diagATA 
    print('x0',x0.shape,x0.min(),x0.max(), ATb.shape, diagATA.shape)
    clf()
    subplot(221)
    imshow( ATb.reshape(31,31))
    colorbar()
    subplot(222)
    imshow( diagATA.reshape(31,31))
    colorbar()
    subplot(223)
    imshow( x0.reshape(31,31))
    colorbar()
    subplot(224)
    imshow(b.reshape(60,21).T)
    colorbar()

    print(ATb.shape, A.shape, x0.shape, b.shape )
    print(A.dot(x0).shape)
    residual = A.dot( x0 ) - b
    subplot(223)
    imshow( x0.reshape(31,31))    
    show()
    print(np.dot(residual, residual))
    for i in range(3):
        x0 += A.T.dot( residual ) / diagATA
        residual = A.dot( x0 ) - b
        print(np.dot(residual, residual))
    return x0, ATb, diagATA


if 0:
    scalcb, amatb, xb =  test_sparse_recon(NY=21, NA=60, NOUT=31, method='bilinear' )
    x0, ATb, diagATA= diagsolve( amatb, scalcb.ravel() )
    figure()
    scalcn, amatn, xn =  test_sparse_recon(NY=21, NA=60, NOUT=31, method='nearest' )
    x0n, ATbn, diagATAn= diagsolve( amatn, scalcn.ravel() )
if 0:
    sino = scalc
    sino.shape = 60, 21
    recon = x
    recon.shape = 31, 31
    testsino = amat.dot( recon.ravel() )
#    assert (testsino == 0).all()
    ret = scipy.sparse.linalg.lsmr( amat, sino.ravel(), show=True, maxiter=1000 )
    xnew = ret[0]
    xnew.shape = recon.shape
    istop = ret[1]
    print("Got istop=",istop,LSMR_DOC[istop])
    for v,n in zip(ret[2:],"""Number of iterations used;norm(b-Ax);norm(A^T (b - Ax));norm(A);Condition number of A.;norm(x)""".split(";")):
        print(n,":\t\t",v)

    
                   


    
    
    
    

# TODO : bilinear into proj...
                     
        
def map_out_cell( g, flt ):
    sol = fit_dty( g, flt )
    print("#",sol)
    return sol

def main():
    flt    = columnfile.columnfile( sys.argv[1] )
    grains = grain.read_grain_file( sys.argv[2] )
    pars   = parameters.read_par_file( sys.argv[3] )
    newgrainfile  = sys.argv[4]

    hkltol = 0.05    #  for first peak assignments
    nmedian = 5      #  for removing peak fit outliers
    omegastep = 1.0  #  for omega images
    ymin = -18      #  dty start (so -15 -> +15 in 0.25 steps)
    ystep = 0.25     #  step in dty from scan
    rcut  = 0.2      #  cutoff for segmentation of reconstruction
    
    OMSLOP = omegastep / 2.0
    
    
    tth, eta, gve = update_cols( flt, pars, OMSLOP )
    assign_peaks( grains, gve, flt, pars, nmedian,  hkltol )
    pl.ioff()
    print("\n\n")
    for i,g in enumerate(grains):
        print("# Grain:",i)
        fit_one_grain( g, flt, pars )
        y0,x,y = map_out_cell( g, flt )
        sinoangles, sino, recon = map_grain( g, flt, ymin, ystep, omegastep )
        if 0:
            pl.subplot(211)
            pl.imshow( sino )
            pl.subplot(212)
            pl.imshow( recon )
            pl.show()
        active = recon > recon.max() * rcut
        ii, jj = np.mgrid[ 0:recon.shape[0], 0:recon.shape[0] ] - recon.shape[0]//2
        for ix, iy in zip(ii[active], jj[active]):
            gf = fit_one_point( g, flt, pars, ix, iy, ystep )
            print(("%-4d  "*4)%(i,ix,iy,gf.mask.astype(int).sum()),end=" ")
            print(("%.6f  "*6)%( indexing.ubitocellpars( gf.ubi )), end=" ")
            print(("%.6f  "*3)%tuple(gf.Rod ))
        g.translation = (x,y,0)
            
    grain.write_grain_file( newgrainfile, grains )
    

if __name__=="__main__":

    main()



    
