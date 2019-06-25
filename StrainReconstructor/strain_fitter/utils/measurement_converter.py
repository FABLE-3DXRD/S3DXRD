import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import iradon, radon
from xfab import tools

def sind( a ): return np.sin( np.radians( a ) )

def cosd( a ): return np.cos( np.radians( a ) )

def omega_matrix( om ):
    return np.array([[cosd(om),-sind(om),0],[sind(om),cosd(om),0],[0,0,1]])

def get_k(tth, eta, wavelength):
    return (2*np.pi/wavelength)*np.array([1, 0, 0])

def get_k_prime(tth, eta, wavelength):
    k1 = cosd(tth)
    k2 = -sind(tth)*sind(eta)
    k3 = sind(tth)*cosd(eta)
    return (2*np.pi/wavelength)*np.array([k1,k2,k3])

def get_Q(tth, eta, wavelength):
    k = get_k(tth, eta, wavelength)
    kp = get_k_prime(tth, eta, wavelength)
    return k - kp

def normal_omega(tth, eta, wavelength, omega):
    '''
    Return strained planes normal in omega coordinate system
    '''
    Qlab = get_Q(tth, eta, wavelength)
    Qomega = np.dot( np.linalg.inv( omega_matrix( omega ) ), Qlab )
    return -Qomega/np.linalg.norm(Qomega)

def strain(B, h, k, l, tth, wavelength):
    #G_original = np.sqrt((h*a_rec)**2 + (k*b_rec)**2 + (l*c_rec)**2)

    G_original = np.dot(B,np.array([h,k,l]))

    # Elm. of Modern X-ray P. pp. 157
    d_original = 2*np.pi / np.linalg.norm(G_original)

    # Bragg's law
    m = np.round(2*d_original*sind(tth/2.)/wavelength)
    d_measured = (m*wavelength)/(2*sind(tth/2.))

    return m, d_measured, d_original, (d_measured-d_original)/d_original

def weight(d_measured, d_original, strain, tth, wavelength, bragg_order, distance, pixelsize):
    '''
    Calculate weights based on resolution concerns, to be used in As=m taking
    into account that the rows of A are differently accurate measurements.

    UNITS:
          wavelength  -  arbitrary unit A
          d_original  -  arbitrary unit A
          d_measured  -  arbitrary unit A
          distance    -  arbitrary unit B
          pixelsize   -  arbitrary unit B
    '''

    tth_rad = np.radians( tth )
    r = np.tan( tth_rad )*distance # peak location radius
    dtth_rad = np.arctan( (r + pixelsize )/distance ) - tth_rad # angular width of a pixel
    d_rdr = ( (bragg_order*wavelength)/(2*np.sin( (tth_rad+dtth_rad)/2. ) ) )
    eps = (d_rdr-d_original)/d_original # strain at radius r + dr
    w = abs(1 / (strain - eps) )
    assert strain>eps
    return w

def uniq( vals ):
    d = {}
    newvals = []
    for v in vals:
        if v not in d:
            d[v]=0
            newvals.append(v)
    return newvals


def extract_strain_and_directions(cell, wavelength, distance, pixelsize, g, flt, ymin, ystep, omegastep, NY ):

    B = tools.form_b_mat(cell)

    keys = [ (hkl[0], hkl[1], hkl[2], int(s))
             for hkl, s in zip(g.hkl.T , g.etasigns)]

    uni = uniq(keys)
    akeys = np.array( keys )
    strains = []
    directions = []
    all_omegas = []
    dtys = []
    all_tths = []
    all_etas = []
    weights = []
    all_intensity = []
    all_sc = []
    all_Gws = []
    all_hkl = []

    for refi,u in enumerate(uni):

        # h==h, k==k, l==l, sign==sign
        mask = (akeys == u).astype(int).sum(axis=1) == 4
        tths = flt.tth[g.mask][mask]
        etas = flt.eta[g.mask][mask]
        omegas = flt.omega[g.mask][mask]
        scs = flt.sc[g.mask][mask]
        detector_y_pos = flt.dty[ g.mask ][mask]
        intensity = flt.sum_intensity[g.mask][mask]
        scs = flt.sc[g.mask][mask]
        G_ws = np.array( (flt.gx[g.mask][mask], flt.gy[g.mask][mask], flt.gz[g.mask][mask]) ).T
        h = u[0]
        k = u[1]
        l = u[2]

        for sc, dty, tth, eta, om, I, sc, G_w in zip( scs, detector_y_pos, tths, etas, omegas, intensity, scs, G_ws ):

            all_hkl.append( [h,k,l] )
            bragg_order, d_measured, d_original, eps = strain(B, h, k, l, tth, wavelength)
            strains.append( eps )
            directions.append( normal_omega(tth, eta, wavelength, om) )
            all_omegas.append( om )
            all_tths.append( tth )
            all_etas.append( eta )
            dtys.append( dty )
            all_intensity.append( I )
            all_sc.append( sc )
            all_Gws.append(G_w)
            weights.append( weight( d_measured, d_original, eps, tth, wavelength, bragg_order, distance, pixelsize ) )

    return np.array(strains), np.array(directions), np.array(all_omegas), np.array(dtys), np.array(weights), np.array(all_tths), np.array(all_etas), np.array(all_intensity), np.array(all_sc), np.asarray(all_Gws), np.asarray(all_hkl)
