import numpy as np
from scipy.optimize import leastsq
from ImageD11 import parameters, grain, transform
import copy
import scanning_transform

class GrainFitter(object):

    def __init__(object):
        pass
    
    def fit_one_grain(self, gr, flt, pars ):
        """
        Uses scipy.optimize to fit a single grain
        """
        args = flt, pars, gr
        x0 = gr.ub.ravel().copy()
        xf, cov_v, info, mesg, ier = leastsq( self.calc_teo_fit, x0, args, full_output=True )
        ub = xf.copy()
        ub.shape = 3,3
        ubi = np.linalg.inv(ub)

        gr.set_ubi( ubi )
    
    def calc_teo_fit(self, ub, flt, pars, gr, return_concatenated = True):
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

        tthcalc, etacalc, omegacalc = self.calc_tth_eta_omega( UB, gr.hkl, pars, gr.etasigns )

        dtth   = ( flt.tth[ gr.mask ]   - tthcalc   ) * flt.wtth[ gr.mask ]
        deta   = ( flt.eta[ gr.mask ]   - etacalc   ) * flt.weta[ gr.mask ]
        domega = ( flt.omega[ gr.mask ] - omegacalc ) * flt.womega[ gr.mask ]
        if return_concatenated:
            return np.concatenate( (dtth, deta, domega) )
        else:
            return dtth, deta, domega

    def calc_tth_eta_omega( self, ub, hkls, pars, etasigns ):
        """
        Predict the tth, eta, omega for each grain
        ub = ub matrix (inverse ubi)
        hkls = peaks to predict
        pars = diffractometer info (wavelength, rotation axis)
        etasigns = which solution for omega/eta to choose (+y or -y)
        """
        g = np.dot( ub, hkls )
        #TODO: Here we need to update by CMS
        tthcalc, eta2, omega2 = transform.uncompute_g_vectors( g,  pars.get('wavelength'),
                                                            wedge=pars.get('wedge'),
                                                            chi=pars.get('chi') )
        # choose which solution (eta+ or eta-)
        e0 = np.sign(eta2[0]) == etasigns
        etacalc = np.where( e0, eta2[0], eta2[1] )
        omegacalc = np.where( e0, omega2[0], omega2[1] )
        return tthcalc, etacalc, omegacalc

    def get_peak_quantities(self, flt, pars, OMSLOP, origin=[0,0,0] ):
        """
        Compute twotheta, eta, g-vector for given origin = [x, y]
        obs: Computes for all peaks generated from all grains.

        If no origin specified origin is set to [0,0,0]
        """
        pars.parameters['t_x'] = np.ones(flt.nrows)*origin[0]
        pars.parameters['t_y'] = np.ones(flt.nrows)*origin[1]
        pars.parameters['t_z'] = np.zeros(flt.nrows)

        tth, eta = scanning_transform.compute_tth_eta( [flt.sc, flt.fc], omega=flt.omega, **pars.parameters )
        gve = transform.compute_g_vectors( tth, eta, flt.omega,
                                            pars.get('wavelength'),
                                            wedge=pars.get('wedge'),
                                            chi=pars.get('chi') )

        pars.parameters['t_x'] = 0
        pars.parameters['t_y'] = 0
        pars.parameters['t_z'] = 0

        return tth, eta, gve

    def estimate_weights( self, pars, flt, OMSLOP, g=None ):
        distance  = pars.get('distance')
        pixelsize = ( pars.get('y_size') + pars.get('z_size') ) / 2.0
        # 1 pixel - high energy far detector approximation
        if g:
            wtth = np.ones(np.sum(g.mask))/ np.degrees( pixelsize / distance )
            weta = wtth * np.tan( np.radians( flt.tth[ g.mask ] ) )
            womega = np.ones(np.sum(g.mask))/OMSLOP
        else:
            wtth = np.ones(flt.nrows)/ np.degrees( pixelsize / distance )
            weta = wtth * np.tan( np.radians( flt.tth ) )
            womega = np.ones(flt.nrows)/OMSLOP

        return wtth, weta, womega

    def get_grain_cms(self, grain, flt):
        """
        Find grain geometrical centre of mass from sinogram
        Derived from Azevedo et al. (1990) replacing g(tth,s)
        with a binary window.
        """
        dty = flt.dty[ grain.mask ]
        omega = np.radians( flt.omega[ grain.mask ] )
        omega_uniq = np.unique( omega )

        N = len( omega_uniq )
        co = np.cos( omega_uniq )
        so = np.sin( omega_uniq )

        # Az=y is matrix format of s'_r = c_s + x' cos(om_i) + y' sin(om_i)
        # The validity of this is proven in Azevedo et al. (1990)
        A = np.array( [np.ones(co.shape),co,so] ).T
        p = np.zeros( ( N, ) )


        int = flt.sum_intensity[ grain.mask ]
        for i,om in enumerate( omega_uniq ):
            omega_mask = (omega==om)
            s = dty[ np.where( omega_mask ) ]
            g = int[ np.where( omega_mask ) ]
            p[i] = np.dot(s,g)/np.sum(g)

        # Lest squares solution to Az=y, i.e z*=(A^TA)^-1A^Tp, z*=[cs x y]^T
        # here cs, x and y is as in Azevedo et al. (1990)
        cs, x, y = np.dot( np.linalg.inv( ( np.dot( A.T, A) ) ), np.dot( A.T, p ) )

        # convert to 3DXRD coordinates, retruns: cs, x, y in 3DXRD system.
        # print(y, -x)
        return cs, y, -x
