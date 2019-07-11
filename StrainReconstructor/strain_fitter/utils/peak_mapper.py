import numpy as np
import matplotlib.pyplot as plt
from ImageD11 import parameters, grain, transform
import scanning_transform
from grain_fitter import GrainFitter

class PeakMapper(object):

    def __init__(self ):
        self.grain_fitter = GrainFitter()

    def map_peaks(self, flt, grains, params, omslop, hkltol, nmedian, rcut, ymin, ystep, number_y_scans ):
        
        print('Mapping peaks to grains..')

        tth, eta, gve = self.initiate_cols( flt, params, omslop)

        # #
        # self.assign_peaks( grains, gve, flt, params, nmedian,  hkltol )
        # tth, eta, gve = self.initiate_cols(flt, params, omslop)
        # tth, eta, gve = self.grain_fitter.get_peak_quantities( flt, params, omslop)
        # #

        print('Initated columns')

        for i,gr in enumerate(grains):
            self.assign_peaks_to_grain( gr, gve, flt, params, nmedian,  hkltol )
        self.discard_overlaping_spots( grains, gve, flt, params, nmedian, hkltol )

        init_assigned_peaks=0
        for i,gr in enumerate(grains):
            init_assigned_peaks += np.sum(gr.mask)
        print('Performed first assignment based on input inverse UB matrices setting all grain centroids to (0,0,0)')
        print(str(init_assigned_peaks)+' peaks out of'+str(flt.nrows)+' where succesfully assigned')

        # Simple version
        #---------------------------------------------------------------------
        itr=0                   # current iteration number

        cs = np.zeros((len(grains,)))       # current grain centre of rotation
        x = np.zeros((len(grains,)))        # current grain centroid x xoordinates
        y = np.zeros((len(grains,)))        # current grain centroid y coordinates
        assigned_peaks = 0
        prev_assigned_peaks = 0

        print('')
        print('Starting grain centroid and UBI refinement for peak assignment')
        # Keep iterating until the change in assigned peaks   
        # is less than 1% of the total measured peak number.
        #while( itr<3 or (assigned_peaks-prev_assigned_peaks)>(flt.nrows/100.) ):
        for i in range(2):
            prev_assigned_peaks = assigned_peaks
            assigned_peaks = 0 

            # Compute current grain centroids and assign peaks
            for i,gr in enumerate(grains):
                cs[i], x[i], y[i] = self.grain_fitter.get_grain_cms(gr, flt)
                #print(cs[i], x[i], y[i])
                origin = [ x[i],y[i] ]
                tth, eta, gve = self.grain_fitter.get_peak_quantities( flt, params, omslop, origin )
                self.assign_peaks_to_grain( gr, gve, flt, params, nmedian,  hkltol )
            self.discard_overlaping_spots( grains, gve, flt, params, nmedian,  hkltol )

            # Update grain ubi based on the newly assigned peak sets
            for i,gr in enumerate(grains):
                assigned_peaks += np.sum(gr.mask)
                self.grain_fitter.fit_one_grain( gr, flt, params )

            print("Iteration ",itr," number of indexed peaks:", assigned_peaks, "of ", flt.nrows, "hkltol: ",hkltol)
            itr += 1
        #---------------------------------------------------------------------
        tth, eta, gve = self.update_cols_per_grain( flt, params, omslop, grains, x, y )



    def initiate_cols(self, flt, pars, OMSLOP, weights=True ):
        tth, eta, gve = self.grain_fitter.get_peak_quantities( flt, pars, OMSLOP)
        flt.addcolumn( tth  , "tth" )
        flt.addcolumn( eta , "eta" )
        flt.addcolumn( gve[0], "gx" )
        flt.addcolumn( gve[1], "gy" )
        flt.addcolumn( gve[2], "gz" )

        if weights:
            wtth, weta, womega = self.grain_fitter.estimate_weights( pars, flt, OMSLOP )
            flt.addcolumn( wtth, "wtth" )
            flt.addcolumn( weta, "weta" )
            flt.addcolumn( womega, "womega" )

        return tth, eta, gve
    
    def assign_peaks( self, grains, gve, flt, pars, nmedian, hkltol ):
        """
        Assign peaks to grains for fitting
        - each grain chooses the spots it likes
        - overlapping spots (chosen by more than 1 grain) are removed
        - fit outliers are removed abs(median err) > nmedian
        Fills in grain.mask for each grain
        """
        for i, g in enumerate(grains):
            hkl = np.dot( g.ubi, gve )
            hkli = np.round( hkl )
            drlv = hkli - hkl
            drlv2 = (drlv*drlv).sum(axis=0)
            g.mask = drlv2 < hkltol*hkltol
        self.discard_overlaping_spots( grains, gve, flt, pars, nmedian,  hkltol )

    def update_mask( self, mygr, flt, pars, nmedian ):
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
        tthcalc, etacalc, omegacalc = self.grain_fitter.calc_tth_eta_omega( mygr.ub, hkl, pars, etasigns )
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
    

    def assign_peaks_to_grain(self, gr, gve, flt, pars, nmedian, hkltol):
        """
        Assign peaks to grains for fitting
        - each grain chooses the spots it likes
        Fills in grain.mask for each grain
        """

        # For each grain we compute the hkl integer labels
        hkl = np.dot( gr.ubi, gve )
        hkli = np.round( hkl )
        # Error on these:
        drlv = hkli - hkl
        drlv2 = (drlv*drlv).sum(axis=0)
        # Tolerance to assign to a grain is rather poor
        gr.mask = drlv2 < hkltol*hkltol # g.mask is a boolean declaration of all peaks that can belong to grain g
        #raise KeyboardInterrupt

    def discard_overlaping_spots( self, grains, gve, flt, pars, nmedian, hkltol ):
        """
        Iterate over all grains and discard any spots choosen by more than one grain
        """

        overlapping = np.zeros( flt.nrows, dtype=bool )
        for i in range(len(grains)):
            for j in range(i+1,len(grains)):
                overlapping |= grains[i].mask & grains[j].mask

        for i, g in enumerate(grains):
            g.mask &= ~overlapping

            while 1:
                ret = self.update_mask( g, flt, pars, nmedian )
                if ret == 0:
                    break

    def update_cols_per_grain(self, flt, pars, OMSLOP, grains, x, y):
        """
        update the twotheta, eta, g-vector columns to be sure they are right
        fill in some weighting estimates for fitting
        """

        # Fill flt columns by iterating over the grains
        for i,gr in enumerate(grains):
            peak_pos = [flt.sc[gr.mask], flt.fc[gr.mask]]
            omega = flt.omega[ gr.mask ]

            pars.parameters['t_x'] = np.ones(np.sum(gr.mask))*x[i]#origin[0]
            pars.parameters['t_y'] = np.ones(np.sum(gr.mask))*y[i]#0 #origin[1]
            pars.parameters['t_z'] = np.zeros(np.sum(gr.mask))

            tth, eta = scanning_transform.compute_tth_eta( peak_pos, omega=omega, **pars.parameters )
            gve = transform.compute_g_vectors( tth, eta, omega,
                                                pars.get('wavelength'),
                                                wedge=pars.get('wedge'),
                                                chi=pars.get('chi') )
            pars.parameters['t_x'] = 0
            pars.parameters['t_y'] = 0
            pars.parameters['t_z'] = 0

            flt.tth[ gr.mask ] = tth
            flt.eta[ gr.mask ] = eta
            if flt.wtth.any(): 
                # Compute the relative tth, eta, omega errors ...
                wtth, weta, womega = self.grain_fitter.estimate_weights( pars, flt, OMSLOP, gr )
                flt.wtth[ gr.mask ] = wtth
                flt.weta[ gr.mask ] = weta
                flt.womega[ gr.mask ] = womega
            flt.gx[ gr.mask ] = gve[0]
            flt.gy[ gr.mask ] = gve[1]
            flt.gz[ gr.mask ] = gve[2]

        return flt.tth, flt.eta, np.array([flt.gx, flt.gy, flt.gz])