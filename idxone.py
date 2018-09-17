
from __future__ import print_function, division

import sys, os
import numpy as np, pylab as pl
from ImageD11 import grid_index_parallel, columnfile, parameters, unitcell, grain
import multiprocessing

help = """You need to edit the script to set gridpars
Also define the grid below if name == main
"""


"""
Ring assignment array shape (4576,)
Ring     (  h,  k,  l) Mult  total indexed to_index  ubis  peaks_per_ubi
Ring 10  ( -4, -4,  0)   12    196      15      181    16     11
Ring 9   ( -3, -3, -3)   32    581      43      538    18     31
Ring 8   ( -4, -2, -2)   24    482      36      446    20     23
Ring 7   ( -4, -2,  0)   24    419      39      380    17     23
Ring 6   ( -3, -3, -1)   24    494      34      460    20     23
Ring 5   ( -4,  0,  0)    6    178      11      167    29      5
Ring 4   ( -2, -2, -2)    8    176      15      161    22      7
Ring 3   ( -1, -1, -3)   24    706      44      662    29     23
Ring 2   ( -2, -2,  0)   12    583      22      561    48     11
Ring 1   ( -2,  0,  0)    6    205      12      193    34      5
Ring 0   ( -1, -1, -1)    8    554      14      540    69      7
"""
gridpars = {
    'DSTOL' : 0.005,
    'OMEGAFLOAT' : 0.501 ,
    'COSTOL' : 0.004,
    'NPKS' : 100,
    'TOLSEQ' : [ 0.03, 0.015 ],
    'SYMMETRY' : "cubic",
    'RING1'  : [5,4,1,0],
    'RING2' : [5,4,1,0],
    'NUL' : True,
}



def select_x_y( cf, x, y, step=0.25):
    """
    Choose the peaks corresponding to point x,y
    """
    so = np.sin( np.radians( cf.omega ) )
    co = np.cos( np.radians( cf.omega ) )
    dty_calc_int = np.round((x*co + y*so)/step).astype(int)
    dtyi = np.round( cf.dty/step ).astype(int)
    icf = cf.copy()
    icf.filter( dtyi == dty_calc_int )
    return icf


def do_xy(a):
    x,y  = a
    icf = select_x_y( flt, x, y, step=0.25)
    gve = np.vstack(( icf.gx, icf.gy, icf.gz ))
    grains = grid_index_parallel.doindex( gve, 0, 0, 0, wvln, gridpars )
    l1 = len(grains)
    grains = grid_index_parallel.domap( pars, icf, grains, gridpars )
    for g in grains:
        g.translation=[x,y,0]
    print(len(grains), l1, x,y)
    return grains
    
    


 
if __name__=="__main__":
    flt  = columnfile.columnfile( sys.argv[1] )

    flt.filter( flt.Number_of_pixels > 4 )
    flt.filter( flt.tth < 20 )
    
    
    flt.addcolumn(np.zeros(flt.nrows)-1,"labels")
    flt.addcolumn(np.zeros(flt.nrows)-1,"drlv2")
    pars = parameters.read_par_file( sys.argv[2] )
    
    gridpars[ 'UC' ] = unitcell.unitcell_from_parameters( pars )

    wvln = pars.get("wavelength")
    pars.stepsizes['t_x']=1
    pars.stepsizes['t_y']=1
    pars.stepsizes['t_z']=1
    xy = []
    


    for x in np.arange(-0.31, 0.3101, 0.001):
        for y in np.arange(-0.31, 0.3101, 0.001):
            if x*x + y*y > 0.31*0.31:
                continue
            xy.append( (x,y) )

    with open(sys.argv[3], "w") as resultsfile:
        resultsfile.write("#  x  y  z  UBI00  UBI01  UBI02  UBI10  UBI11  UBI12  UBI20  UBI21  UBI22  npks  nuniq\n")
        NPR = int(multiprocessing.cpu_count() )
        p = multiprocessing.Pool( processes=NPR )
        for glist in p.imap_unordered( do_xy, xy):
            for g in glist:
                resultsfile.write(("%.8f  "*12+"%d  %d\n")%tuple(
                    list(g.translation) + list(g.ubi.ravel()) + [ g.npks, g.nuniq ] ) )
            resultsfile.flush()
    
