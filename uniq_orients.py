
from __future__ import print_function, division

from ImageD11 import sym_u, columnfile, grain
import sys, numpy as np, pylab as pl

symmetry = sys.argv[1]
grp = getattr( sym_u, symmetry )()
ubifile = sys.argv[2]
uniqfile = sys.argv[3]


if ubifile[-3:] == "flt":
    c = columnfile.columnfile( ubifile )
    print("Got c.nrows",c.nrows)
    ubis = np.zeros( (c.nrows, 3, 3), np.float) 
    for i in range(3):
        for j in range(3):
            ubis[:,i,j] = c.getcolumn( "UBI%d%d"%(i,j) )
    del c
else:
    c = grain.read_grain_file( ubifile )
    ubis = np.array( [g.ubi for g in c] )
    del c

ubis.shape = len(ubis), 9

# First is to cluster without symmetry

def removeone( ubis, testubi, tol=None):
    diffs = abs(ubis - testubi).sum(axis=1)
    if tol is None:
        pl.hist(diffs,bins=1024)
        pl.show()
        tol = float( input("Tolerance?"))
    matching = diffs < tol
    # iterate in case this was not the centre of the bunch
    ubi = ubis[matching].mean(axis=0)
    diffs = abs(ubis - ubi).sum(axis=1)
    matching = diffs < tol
    ubi = ubis[matching].mean(axis=0)
    remaining = ubis[~matching]
    return ubi, remaining, tol

ubiall = ubis.copy()
tol = None
uniqs = []
while len( ubis > 0 ):
    uniq, ubis, tol = removeone( ubis, ubis[0], tol )
    print(uniq, len(ubis))
    uniqs.append( uniq )

# now check if any uniqs have a collision due to symmetry:

uniqs = np.array( uniqs )

for i in range(len(uniqs)):
    for operator in grp.group[1:]: # skip the identity
        symubi = np.dot( operator, uniqs[i].reshape(3,3) ).ravel()
        scors = abs(uniqs - symubi).sum(axis=1)
        found = scors < tol
        if found.sum() > 0:
            print("Symmetry collision!")
            print( i, operator.ravel(),
                   scors[found], np.arange(len(uniqs))[found])

grain.write_grain_file( uniqfile, [ grain.grain( ubi.reshape(3,3), (0,0,0) )
                                    for ubi in uniqs ] )
