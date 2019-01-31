
from __future__ import print_function, division
# read the output from idx one and group it

import sys
import numpy as np, pylab as pl
from scipy.cluster.hierarchy import fcluster


try:
    import fastcluster
except:
    print("missing the fastcluster package ....")
    print("try 'pip install fastcluster --user'")
    print("or look at https://github.com/dmuellner/fastcluster/")
    sys.exit()

from ImageD11.columnfile import columnfile

import xfab.tools, xfab.symmetry


symmops = [tuple(m.ravel()) for m in
           abs(xfab.symmetry.rotations(6).astype(int))]
uniqs = list( set( symmops ) )
smask = [ np.array(u)==1 for u in uniqs ]
r = range(9)
sinds = np.array([ np.compress(r,u) for u in smask ])

for s in smask:
    print (s)

def misori( ub, ubi, debug=False ):
    if debug:
        print("ub=",repr(ub))
        print("ubi=",repr(ubi))
        t = xfab.symmetry.Umis( ub, ubi.T, 7 )
        print(t)
        print("best",xfab.symmetry.rotations(7)[np.argmin(t[:,1])])
    rmat = abs(np.dot(ub, ubi)).ravel()
    scors=[rmat[s].sum() for s in smask]
    #i = np.argmax(scors)
    tr = np.max(scors)
    # mat = (np.sign( rmat ).ravel() * uniqs[ i ]).reshape(3,3)
    # nrmat = np.dot( ub, np.dot(ubi, mat.T) )
    c = (tr - 1)/2
    # assert np.allclose( k, np.trace( nrmat ), 6)
    # ang = np.degrees(np.arccos(c))
    if debug:
        print("rmat",rmat)
        print(armat)
        print(scors)
        print(i)
        print( uniqs[i] )
        print(mat)
        print(nrmat)
    if 0:
        t = xfab.symmetry.Umis( ub, ubi.T, 7 )
        ans = t[:,1].min()
        if not np.allclose( ang , ans, 6):
            if debug:
                sys.exit()
            else:
                print(ang,ans)
                misori(ub, ubi, debug=True )
    return c

c = columnfile( sys.argv[1] )
c.filter( np.arange(c.nrows)%2 == 0)
c.filter( np.arange(c.nrows)%3 == 0)


ubi = [ c.getcolumn("UBI%d%d"%(i,j)) for i in range(3) for j in range(3) ]
ubi = np.array( ubi ).T
ubi.shape = ubi.shape[0], 3, 3

umats = [ xfab.tools.ubi_to_u( m ).T for m in ubi ]

print("Computing a distance matrix as misorientation angles in cubic (deg)")
need = np.cumsum( np.arange( 1,len(ubi) ) )[-1]
D = np.zeros( need, float )
k = 0
print("%8d%8d"%(0,len(ubi)),end="\r")
for i in range(len(ubi)):
    print("%8d"%(i),end="\r")
    sys.stdout.flush()
    for j in range(i+1,len(ubi)):
        D[k] = misori( umats[i].T, umats[j] )
        k += 1
D=np.degrees(np.arccos(D))
print()
print("Range of misorientation angles : ",D.min(),D.max())
print("Read your grains, clustering based on cubic misorientation")
Z = fastcluster.linkage( D,
                         method = "single",     # closest of any two pairs
                         preserve_input=True)

pl.plot( Z[::-1,2], "-o" )
pl.xlabel( "Number of grains")
pl.ylabel( "Misorientation (/2?)" )
pl.show()
tol = float( input("Enter cutoff: ") )
print(tol)
fc=fcluster( Z, float(tol), criterion='distance')                
print(fc)     

labels = np.unique( fc )
npx = [ (fc==j).sum() for j in labels ]
order = np.argsort( npx )

for j,i in enumerate(order[::-1]):
    l = labels[i]
    m = fc == l
    print("# %d %d npixels",j,i,npx[i])
    usel = ubi[m]
    avgubi = usel.mean(axis=0)
    stdubi = usel.std(axis=0)
    print( avgubi )
    print( stdubi )
