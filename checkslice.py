
from __future__ import print_function, division
import h5py, sys, numpy as np, pylab as pl
from ImageD11 import columnfile
import dorecon_grain

cc = dorecon_grain.colfilecache(parfile='fit.par')

""" Check the grains do not share too many peaks

Report on which ones we like best

See what is remaining not indexed
"""

hf = h5py.File( sys.argv[1], 'r' )
grain_names = hf.keys()

masks = {}
colf = None

print("Name  npks  <sum_intensity>  <avg_intensity>  <npixels>")

ng = len(grain_names)
grain_names = ["grain_%d"%(i) for i in range(ng)]

for i,name in enumerate(grain_names):
    if colf is None:
        colf = cc.get( hf[name].attrs['pksfile'] )
    else:
        assert colf.filename == hf[name].attrs['pksfile'] 
    ids = hf[name]["pkid"][:]
    m = np.zeros( colf.nrows, np.bool )
    m[ids] = True
    masks[name]=m
    print(name, len(ids), colf.sum_intensity[ ids ].mean(),
          colf.avg_intensity[ids].mean(), colf.Number_of_pixels[ids].mean() )

cov = np.zeros( (ng,ng), np.float)
for i in range(len(grain_names)):
    m = masks[grain_names[i]]
    ms = m.sum()
    for j in range(i):
        mo = masks[ grain_names[j] ]
        clashes = (m & mo).sum()
        cov[i,j] = cov[j,i] = float(clashes) / ms
        if cov[i,j] > 0.2:
            print("i   j   nshare  percent percent")
            print("%d  %d  %d  %.2f  %.2f"%(i,j,clashes,clashes/m.sum(),clashes/mo.sum()))
            print("ubi[i]")
            print( hf[grain_names[i]]["ubi"][:] )
            print("ubi[j]")
            print( hf[grain_names[j]]["ubi"][:] )

pl.imshow(cov)
pl.title("Peak sharing between grains")
pl.show()


# Now see what is left globally 

allmasks = np.array( masks.values() )
not_indexed_peaks = allmasks.sum(axis=0) == 0

r0 = dorecon_grain.recon_all_peaks( colf )
r1 = dorecon_grain.recon_all_peaks( colf, mask = not_indexed_peaks )
pl.figure(2)
pl.subplot(121)
pl.imshow(r0)
pl.title("All peaks")
pl.subplot(122)
pl.imshow(r1)
pl.title("Left over peaks")

allsinos = np.array( [ hf[name]['sinogram'][:]
                       for name in grain_names ] )

allrecon = np.array( [ hf[name]['recon'][:]
                       for name in grain_names ] )
    # etc
