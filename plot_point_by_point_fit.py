
import sys
import pylab as pl, numpy as np

stem = sys.argv[1]

lines  = [ line for line in open(stem+".fit").readlines() ]
tokens = [ line.split() for line in lines if (len(line.split())==13 and line.find("Grain")<0)]
ig = np.array( [ int(t[0]) for t in tokens ] )
ix = np.array( [ int(t[1]) for t in tokens ] )
iy = np.array( [ int(t[2]) for t in tokens ] )
iN = np.array( [ int(t[3]) for t in tokens ] )
cpr = np.array( [[ float(v) for v in t[4:] ] for t in tokens] )

grains = sorted(list(set(ig)))

# move to an all positive origin so ix, iy can be array indices
ix -= ix.min()
iy -= iy.min()
# start with -1 everywhere
gmap = np.zeros( ( ix.max()+1, iy.max()+1), np.int )-1
nmap = np.zeros( ( ix.max()+1, iy.max()+1), np.int )-1
pmaps = [ np.zeros( ( ix.max()+1, iy.max()+1), np.int )*np.nan
          for p in range(18) ]

pfuncs = [
    lambda a :  (a - a.mean())/a.mean(),
    lambda b :  (b - b.mean())/b.mean(),
    lambda c :  (c - c.mean())/c.mean(),
    lambda d :  np.sin(np.radians( 90 - d ) ),
    lambda d :  np.sin(np.radians( 90 - d ) ),
    lambda d :  np.sin(np.radians( 90 - d) ),
    lambda r :  (r - r.mean()),
    lambda r :  (r - r.mean()),
    lambda r :  (r - r.mean()),
    ]
    
for i in grains:
    m = (ig == i)
    current = nmap[ix[m], iy[m]]
    me = iN[m] > current
    gmap[ ix[m][me],iy[m][me] ] = i
    nmap[ ix[m][me],iy[m][me] ] = iN[m][me]    
    myvals = cpr[m][me]
    for j,v in enumerate(myvals.T):
        pmaps[j*2  ][ix[m][me],iy[m][me]] = v
        pmaps[j*2+1][ix[m][me],iy[m][me]] = pfuncs[j](v)
    
gmap = np.where(gmap < 0, np.nan, gmap)
gmap = np.where(nmap < 0, np.nan, gmap)

pl.rcParams['figure.figsize']=(24,16)
pl.rcParams['savefig.dpi']=100

pl.figure(1)
pl.imshow(gmap, interpolation='nearest')
pl.title("Grain ID")
pl.savefig(stem+"_grain_id.png")
pl.colorbar()

pl.figure(2)
pl.imshow(nmap, interpolation='nearest')
pl.title("Number of Peaks")
pl.colorbar()
pl.savefig(stem+"_npks.png")

titles = 'a-axis a-strain b-axis b-strain c-axis c-strain alpha sin(90-alpha) beta sin(90-beta) gamma sin(90-gamma) rod0 rod0-mean rod1 rod1-mean rod2 rod2-mean'.split()
for i in range(9):
    pl.figure(3)
    pl.subplot(3,3,i+1)
    pl.imshow(pmaps[2*i], interpolation='nearest')
    pl.title("absolute "+titles[2*i])
    pl.colorbar()
    pl.figure(4)
    pl.subplot(3,3,i+1)
    pl.imshow(pmaps[2*i+1], interpolation='nearest')
    pl.title("relative "+titles[2*i+1])
    pl.colorbar()
pl.figure(3)
pl.savefig(stem+"_cell_rod.png")
pl.figure(4)
pl.savefig(stem+"_cell_rod_rel.png")
