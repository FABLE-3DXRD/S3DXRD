'''
Profiler for ModelScanning3DXRD.py
Runs a series of simulations and times them.
Fits a linear function to the resulting datapoints
'''

from modelscanning3DXRD import gomodelscanning3DXRD
import voxelatedGrain
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as n
import os

grain_sizes = n.array([3,5,7,9])
voxel_size = 1
print_input=None
filename = "<insert input file path>"
killfile = None
debug = None
parallel = None

if not os.path.exists('<insert file path>/profileReports/'):
    os.makedirs('<insert input file path>/profileReports/')

cumtime = n.zeros((1,len(grain_sizes)))
for i,grain_size in enumerate(grain_sizes):
    no_voxels = grain_size
    profile = "profileReports/profileReport_"+str(no_voxels)+"x"+str(no_voxels)+".txt"
    p = voxelatedGrain.profiler()
    p.makeAsample(no_voxels,voxel_size)
    gomodelscanning3DXRD.run(print_input,filename,killfile,profile,debug,parallel)

    # Hack to get the cumtime for each run
    with open(profile, 'r') as f:
         report = f.readlines()
    line = report[0].split(" ")
    tottime = line[-2]
    cumtime[0,i]=tottime

N = grain_sizes**2 #number of voxels
T = cumtime[0,:] #Time spent computing
coef = n.polyfit(N, T, 1) #try to fit a linear function to data
x = n.linspace(0,N[-1]+10,N[-1]+10)
y = coef[0]*x+coef[1]

plt.figure()
plt.plot(N, T, 'ko', label="Original Noised Data")
plt.plot(x, y, 'r-', label="Fitted Curve y=kx+m, "+"k = "+str(n.round(coef[0],decimals=4)))
plt.legend()
plt.title("Time Consumption ModelScanning3DXRD: Square sample")
plt.xlabel("N [number of voxels]")
plt.ylabel("Time [seconds]")
plt.savefig("profileReports/TimePlot.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
plt.show()





