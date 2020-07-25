import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import ticker, cm
from matplotlib import rc
import matplotlib as mpl


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def plot_field( mesh, element_values, labels=None, title=None, vmin=None, vmax=None, save=None  ):
    '''
    Creates a plot of a scalar field over the mesh as a color plot. The values of
    each element is constantly given by element_vaues.

    mesh - a mesh numpy array, each row is an element with 3 nodes ([x1, y1, x2, y2, x3, y3])
    element_values - The values to be plotted, one for each element (numpy array)
    '''
    fig, ax = plt.subplots(figsize=(10,8))
    ev = element_values*(10**4)
    patches = []
    for i,elm in enumerate(mesh):
        if len(elm)==6:
            ax.plot([elm[0],elm[2]],[elm[1],elm[3]],'ro-')
            ax.plot([elm[0],elm[4]],[elm[1],elm[5]],'ro-')     
            ax.plot([elm[2],elm[4]],[elm[3],elm[5]],'ro-')
            xy = np.array([elm[0:2], elm[2:4], elm[4:6] ])
        if len(elm)==8:
            ax.plot([elm[0],elm[2]],[elm[1],elm[3]],'ro-')
            ax.plot([elm[2],elm[4]],[elm[3],elm[5]],'ro-')
            ax.plot([elm[4],elm[6]],[elm[5],elm[7]],'ro-')
            ax.plot([elm[6],elm[0]],[elm[7],elm[1]],'ro-')
            xy = np.array([elm[0:2], elm[2:4], elm[4:6], elm[6:8] ])
        polygon = Polygon(xy, True)
        patches.append(polygon)
        ax.text(np.mean(elm[0::2]),np.mean(elm[1::2]),str(i),size=17)
        if labels is not None:
           x = (elm[0]+elm[2]+elm[4])/3.
           y = (elm[1]+elm[3]+elm[5])/3.
           plt.text(x,y,str(i),size=17)

    p = PatchCollection(patches, alpha=1.0)
    if vmin is not None: ev[ev<vmin]=vmin
    if vmax is not None: ev[ev>vmax]=vmax
    p.set(array=ev, cmap='jet')
    ax.add_collection(p)
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.set_title(r'$10^{-4}$', size=27)
    cbar.ax.tick_params(labelsize=17)
    ax.tick_params(labelsize=17)

    l = 0.3*np.abs(  np.max( mesh ) )
    ax.text(l*1.03, l/10., "x",size=27)
    ax.quiver(0, 0, l, 0,angles='xy', scale_units='xy', scale=1)
    ax.text(-l/5., 1.05*l, "y",size=27)
    ax.quiver(0, 0, 0, l,angles='xy', scale_units='xy', scale=1)
    ax.axis('equal')
    if title is not None:
        ax.set_title(title, size=37)
    if save is not None: plt.savefig(save)



def plot_mesh( mesh, intersect=None, beam_width=None, labels=None ):
    '''
    Creates a plot of a mesh, with the option of illustrating beam intersection.

    mesh - a mesh numpy array, each row is an element with 3 nodes ([x1, y1, x2, y2, x3, y3])
    intersect - a subpart of mesh, highligted with filled elements (same data type as mesh)
    beam_width - plot beam in dahsed lines. (scalar)
    labels - plot element numbers (numpy array)

    '''
    fig, ax = plt.subplots()

    for i,elm in enumerate(mesh):
        if len(elm)==6:
            ax.plot([elm[0],elm[2]],[elm[1],elm[3]],'ro-')
            ax.plot([elm[0],elm[4]],[elm[1],elm[5]],'ro-')     
            ax.plot([elm[2],elm[4]],[elm[3],elm[5]],'ro-')
        if len(elm)==8:
            ax.plot([elm[0],elm[2]],[elm[1],elm[3]],'ro-')
            ax.plot([elm[2],elm[4]],[elm[3],elm[5]],'ro-')
            ax.plot([elm[4],elm[6]],[elm[5],elm[7]],'ro-')
            ax.plot([elm[6],elm[0]],[elm[7],elm[1]],'ro-')

        if labels is not None:
            x = (elm[0]+elm[2]+elm[4])/3.
            y = (elm[1]+elm[3]+elm[5])/3.
            plt.text(x,y,str(i),size=17)

    l = 2#1.1*np.abs(  np.max( mesh ) )
    ax.text(l, 0, "x",size=20)
    ax.quiver(-1, 0, l, 0,angles='xy', scale_units='xy', scale=1)
    ax.text(0, l, "y",size=20)
    ax.quiver(0, -1, 0, l,angles='xy', scale_units='xy', scale=1)
    ax.axis('equal')

    if intersect is not None:
        # paint all elements graced by the beam yellow
        patches = []
        for elm in intersect:
            xy = np.array([elm[0:2], elm[2:4], elm[4:6] ])
            polygon = Polygon(xy, True)
            patches.append(polygon)
        p = PatchCollection(patches, alpha=0.4, facecolor='y')
        ax.add_collection(p)

    if beam_width is not None:
        # add dashed lines defining the beam boundaries
        ax.axhline(beam_width/2.,color='k',linestyle='dashed')
        ax.axhline(-beam_width/2.,color='k',linestyle='dashed')
        ax.text(np.min(mesh)-15., -beam_width/3., "x-ray",size=1.5*beam_width)
