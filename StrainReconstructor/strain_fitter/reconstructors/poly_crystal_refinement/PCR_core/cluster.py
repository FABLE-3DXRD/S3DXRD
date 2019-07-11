'''
Define a module for 'cluster' (not really clustering, more like grouping) analysis of delta peak set
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
from matplotlib import rc
import random
import time
import itertools
import copy
#################################

def verify_peak_pair(p1,p2,dim,tol):
    if not abs(p1[dim] - p2[dim]) <= tol:
        return False
    else:
        return True

def verify_cluster(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    for dim,tol in zip([index_z, index_y, index_om],[tol_z,tol_y,tol_om]):
        cluster.sort(key=lambda x: x[dim])
        for i in range(len(cluster)-1):
            p1 = cluster[i]
            p2 = cluster[i+1]
            if not verify_peak_pair(p1,p2,dim,tol):
                # print_cluster(cluster)
                # print("")
                # print(p1)
                # print(p2)
                return False
    return True

def verify_many_clusters(clusters,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    for cluster in clusters:
        if not verify_cluster(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om):
            return False
    return True

def cluster(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    '''
    Take a set of peaks from a framestack and cluster them according
    to the dimensions and tolarences given a input. This will run
    cluster_in_z_y_omega() as many times as is needed until all clusters satisfy
    the condition of tol_z,tol_y and tol_om.
    '''
    clusters = cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)
    i = 0
    #print("Clustering iteration nbr: ",i)
    while( not verify_many_clusters(clusters,index_z,index_y,index_om,tol_z,tol_y,tol_om) ):
        c_new = []
        i += 1
        print("Clustering iteration nbr: ",i)
        if i>10:
            raise
        for cluster in clusters:
            for c in cluster_in_z_y_omega(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om):
                #print_cluster( c )
                c_new.append( c )
        clusters = c_new
    return clusters


#################################

def merge_single_cluster(peak_cluster,index_z,index_y,index_om,index_int):
    '''
    Take a list of delta peaks which belong to a cluster and return their
    merged represenattion. I.e their cms and their total intensity.
    '''
    merged_peak = copy.copy(peak_cluster[0])

    #DEBUG
    index_h = 5
    index_k = 6
    index_l = 7
    #for i,peak in enumerate(peak_cluster):
        # if peak[index_h]!=merged_peak[index_h] or \
        #    peak[index_k]!=merged_peak[index_k] or \
        #    peak[index_l]!=merged_peak[index_l]:
        #     print("used for merged peak:",merged_peak)
        #     print("----------")
        #     print("peak_cluster")
        #     print(peak[i-2:i+3])
        #     raise
    #---

    cms_z=cms_y=cms_om=summed_intensity=0
    summed_intensity = 0

    # # check that the same voxel don't appear twice
    # voxel_ids=[]
    # for peak in peak_cluster:
    #     voxel_ids.append(peak[8])
    # if len(voxel_ids)!=len(np.unique(voxel_ids)):
    #     raise


    for peak in peak_cluster:
        intensity = peak[index_int]
        summed_intensity += intensity
        cms_z += peak[index_z]*intensity
        cms_y += peak[index_y]*intensity
        cms_om += peak[index_om]*intensity
    cms_z = float(cms_z/summed_intensity)
    cms_y = float(cms_y/summed_intensity)
    cms_om = float(cms_om/summed_intensity)

    merged_peak[index_z] = cms_z
    merged_peak[index_y] = cms_y
    merged_peak[index_om] = cms_om
    merged_peak[index_int] = summed_intensity


    return merged_peak

def merge_many_clusters(peak_clusters,index_z,index_y,index_om,index_int):
    '''
    Take a list of delta peak clusters and return their respective
    merged peak represenation.
    '''
    filtered_peak_clusters = []
    merged_peak_list = []
    for cluster in peak_clusters:
        merged_peak = merge_single_cluster(cluster,index_z,index_y,index_om,index_int)
        #Remove low intensity peaks
        if merged_peak[index_int]>0.000001:
            merged_peak_list.append(merged_peak)
            filtered_peak_clusters.append( cluster )
    return merged_peak_list, filtered_peak_clusters


def cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    '''
    Cluster peaks in frame stack space
    '''
    clustered_peaks=[]
    groups_z = cluster_in_one_dimension(peaks,index_z,tol_z) #[ [],[],[] ]
    for group_z in groups_z:
        groups_z_y = cluster_in_one_dimension(group_z,index_y,tol_y) # [[],[],[]]
        for group_z_y in groups_z_y:
            groups_z_y_om = cluster_in_one_dimension(group_z_y,index_om,tol_om)
            for group_z_y_om in groups_z_y_om:
                clustered_peaks.append(group_z_y_om)
    return clustered_peaks


# Bad: one hkl can still have several orders of reflection
# And how are we to know if one order of reflections can interfere
# with a different order at a different hkl
# def cluster_by_hkl(peaks, index_h, index_k, index_l):
#     '''
#     Cluster peaks based on their (h, k, l) plane of reflection
#     this can perhaps get around the problem of discretization
#     where peaks get split on the detector due to strain gradients
#     nor being perfectly continous. Therefore this is an approximation
#     where we say that we expect the peak of a certain hkl to always be
#     continous, else the grain is breaking appart.
#     '''
#     peaks = sorted(peaks, key=lambda x: (x[index_h], x[index_k], x[index_l]))

#     groups=[]
#     curr_group=[]
#     curr_group.append(peaks[0])
#     for i in range(len(peaks)-1):
#         if abs(peaks[i][index_h]-peaks[i+1][index_h])<0.01 and \
#            abs(peaks[i][index_k]-peaks[i+1][index_k])<0.01 and \
#            abs(peaks[i][index_l]-peaks[i+1][index_l])<0.01:
#             curr_group.append(peaks[i+1])
#         else:
#             groups.append(curr_group)
#             curr_group = []
#             curr_group.append(peaks[i+1])

#     if len(curr_group)!=0:
#       groups.append(curr_group)

#     return groups

def cluster_in_one_dimension(peaks,index_dimension,tol):
    '''
    Cluster list of peaks in groups by variable found in index_dimension

        * Takes a list of peak events, each peak is a list of numerical values
        * Sorts the peaks by index_dimension:th colon
        * Bases groups upon the tolerated distance, tol, in index_dimension dimension

    Input: peaks = [ [peak1], [peak2], [peak3] ... etc ]
           index_dimension = colon index of peak list to sort by e.g. 0,1 ..
           tol = maximum deviation between peaks to cluster them

    Output: list of groups wich are close enough to form a cluster
    '''

    peaks.sort(key=lambda x: x[index_dimension])
    groups=[]
    curr_group=[]
    curr_group.append(peaks[0])
    for i in range(len(peaks)-1):
        if abs(peaks[i][index_dimension]-peaks[i+1][index_dimension])<=tol:
            curr_group.append(peaks[i+1])
        else:
            groups.append(curr_group)
            curr_group = []
            curr_group.append(peaks[i+1])

    if len(curr_group)!=0:
      groups.append(curr_group)

    return groups







#DEBUG functions below
#--------------------------------------------
def plot_time():
    tolx=5
    toly=5
    tolz=5
    t=[]
    ns=[10000,20000,40000,70000,100000,170000,250000]
    for n in ns:

        sq = np.round(np.sqrt(n))
        peaks = generate_test_data(int(sq),int(np.round(n/sq)),tolx)
        t1 = time.clock()
        clustered_peaks = cluster_in_z_y_omega(peaks, 0, 1, 2, tolx, toly, tolz)
        merged_peaks = merge_many_clusters(clustered_peaks, 0, 1, 2, 3)
        t2 = time.clock()
        t.append(t2-t1)
        print("Time for N peaks = ", n," time = ",np.round((t2-t1),4),"s")

    fig, ax = plt.subplots()
    ax.plot(ns,t,'-^',c='blue')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Time consumption 3D clustering')
    ax.set_xlabel('Number of peaks')
    ax.set_ylabel('Time [s]')



def generate_test_data(nbr_clusters,cluster_size,spread):
    peaks=[]
    for j in range(nbr_clusters):
        x=random.randint(0,100)
        y=random.randint(0,100)
        z=random.randint(0,100)
        #i=random.randint(0,10)
        Int=10
        for i in range(cluster_size):
            peak = [x+random.randint(-spread,spread), \
                    y+random.randint(-spread,spread), \
                    z+random.randint(-spread,spread), \
                    Int]
            peaks.append(peak)
    return peaks

def plot_unclustered_data(peaks,):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in peaks:
        ax.scatter(p[0],p[1],p[2],marker='o',c='purple')
    ax.set_title('Original Data')

def plot_clustered(clustered_peaks,index_z,index_y,index_om,om_range):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = itertools.cycle(["r", "b", "g", "y","pink","purple","orange"])
    markers= itertools.cycle(['^','o','s','*'])
    i=0
    for g in clustered_peaks:
        i+=1
        c = next(colors)
        m = next(markers)
        for p in g:
            if p[index_om]>om_range[0] and p[index_om]<om_range[1]:
                ax.scatter(p[index_y],p[index_om],p[index_z],marker=m,c=c)
                ax.text(g[0][index_y],g[0][index_om],g[0][index_z],str(i),size=20, zorder=1, color='k')
    ax.set_title('Clustered Data')
    ax.set_xlabel('y')
    ax.set_ylabel('omega')
    ax.set_zlabel('z')
    plt.show()

def plot_merged_data(merged_peaks):
    fig = plt.figure()
    colors = itertools.cycle(["r", "b", "g", "y","pink","purple","orange"])
    markers= itertools.cycle(['^','o','s','*'])
    ax = fig.add_subplot(111, projection='3d')
    i=0
    for p in merged_peaks:
        i+=1
        c = next(colors)
        m = next(markers)
        ax.text(p[0],p[1],p[2],str(i),size=20, zorder=1, color='k')
        ax.scatter(p[0],p[1],p[2],marker=m,c=c)
        #print(p[0],p[1],p[2])
    ax.set_title('Merged Data')


def plot_stuff():
    plot_time()
    peaks = generate_test_data(10,10,3)
    peaks.append([-20,-20,-20, 10])
    peaks.append([-10,-10,0,20])
    peaks.append([-10,-8,0,10])
    plot_unclustered_data(peaks)
    clustered_peaks = cluster_in_z_y_omega(peaks,0,1,2,3,3,3)
    plot_clustered(clustered_peaks)
    merged_peaks = merge_many_clusters(clustered_peaks,0,1,2,3)
    plot_merged_data(merged_peaks)
    plt.show()
