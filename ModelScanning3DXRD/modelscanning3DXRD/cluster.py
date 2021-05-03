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



def merge_single_cluster(peak_cluster,index_z,index_y,index_om,index_int,index_nbr_of_pixels):
    '''
    Take a list of delta peaks which belong to a cluster and return their
    merged represenattion. I.e their cms and their total intensity.
    '''
    # for peak in peak_cluster:
    #     if peak[3]!=peak_cluster[0][3] or \
    #     peak[4]!=peak_cluster[0][4] or \
    #     peak[5]!=peak_cluster[0][5]:
    #         print("peak: ",peak)
    #         print("peak 0 ",peak_cluster[0])
    #         raise
    merged_peak = copy.copy(peak_cluster[0])
    cms_z=cms_y=cms_om=summed_intensity=0
    summed_intensity = 0
    unique=[]
    for peak in peak_cluster:
        coord = [peak[index_z],peak[index_y],peak[index_om]]
        if coord not in unique:
            unique.append(coord)
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

    merged_peak[index_nbr_of_pixels] = len(unique)
    # if merged_peak[index_nbr_of_pixels]>50:
    #     print("merged_peak ",merged_peak)
    #     print("")
    #     print("merged from: ")
    #     print("detz    dety    omega    dty")
    #     for peak in peak_cluster:
    #         print(peak[index_z],peak[index_y],peak[index_om],peak[21])
    #     raise
    return merged_peak

def merge_many_clusters(peak_clusters,index_z,index_y,index_om,index_int,index_nbr_of_pixels):
    '''
    Take a list of delta peak clusters and return their respective
    merged peak represenation.
    '''
    # print("len clusters",len(peak_clusters))
    # for cluster in peak_clusters:
    #     z = cluster[0][index_z]
    #     y =  cluster[0][index_y]
    #     for peak in peak_clusters:
    #         print(peak)
    #         raise
    #         if abs(peak[index_z]-z)>50 or abs(peak[index_y]-y)>50:
    #             print(cluster[0][index_z],cluster[0][index_y],cluster[0][index_om],cluster[0][21])
    #             print(peak[index_z],peak[index_y],peak[index_om],peak[21])
    #             raise

    merged_peak_list = []
    for cluster in peak_clusters:
        merged_peak = merge_single_cluster(cluster,index_z,index_y,index_om,index_int,index_nbr_of_pixels)
        #Remove low intensity peaks
        if np.round(merged_peak[index_int],6)!=0:
            merged_peak_list.append(merged_peak)
    return merged_peak_list


def cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    '''
    Cluster peaks in frame stack space
    '''
    clustered_peaks=[]
    mergedpks=0
    curr=0
    groups_z = cluster_in_one_dimension(peaks,index_z,tol_z)
    for group_z in groups_z:
        groups_z_y = cluster_in_one_dimension(group_z,index_y,tol_y)
        for group_z_y in groups_z_y:
            groups_z_y_om = cluster_in_one_dimension(group_z_y,index_om,tol_om)
            # if len(group_z_y)>50:
            #         for peak in group_z_y:
            #             print(peak[index_z],peak[index_y],peak[index_om],peak[21])
            #             print(index_z,index_y,index_om)
            #         raise
            for group_z_y_om in groups_z_y_om:
                clustered_peaks.append(group_z_y_om)
    #print("len(clustered_peaks)",len(clustered_peaks))
    #plot_clustered(clustered_peaks,index_z,index_y,index_om,[0, 90*(np.pi/180)])
    return clustered_peaks


# def cluster_in_one_dimension(peaks,index_dimension,tol,prev_dim=None,prev_tol=None):

#     groups=[]
#     curr_group=[]
#     p = copy.deepcopy(peaks)

#     for peak in p:
#         if len(np.asarray(peak).shape)!=1 or np.asarray(peak).shape[0]!=23:
#             print("index_dimension",index_dimension)
#             print(peak)
#             raise

#     if prev_dim!=None and prev_tol!=None and prev_dim==10:
#         for i in range(len(p)-1):
#             if abs(p[i][prev_dim]-p[i+1][prev_dim])>=prev_tol:
#                 for peak in p:
#                     print(peak[10],peak[9],peak[7],peak[21])
#                 raise

#     for i,peak in enumerate(p):
#         if i<10:
#             print(peak[index_dimension])

#     p.sort(key=lambda x: x[index_dimension])
#     curr_group.append(copy.deepcopy(p[0]))

#     print("")
#     for i,peak in enumerate(p):
#         if i<10:
#             print(peak[index_dimension])


#     for i in range(len(p)-1):
#         if p[i][index_dimension]!=curr_group[-1][index_dimension]:
#             print("i",i)
#             print("p[i][index_dimension]",p[i][index_dimension])
#             print("curr_group[-1][index_dimension]",curr_group[-1][index_dimension])
#             raise
#         if abs(p[i][index_dimension]-p[i+1][index_dimension])<=tol:
#             curr_group.append(copy.deepcopy(p[i+1]))
#         else:
#             groups.append(copy.deepcopy(curr_group))
#             curr_group = []
#             curr_group.append(copy.deepcopy(p[i+1]))
#         if i>0:
#             for j in range(len(curr_group)-1):
#                 if abs(curr_group[j][index_dimension]-curr_group[j+1][index_dimension])>tol:
#                     print(curr_group)
#                     print("i ",i)
#                     print("len(p) ",len(p))
#                     print("j ",j)
#                     print("len(curr_group) ",len(curr_group))
#                     print(curr_group[j][index_dimension])
#                     print(curr_group[j+1][index_dimension])
#                     raise

#     if len(curr_group)!=0:
#       groups.append(copy.deepcopy(curr_group))

#     return copy.deepcopy(groups)


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
        # if len(np.asarray(peaks[i]).shape)!=1 or np.asarray(peaks[i]).shape[0]!=23:
        #     print("index_dimension",index_dimension)
        #     print(peaks[i])
        #     raise
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

def plot_unclustered_data(peaks):
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

# nbr_clusters = 10
# cluster_size = 10
# spread = 2
# index_z = 0
# index_y = 1
# index_om = 2
# tol_z=2
# tol_y=2
# tol_om=2
# om_range = [-100,100]
# peaks = generate_test_data(nbr_clusters,cluster_size,spread)
# #peaks_by_z = cluster_in_one_dimension(peaks,index_z,tol_z)

# clustered_peaks = cluster_in_z_y_omega(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)
# for cluster in clustered_peaks:
#     for peak in cluster:
#         print(peak[index_z])
#     print("")
# plot_unclustered_data(peaks)
# plot_clustered(clustered_peaks,index_z,index_y,index_om,om_range)
# plt.show()