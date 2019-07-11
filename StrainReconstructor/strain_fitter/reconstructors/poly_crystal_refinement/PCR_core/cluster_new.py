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
    return clusters, i









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
            for group_z_y_om in groups_z_y_om:
                clustered_peaks.append(group_z_y_om)

    return clustered_peaks


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

# ('peak 1: ', [60, 44, 16, 10])
# ('peak 2: ', [57, 30, 17, 10])

    # print("Dimension: ",index_dimension)
    # print_many_clusters(groups)
    # print("")
    # print("-----------------------------")

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
    #for j in range(np.random.randint( np.round(nbr_clusters/2), nbr_clusters) ):
    for j in range( nbr_clusters ):
        y = np.random.rand()*2048
        z = np.random.rand()*2048
        om = np.random.rand()*np.pi
        #i=random.randint(0,10)

        #for i in range( np.random.randint( round(cluster_size/2), cluster_size) ):
        for i in range( cluster_size ):
            peak = [y+np.random.randint( -spread, spread ), \
                    z+np.random.randint( -spread, spread ), \
                    np.random.rand()*30000., \
                    om+np.radians( np.random.randint(-spread,spread)/12. )]
            for i in range(len(peak)):
                if peak[i]<0:
                    peak[i] = 0
            peaks.append(peak)
    random.shuffle( peaks )
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

def print_cluster(cluster):
    print("--------------------------------")
    for peak in cluster:
        print(peak)
    print("--------------------------------")

def print_many_clusters(peak_clusters):
    for i,cluster in enumerate(peak_clusters):
        print("Cluster No ",i )
        print_cluster( cluster )
        print("")

def check_peak_pair(p1,p2,dim,tol):
    if abs(p1[dim] - p2[dim]) <= tol:
        return True
    else:
        return False

def check_cluster(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    for dim,tol in zip([index_z,index_y,index_om],[tol_z,tol_y,tol_om]):
        cluster.sort(key=lambda x: x[dim])
    for i in range(len(cluster)-1):
        p1 = cluster[i]
        p2 = cluster[i+1]
        if not check_peak_pair(p1,p2,dim,tol):
            print_cluster(cluster)
            print("")
            print("These peaks do not belong:")
            print("peak 1: ",p1)
            print("peak 2: ",p2)
            raise

def basic_check_cluster(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    no_peaks = len( cluster )
    if no_peaks>1:
        neighbour_counts = [0]*no_peaks
        neighbours=[]

        for i,p1 in enumerate(cluster):
            ne=[]
            for j,p2 in enumerate(cluster):
                if j!=i:
                    if abs(p1[index_z] - p2[index_z]) <= tol_z and \
                       abs(p1[index_y] - p2[index_y]) <= tol_y and \
                       abs(p1[index_om] - p2[index_om]) <= tol_om:
                       neighbour_counts[i] += 1
                       ne.append(j)
            neighbours.append(ne)

        # check that no peaks are solo
        for n in neighbour_counts:
            if n==0:
                print("Solo peak in cluster")
                print_cluster( cluster )
                raise

        if no_peaks>2:
            # check that not peaks are solo two and two
            for i,ne in enumerate(neighbours):
                if neighbour_counts[ ne[0] ]==1 and neighbour_counts[ i ]==1:
                    if neighbours[ ne[0] ]==i:
                        print("Two peaks are solo together")
                        raise


def basic_check_many_clusters(clusters,index_z,index_y,index_om,tol_z,tol_y,tol_om):
    for cluster in clusters:
        basic_check_cluster(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om)


def check_many_clusters( peak_clusters, index_z,index_y,index_om,tol_z,tol_y,tol_om ):

    #---------------------------------------------------------------------------
    # check format:
    if len(peak_clusters) < 0:
        print("Expected peak clusters but got something with length zero")

    for cluster in peak_clusters:
        assert type(cluster)==list
        for peak in cluster:
            assert type(peak)==list
            for val in peak:
                assert type(val)!=list
    #---------------------------------------------------------------------------

    for i,cluster in enumerate(peak_clusters):
        #print("Checking cluster no ", i,"...")
        check_cluster(cluster,index_z,index_y,index_om,tol_z,tol_y,tol_om)
    #print("All Ok!")


# index_z = 1
# index_y = 0
# index_om = 3
# tol_z = 10
# tol_y = 10
# tol_om = np.pi/180.


# if 0:
#     peaks = generate_test_data(5,5,2)
#     with open('bad_clusters.txt','w') as f:
#         f.write('peaks = [')
#         for peak in peaks:
#             f.write(str(peak)+',')
#         f.write(']')
# else:
#     lines = [line for line in open('bad_clusters.txt').readlines()]
#     for l in lines: exec(l)



#peaks = [ [60, 44, 16, 10],[99, 35, 18, 10], [57, 30, 17, 10] ]
# t=[]
# iterations=[]
# no_peaks=[]
# no_runs = 2
# for i in range(no_runs):
#     #print_cluster(peaks)
#     peaks = generate_test_data(100*(i+1),50,3)
#     no_peaks.append( len(peaks) )
#     print("Done Generating Data, no peaks: ", len(peaks))
#     t1 = time.clock()
#     peak_clusters, iter = cluster(peaks,index_z,index_y,index_om,tol_z,tol_y,tol_om)
#     t2 = time.clock()
#     t.append( t2-t1 )
#
#     iterations.append(iter)
#     #print_many_clusters(peak_clusters)
#     #basic_check_many_clusters(peak_clusters,index_z,index_y,index_om,tol_z,tol_y,tol_om)
#     check_many_clusters(peak_clusters,index_z,index_y,index_om,tol_z,tol_y,tol_om)
#     print(" => ALL OK ! <=")
# print("Average number of reiterations: ", sum(iterations)/float(no_runs))
# print("Maximum number of reiterations: ", max(iterations))
# print("Average time use: ", np.round(sum(t)/float(no_runs),3)," s")
# import matplotlib.pyplot as plt
# plt.scatter( no_peaks, t)
# plt.show()


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
