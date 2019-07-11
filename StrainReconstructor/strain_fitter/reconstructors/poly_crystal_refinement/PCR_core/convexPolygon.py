'''
This is a module for computing area and centroid of convex
non-self intersecting closed 2D polygons

(The idea is that each voxel can be represented as a unit square
with a local coordinate system attached to it's cms.
The problem is then reduced to finding the polygon areas and cms formed
by the rotated unitsquare and the one or two intersecting lines which
are the transition lines between beam regions)
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def intersection_point(a,b,y_coord):
    '''
    Return the point (x,y_coord) defined by the intersection
    of y_coord with the line x=ky+m formed by point 'a' and 'b'
    in the interval formed by the y coordinates a of 'a' and 'b'
    assumes a[1] - b[1] > 0, (i.e 'a' to left of 'b' in below picture)
    Local coordinates:
    (aligned with labaratory system)
             ^ x
      y      |
      <------|----
             |
    x = x_laboratory + x_translation
    y = y_laboratory + y_translation
    '''
    if not (y_coord < a[1] and y_coord > b[1]):
        return None
    dy = a[1]-b[1]
    dx = a[0]-b[0]
    if dy==0:
        return None #cannot intersect vertical line k=>infty
    k = dx/dy
    m = a[0] - k*a[1]
    x = k*y_coord + m
    return [x, y_coord]

def area_and_cms(points):
    '''
    Compute the centroid and area of a closed convex
    polygon defined by the vector of (x,y) coordinates 'points'

    if the points are in clockwise order Area<0 !
    '''
    Area = 0
    CMS_x = 0
    CMS_y = 0
    for i in range(len(points)-1):
        common_factor = points[i][0]*points[i+1][1]-points[i+1][0]*points[i][1]
        Area += common_factor
        CMS_x += ( points[i][0]+points[i+1][0] )*common_factor
        CMS_y += ( points[i][1]+points[i+1][1] )*common_factor
    Area = Area/2.
    CMS_x = CMS_x/(6.*Area)
    CMS_y = CMS_y/(6.*Area)

    return Area,CMS_x,CMS_y

def compute_voxel_beam_overlap(tx,ty,omega,central_beam_center, beam_width):
    '''
    Compute the area fraction of voxel which overlaps with beam dty settings
    Return the centroids of respective area fraction in global lab coordinates
    '''
    if omega<0:
        raise

    while(omega>np.pi/2.):
        omega = omega - np.pi/2.
    #print("omega ",np.degrees(omega))
    alpha = np.pi/4. + omega
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    fact = ( 1/np.sqrt(2) )
    a = [ fact*ca, fact*sa ]
    b = [ fact*sa, -fact*ca ]
    c = [ -a[0], -a[1] ]
    d = [ -b[0], -b[1] ]

    #DEBUG
    # ----
    # points = [a,d,c,b,a]
    # pnames=['a','d','c','b','a']
    # e_p = []
    # e_pname = []
    # Area,cx,cy = area_and_cms(points)
    # ----

    A0=cx_0=cy_0 = 0
    A1=cx_1=cy_1 = 0
    A2=cx_2=cy_2 = 0

    # Local description of the two lines that can cut the voxel
    # the local description uses beam_width as units
    l1 = (central_beam_center - ty)/beam_width + 0.5
    l2 = (central_beam_center - ty)/beam_width - 0.5


    if a[1] > l1:
        #print("a[1] > l1 ", a[1] ,">", l1)
        #a in bcy0
        if c[1] < l2:
            #print("c[1] < l2 ", c[1],"<", l2)
            #a in bcy0 c in bcy2
            # Three polygons four intersections
            p = intersection_point(a,b,l1)
            l = intersection_point(b,c,l2)
            m = intersection_point(d,c,l2)
            n = intersection_point(a,d,l1)
            # print("p",p)
            # print("l",l)
            # print("m",m)
            # print("n",n)
            A0,cx_0,cy_0 = area_and_cms([a,n,p,a])
            A1,cx_1,cy_1 = area_and_cms([p,n,d,m,l,b,p])
            A2,cx_2,cy_2 = area_and_cms([l,m,c,l])

            #DEBUG
            #----
            # e_p = [p,l,m,n,[cx_0,cy_0],[cx_1,cy_1],[cx_2,cy_2]]
            # e_pname = ['p','l','m','n','cms0','cms1','cms2']
            #----

        else:
            #print("c[1] > l2 ", c[1],"<", l2)
            #a in bcy0 c in bcy1
            #Two polygons two intersections
            p = intersection_point(a,b,l1)
            n = intersection_point(a,d,l1)
            if p is None:
                p = intersection_point(b,c,l1)
                A0,cx_0,cy_0 = area_and_cms([a,n,p,b,a])
                A1,cx_1,cy_1 = area_and_cms([p,n,d,c,p])
            elif n is None:
                n = intersection_point(d,c,l1)
                A0,cx_0,cy_0 = area_and_cms([a,d,n,p,a])
                A1,cx_1,cy_1 = area_and_cms([p,n,c,b,p])
            else:
                A0,cx_0,cy_0 = area_and_cms([a,n,p,a])
                A1,cx_1,cy_1 = area_and_cms([p,n,d,c,b,p])
            #DEBUG
            #------
            # e_p = [p,n,[cx_0,cy_0],[cx_1,cy_1]]
            # e_pname = ['p','n','cms0','cms1']
            #-----

    elif c[1] < l2:
        #c in bcy1 a in bcy0
        l = intersection_point(b,c,l2)
        m = intersection_point(d,c,l2)
        if l is None:
            l = intersection_point(a,b,l2)
            A1,cx_1,cy_1 = area_and_cms([l,a,d,m,l])
            A2,cx_2,cy_2 = area_and_cms([l,m,c,b,l])
        elif m is None:
            m = intersection_point(a,d,l2)
            A1,cx_1,cy_1 = area_and_cms([b,a,m,l,b])
            A2,cx_2,cy_2 = area_and_cms([l,m,d,c,l])
        else:
            A1,cx_1,cy_1 = area_and_cms([a,d,m,l,b,a])
            A2,cx_2,cy_2 = area_and_cms([l,m,c,l])
        #DEBUG
        #-----
        # e_p = [l,m,[cx_1,cy_1],[cx_2,cy_2]]
        # e_pname = ['l','m','cms1','cms2']
        #-----
    else:
        #all in bcy0
        A0,cx_0,cy_0 = area_and_cms([a,d,c,b,a])


    #DEBUG
    #----
    # title = 'A0='+str(np.round(A0,3))+',  '+ \
    # 'A1='+str(np.round(A1,3))+ \
    # ',  '+'A2='+str(np.round(A2,3))+ \
    # '    => Atot='+str(np.round(A0+A1+A2,3))
    # plot_more(points,pnames,Area,cx,cy,[l1, l2],e_p,e_pname,title)
    # plt.show()
    #----

    # return result in lab frame:
    # the unit is beam_width and the translation tx and ty
    return (A0,beam_width*cx_0+tx,beam_width*cy_0+ty), \
           (A1,beam_width*cx_1+tx,beam_width*cy_1+ty), \
           (A2,beam_width*cx_2+tx,beam_width*cy_2+ty)







#DEBUG functions below
#----------------------------------------

def plot_more(points,pnames,Area,cx,cy,lines, extra_points=None, e_pnames=None,title=None):
    fig,ax = plt.subplots()
    po = np.asarray(points)
    for i in range(len(po)-1):
        p=po[i]
        ax.scatter(p[0],p[1],c='b')
        ax.text(p[0],p[1],str(pnames[i]),fontsize=14)
        ax.plot(po[i+1],p,'b')
    # ax.scatter(po[-1][0],po[-1][1],c='b')
    # ax.plot(po[0],po[-1],'b')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(("Area = "+str(np.round(Area,3))))
    ax.scatter(cx,cy,c='r')
    ax.text(cx,cy,str(np.round(cx,3))+","+str(np.round(cy,3)))
    for l in lines:
        ax.axhline(l,c='y')
    ax.axhline(lines[0]-0.5,linestyle=':',c='y')
    if extra_points:
        for i,p in enumerate(extra_points):
            ax.scatter(p[0],p[1],c='r')
            if e_pnames:
                ax.text(p[0],p[1],str(e_pnames[i]),fontsize=14)
    ax.axis('equal')
    ax.axvline(0)
    ax.axhline(0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot(points,Area,cx,cy,inter_p=None):
    fig,ax = plt.subplots()
    po = np.asarray(points)
    for i in range(len(po)-1):
        p=po[i]
        ax.scatter(p[0],p[1],c='b')
        ax.text(p[0],p[1],str(i))
        ax.plot(po[i+1],p,'b')
    # ax.scatter(po[-1][0],po[-1][1],c='b')
    # ax.plot(po[0],po[-1],'b')
    ax.set_title(("Area = "+str(np.round(Area,3))))
    ax.scatter(cx,cy,c='r')
    ax.text(cx,cy,str(np.round(cx,3))+","+str(np.round(cy,3)))
    if inter_p:
        ax.scatter(inter_p[1],inter_p[0],c='r')
        ax.axvline(inter_p[1])
        ax.text(inter_p[1],inter_p[0],'intersection point')

def test_unit_square():
    points=[[0,0],[1,0],[1,1],[0,1],[0,0]]
    A,cx,cy = area_and_cms(points)
    print("Area ",A)
    print("centroid x,y ",cx,cy)
    if A!=1 or cx!=cy!=0.5:
        raise
    else:
        print("ALL OK!")
    return points,A,cx,cy

def test_unit_triangle():
    points=[[0,0],[1,0],[0,1],[0,0]]
    A,cx,cy = area_and_cms(points)
    print("Area ",A)
    print("centroid x,y ",cx,cy)
    if A!=0.5 or cx!=cy!=(1/3.):
        raise
    else:
        print("ALL OK!")
    return points,A,cx,cy


#DEBUG
#-----------
#beam_width=0.01
#tx = beam_width/10.
#ty = beam_width/9.
#omega=np.pi/4.3
#central_beam_center=beam_width/2.

#print(compute_voxel_beam_overlap(tx,ty,omega,central_beam_center, beam_width))
# points,A,cx,cy = test_unit_square()
# p = intersection_point([1,0],[1,1],0.35)
# plot(points,A,cx,cy,p)
# points,A,cx,cy = test_unit_triangle()
# p = intersection_point([1,0],[0,1],0.1)
# plot(points,A,cx,cy,p)
# plt.show()
#----------
