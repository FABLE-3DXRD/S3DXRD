import numpy as np
import math
import pygmsh
# import meshio
# import rasterio.features
import copy
import matplotlib.pyplot as plt # for debugging :=)
from scipy.interpolate import griddata

'''
GENERAL DESCRIPTION:

This module is for converting a boolean matrix into a triangular
mesh. For instance the ndarray:

0 1 1 0 0
0 1 1 0 0
0 1 1 1 1
0 1 1 1 1

would give an L-shaped polygon that could be meshed with triangular elements.

Here, each index is a square with side length specified by the user. So we are
taking a square grid representation into triangular representation, while also
converting into coordinate values rather than index values. This has been designed
for taking a FBP reconstruction maps and meshing the tomograms for further computations.

This module is not optimised w.r.t speed as it was originally intended for
boolean matrices with dimensions less than say 200x200 and meshes with less than
say 2500 elements.
'''


def extract_boundary( logical_shape, bin_length, origin):
    '''
    Take a NxM numpy array and extract the boundary of the
    shaped defined by the non-zero instances of the array.

    - Assumes that no holes exists in the array.
    - Assumes x in colon direction, positive for increasing cols
    - Assumes y in row direction, positive for decreasing rows.

    origin - numpy.array([row_of_origin, colon_of_origin])
    '''

    # get the indices of the boundary
    bound_rows, bound_cols = get_boundary_index( logical_shape )

    ## Debug boundary index
    # A = np.zeros(logical_shape.shape)
    # A[bound_rows, bound_cols]=1
    # plt.imshow(A)
    # plt.show()

    border_points=[]
    for row,col in zip(bound_rows, bound_cols):
        points = index_to_coordinates(logical_shape, row, col, bin_length, origin)
        for point in points: border_points.append(point)

    border_points = np.array(border_points)
    boundary = np.unique(border_points, axis=0)
    boundary = sort_border_points( boundary )

    #Debug boundary coordinates and order
    print(boundary)
    plt.scatter(boundary[:,0],boundary[:,1])
    for i,(x,y) in enumerate(zip(boundary[:,0],boundary[:,1])):
        plt.text(x,y,str(i))
    plt.show()
    #boundary = slim( boundary )

    return boundary

def slim(boundary):
    slim_bound = []
    for i,p0 in enumerate(boundary):
        p1 = boundary[i+1,:]
        p2 = boundary[i+2,:]
        vec1 = p1 - p0
        vec2 = p2 - p1



def sort_border_points( border_points ):
    '''
    take a set of points ndarray x,y defining a border
    return the sorted array "w.r.t walking around the border"
    '''
    b_p = []
    p = list(border_points[0,:])
    b_p.append( p ) #first point
    while(len(b_p)<border_points.shape[0]):
        n1_indx, n2_indx = find_neighbors(p, border_points)
        n1 = list(border_points[n1_indx,:])
        n2 = list(border_points[n2_indx,:])
        if n1 not in b_p:
            b_p.append(n1)
            p = n1
        elif n2 not in b_p:
            b_p.append(n2)
            p = n2
        else:
            raise

    return np.array(b_p)

def find_neighbors(p, points):
    dx = points[:,0] - p[0]
    dy = points[:,1] - p[1]
    L = np.sqrt(dx*dx + dy*dy)
    indx_three_smallest = np.argpartition(L,3)
    indx_neighbors = []
    for indx in indx_three_smallest:
        if L[indx]>0: indx_neighbors.append(indx)
    return indx_neighbors[0], indx_neighbors[1]


def index_to_coordinates(logical_shape, row, col, bin_length, origin):
    centre_x = (col - origin[1])*bin_length
    centre_y = (-row + origin[0])*bin_length

    border_points = []
    border_corners = corners(logical_shape, row, col)
    for i,point in enumerate(border_corners):
        x = centre_x
        y = centre_y
        if point==1:
            if i==0:
                x -=  bin_length/2.
                y -=  bin_length/2.
            elif i==1:
                x -=  bin_length/2.
                y +=  bin_length/2.
            elif i==2:
                x +=  bin_length/2.
                y +=  bin_length/2.
            elif i==3:
                x +=  bin_length/2.
                y -=  bin_length/2.
            border_points.append([x,y])

    return border_points



def get_boundary_index( logical_shape ):
    boundary_mask = np.zeros(logical_shape.shape)
    for row in range(logical_shape.shape[0]):
        for col in range(logical_shape.shape[0]):
            if is_boundary(logical_shape, row, col):
                boundary_mask[row,col]=1
    index = np.where( boundary_mask==1 )
    return index[0],index[1]


def corners(logical_shape, row, col):
    '''
    1----2
    |    |
    0----3
    '''

    border_corners = np.zeros(4)

    if row==logical_shape.shape[0]-1:
        border_corners[(0,3)]=1
    if row==0:
        border_corners[(1,2)]=1
    if col==logical_shape.shape[1]-1:
        border_corners[(2,3)]=1
    if col==0:
        border_corners[(0,1)]=1

    try: border_corners[(1,2)] = (logical_shape[row+1,col]==0)
    except: pass

    try: border_corners[(0,3)] = (logical_shape[row-1,col]==0)
    except: pass

    try: border_corners[(2,3)] = (logical_shape[row,col+1]==0)
    except: pass

    try: border_corners[(1,0)] = (logical_shape[row,col-1]==0)
    except: pass

    try: border_corners[1] = (logical_shape[row-1,col-1]==0)
    except: pass

    try: border_corners[3] = (logical_shape[row+1,col+1]==0)
    except: pass

    try: border_corners[2] = (logical_shape[row-1,col+1]==0)
    except: pass

    try: border_corners[0] = (logical_shape[row+1,col-1]==0)
    except: pass

    return border_corners



def is_boundary( logical_shape, row, col):

    if logical_shape[row,col]==0:
        return False

    if row==logical_shape.shape[0]-1 or\
       row==0 or\
       col==logical_shape.shape[1]-1 or\
       col==0:
        return True

    if logical_shape[row+1,col]==0 or\
       logical_shape[row-1,col]==0 or\
       logical_shape[row,col+1]==0 or\
       logical_shape[row,col-1]==0 or\
       logical_shape[row-1,col-1]==0 or\
       logical_shape[row+1,col+1]==0 or\
       logical_shape[row-1,col+1]==0 or\
       logical_shape[row+1,col-1]==0:
        return True



def create_mesh( boundary, elm_size=0.05  ):
    '''
    Take numpy nx2 array of vertices defining a boundary and
    create a discretisation with nodes and elements.

    Input: boundary nx2 array numpy.array( [ [x0 y0], [x1, y1], ... )
    Return: numpy array of node coordinatesnumpy.array( [ n0e0x, n0e0y, n1e0x, n1e0y, n2e0x, n2e0y], [n0e1x, n1e1y,....], ... )
            each row is a element.
    '''
    print(elm_size)
    # convert numpy boundary to a list with also z=0 coordinates
    polygon = np.zeros( (boundary.shape[0],boundary.shape[1]+1) )
    polygon[:,0:-1] = boundary[:,:]
    polygon = list( polygon )

    # Create polygon object bia pygmsh
    geom = pygmsh.built_in.Geometry()
    poly = geom.add_polygon(
        polygon,
        lcar=elm_size
        )

    # Produce a mesh over the 2D shape via pygmsh
    mesh_object = pygmsh.generate_mesh(geom)
    points = mesh_object.points
    cells = mesh_object.cells
    
    # convert to internal prefered mesh single matrix format
    elm_nodes = cells['triangle'] # mapps elements to row index in points
    mesh = np.zeros((elm_nodes.shape[0],6))
    points = points[:,0:-1] # cut away the z-coordinates

    for i,elm in enumerate(elm_nodes):
        # extract the 3 nodes of the element and insert them as a single row in nodes
        nodes_xy = points[ elm,: ]
        mesh[i,0:2] = nodes_xy[0,:]
        mesh[i,2:4] = nodes_xy[1,:]
        mesh[i,4:6] = nodes_xy[2,:]
    return mesh

def create_pixel_mesh( image, bin_length, origin ):
    rows, cols = image.shape
    mesh = np.zeros((np.sum(image),8))
    elm = 0
    for i in range(rows):
        for j in range(cols):
            if image[i,j]==1:
                mesh[elm,:] = compute_corners(i,j,bin_length, origin)
                elm += 1
    return mesh

def compute_corners(row, col, bin_length, origin):
    centre_x = (-row + origin[0])*bin_length
    centre_y = (col - origin[0])*bin_length

    p1x = centre_x - bin_length/2.
    p1y = centre_y - bin_length/2.
    p2x = centre_x - bin_length/2.
    p2y = centre_y + bin_length/2.
    p3x = centre_x + bin_length/2.
    p3y = centre_y + bin_length/2.
    p4x = centre_x + bin_length/2.
    p4y = centre_y - bin_length/2.

    return np.array([p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y])

def get_elm_centres( mesh ):
    elm_centres = np.zeros( (mesh.shape[0],2) )
    for i,elm in enumerate(mesh):
        elm_centres[i,:] = elm_centre(elm)
    return elm_centres

def elm_centre( elm ):
    mean_x = np.mean( elm[0::2] )
    mean_y = np.mean( elm[1::2] )
    return np.array([mean_x, mean_y])