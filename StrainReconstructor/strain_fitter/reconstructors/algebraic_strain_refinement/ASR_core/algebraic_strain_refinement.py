import numpy as np
from shapely.geometry import Polygon as shapelyPloygon
import matplotlib.pyplot as plt
import illustrate_mesh
import mesher as mesher
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
from xfab import tools
from ImageD11 import indexing
import sys
from scipy.sparse import csr_matrix
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

def get_A_matrix_row( mesh, n, N, omega, dty, beam_width ):
    # rotate and translate mesh by omega and dty
    rot_mesh = rotate_mesh( mesh, omega )

    rot_trans_mesh = translate_mesh( rot_mesh, 0, dty )

    # compute the area of intersection between elements and beam
    areas = compute_intersection_areas( rot_trans_mesh, beam_width )

    # print(dty)
    # illustrate_mesh.plot_field(rot_trans_mesh, areas)
    # plt.show()

    area_tot = np.sum(areas)

    if area_tot==0: # perhaps also remove very slightly gracing lines
        return None

    row_areas = np.repeat(areas,6)
    #print('area_tot',area_tot)
    return row_areas*a(n, N)/area_tot


def compute_intersection_areas( mesh, beam_width, plot=False ):

    intersection_areas = np.zeros(mesh.shape[0])
    w = beam_width/2.
    if mesh.shape[1]==6:
        min_x = np.min(mesh[:,(0,2,4)])
        max_x = np.max(mesh[:,(0,2,4)])
    elif mesh.shape[1]==8:
        min_x = np.min(mesh[:,(0,2,4,6)])
        max_x = np.max(mesh[:,(0,2,4,6)])

    beam = shapelyPloygon([(min_x, w), (max_x, w), (max_x, -w), (min_x, -w)])

    for i,elm in enumerate(mesh):
        
        # TODO: make this loop more efficent
        # save beam as polygon once
        # save plygon mesh once
        # only iterate the elements close to the beam
        if abs(np.mean(elm[1::2]))>=np.sqrt(2)*3*w:
            intersection_areas[i] = 0.0
            continue

        if len(elm)==6:
            poly_elm = shapelyPloygon([(elm[0], elm[1]), (elm[2], elm[3]), (elm[4], elm[5])])
        elif len(elm)==8:
            poly_elm = shapelyPloygon([(elm[0], elm[1]), (elm[2], elm[3]), (elm[4], elm[5]), (elm[6], elm[7])])
        intersection_areas[i] = beam.intersection(poly_elm).area
        
        if plot:
            plt.plot(beam.exterior.xy[0],beam.exterior.xy[1])
            for m in mesh:
                plt.plot(m[0],m[1],'ro')
                plt.plot(m[2],m[3],'ro')
                plt.plot(m[4],m[5],'ro')
                plt.plot(m[6],m[7],'ro')
            plt.plot(poly_elm.exterior.xy[0],poly_elm.exterior.xy[1])
            plt.title(intersection_areas[i])
            plt.show()

    return intersection_areas


def translate_mesh( mesh, translation_x, translation_y ):
    trans_mesh = mesh[:,:]
    if mesh.shape[1]==6:
        trans_mesh[:,(0,2,4)] -= translation_x
        trans_mesh[:,(1,3,5)] -= translation_y
    elif mesh.shape[1]==8:
        trans_mesh[:,(0,2,4,6)] -= translation_x
        trans_mesh[:,(1,3,5,7)] -= translation_y       
    return trans_mesh


def rotate_mesh( mesh, omega ):
    # construct rotation matrix by omega
    s = np.sin( np.radians(omega) )
    c = np.cos( np.radians(omega) )

    if mesh.shape[1]==6:
        rot_matrix = np.array([[c,-s,0,0,0,0],
                            [s,c,0,0,0,0],
                            [0,0,c,-s,0,0],
                            [0,0,s,c,0,0],
                            [0,0,0,0,c,-s],
                            [0,0,0,0,s,c]])
    elif mesh.shape[1]==8:
        rot_matrix = np.array([[c,-s,0,0,0,0,0,0],
                            [s,c,0,0,0,0,0,0],
                            [0,0,c,-s,0,0,0,0],
                            [0,0,s,c,0,0,0,0],
                            [0,0,0,0,c,-s,0,0],
                            [0,0,0,0,s,c,0,0],
                            [0,0,0,0,0,0,c,-s],
                            [0,0,0,0,0,0,s,c]])

    # rotate mesh omgea degrees
    return np.transpose(np.dot(rot_matrix, np.transpose(mesh)))

def a( n, N ):
    '''
    Compute the directional weights of a particular w:y scan beam
    (to be used to form a row in the A matrix)
    '''

    return np.array( [ n[0]*n[0], n[1]*n[1], n[2]*n[2], 2*n[1]*n[2], 2*n[0]*n[2], 2*n[0]*n[1] ]*N )


def calc_A_matrix( mesh, directions, omegas, dtys, beam_width  ):
    M = len(omegas) # number of measurements
    N = mesh.shape[0] # number of elements

    A = np.zeros((M,6*N))

    bad_equations = []
    count=0
    print('Building projection matrix...')
    for k,(omega, dty, n) in enumerate( zip(omegas, dtys, directions) ):
        # each measurement is weigthed by its belived resolution..
        row = get_A_matrix_row( mesh, n, N, omega, dty, beam_width )
        if k%300==0:
            print('Getting row number: '+str(k) +' of '+str(len(dtys)))
        if row is None:
            bad_equations.append(k)
            count+=1
        else:
            A[k,:] = row
    print('')
    print("Rank of A: "+str(np.linalg.matrix_rank(A)) )
    print("Total number of eqs: "+str( M) )
    print("Topology cutoff lead to "+str( 100.*count/float(M) )+" percent eqs to be unusuable")
    print('')
    return A, bad_equations

def constraints(mesh, low_bound, high_bound):
    '''
    Limit the difference in strain between to neighbouring elements
    A neighbour pair is defined as two elements sharing at least one node
    '''
    print('computing constraints matrix')
    c = []
    incl = []

    data = []
    row = []
    col = []
    curr_row = 0
    for i,elm in enumerate(mesh):
        indx = find_index_of_neighbors(mesh, elm)
        #print('Element '+ str(i) + ' neighbours elements' + str(indx) )
        for j in indx:
            if [i,j] in incl or [j,i] in incl: continue
            
            for k in range(6):
                #row = [0]*mesh.shape[0]*6
                #row[(i*6)+k] = 1.
                #row[(j*6)+k] = -1.
                #c.append( row )
                row.append( curr_row )
                row.append( curr_row )
                curr_row+=1
                data.append(1.)
                col.append( (i*6)+k)
                data.append(-1.)
                col.append((j*6)+k)

            incl.append([i,j])
    r,c = curr_row, mesh.shape[0]*6

    # (updated to sparse representation, much faster)
    
    c = csr_matrix( (data, (row, col)), shape=(r, c) )
    #c = np.array(c)
    lb = np.ones(c.shape[0])*low_bound
    ub = np.ones(c.shape[0])*high_bound

    return lb, c, ub



def find_index_of_neighbors(mesh, element):
    elm_side = np.max(element[0::2]) - np.min(element[0::2])
    index_neighbors = []
    for i,elm in enumerate(mesh):
        if sum(elm==element)==len(elm): continue
        is_neighbour = False
        for x,y in zip(elm[0::2],elm[1::2]):
            for x_e,y_e in zip(element[0::2],element[1::2]):
                if abs(x-x_e)<elm_side/10. and abs(y-y_e)<elm_side/10.:
                    is_neighbour = True
        if is_neighbour: index_neighbors.append(i)
    return np.array( index_neighbors )

    


def trust_constr_solve( mesh, etas, hkl, tths, intensity, directions, strains , omegas, dtys, weights, beam_width, grad_constraint, maxiter ):
    '''
    '''

    nelm = mesh.shape[0]
    print('nelm: ', nelm)

    A, bad_equations = calc_A_matrix( mesh, directions, omegas, dtys, beam_width )

    A = np.delete(A, bad_equations, axis=0)
    strains = np.delete(strains, bad_equations, axis=0)
    weights = np.delete(weights, bad_equations, axis=0)
    
    lb,c,ub = constraints(mesh, -grad_constraint, grad_constraint)
    print('constraint matrix shape: ',c.shape)
    # plt.imshow(c.todense())
    # plt.show()

    linear_constraint = LinearConstraint(c, lb, ub, keep_feasible=True)
    x0 = np.zeros(6*nelm)

    def callback( xk, state ):

        out="   {}      {}      {}"
        if state.nit==1: print(out.format("iteration","cost","max strain grad") )
        # x = xk.reshape(nelm,6)
        # eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5]
        # illustrate_mesh.plot_field( mesh, eps_zz )

        # fig, ax = plt.subplots(figsize=(10,8))
        
        # coordinates = mesher.get_elm_centres( mesh )
        # zz = np.zeros((71,71))
        # for zval,coo in zip(eps_zz, coordinates):
        #     row = int( (coo[0]/beam_width)  + zz.shape[1]//2 )
        #     col = int( (coo[1]/beam_width)  + zz.shape[1]//2 )
        #     assert row>=0
        #     assert col>=0
        #     # print('coordinate', coo[0], coo[1])
        #     # print('row,col:', row, col)
        #     # print('beam_width: ',beam_width)
        #     zz[row,col] = zval
        # im = ax.imshow(zz*(10**4), cmap='jet')
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.ax.set_title(r'$10^{-4}$', size=27)
        # cbar.ax.tick_params(labelsize=17)
        # ax.tick_params(labelsize=17)
        # plt.show()
        print( out.format( state.nit, np.round(state.fun,9), np.max(np.abs(c.dot(xk))) ) )
        return state.nit==maxiter

    W = np.diag( weights )
    WA = np.dot(W,A)
    m = strains
    Wm = np.dot(W,m)
    WATWA = np.dot( WA.T, WA )
    WATWm = np.dot( WA.T, Wm )
    def func( x ): return 0.5*np.linalg.norm( (np.dot( WA, x ) - Wm) )**2
    def jac( x ): return ( np.dot(WATWA,x) - WATWm )
    def hess( x ): return WATWA

    res = minimize(func, x0, method='trust-constr', jac=jac, hess=hess,\
                    callback=callback, tol=1e-8, \
                    constraints=[linear_constraint],\
                    options={'disp': True, 'maxiter':maxiter})

    s_tilde = res.x

    #conditions = np.dot(c,s_tilde)

    if 0:
        omegas = np.delete(omegas, bad_equations, axis=0)
        dtys = np.delete(dtys, bad_equations, axis=0)
        directions = np.delete(directions, bad_equations, axis=0)
        etas = np.delete(etas, bad_equations, axis=0)
        hkl = np.delete(hkl, bad_equations, axis=0)
        tths = np.delete(tths, bad_equations, axis=0)
        intensity = np.delete(intensity, bad_equations, axis=0)

        # np.save('/home/axel/Desktop/A.npy', A)
        # np.save('/home/axel/Desktop/W.npy', W)
        # np.save('/home/axel/Desktop/s.npy', s_tilde)
        # np.save('/home/axel/Desktop/m.npy',m)
        # np.save('/home/axel/Desktop/omegas.npy',omegas)
        # np.save('/home/axel/Desktop/dtys.npy', dtys)
        # np.save('/home/axel/Desktop/directions.npy', directions)
        # np.save('/home/axel/Desktop/etas.npy', etas)
        # np.save('/home/axel/Desktop/hkl.npy', hkl)
        # np.save('/home/axel/Desktop/tths.npy', tths)
        # np.save('/home/axel/Desktop/intensity.npy', intensity)


        WAs = np.dot(WA,s_tilde)
        #print(directions)
        xm = []
        xstrainm = []
        xc = []
        xstrainc = []
        for i in range(directions.shape[0]):
            d=directions[i,:]
            #print(d)
            ang = np.degrees( np.arccos( abs(np.dot(d,np.array([1,0,0])) ) ) )
            #print(ang)
            if ang<10.0:
                xm.append( dtys[i] )
                xstrainm.append( Wm[i] )
                xc.append( dtys[i] )
                xstrainc.append( WAs[i] )

        s=30
        t=23

        plt.figure(1)
        plt.scatter( dtys, omegas, s=7, c=np.dot(WA, s_tilde), cmap='viridis')
        plt.xlabel(r'sample y-translation [$\mu$m]',size=s)
        plt.ylabel(r'sample rotation, $\omega$ [$^o$]',size=s)
        plt.title('Computed average strains')
        c1 = plt.colorbar()
        c1.ax.tick_params(labelsize=t)
        plt.tick_params(labelsize=t)

        plt.figure(2)
        plt.scatter( dtys, omegas, s=7, c=Wm, cmap='viridis')
        plt.xlabel(r'sample y-translation [$\mu$m]',size=s)
        plt.ylabel(r'sample rotation, $\omega$ [$^o$]',size=s)
        plt.title('Measured average strains')
        c2 = plt.colorbar()
        c2.ax.tick_params(labelsize=t)
        plt.tick_params(labelsize=t)

        plt.figure(3)
        plt.scatter( xm, xstrainm, s=85, marker="^", label=r'Measured strain ($\mathbf{Wm}$)'  )
        plt.scatter( xc, xstrainc, s=85, marker="o", label=r'Fitted strain ($\mathbf{WAs}$)' )
        plt.xlabel(r'x',size=s)
        plt.ylabel(r'Integrated weighted strain',size=s)
        plt.legend(fontsize=s)
        plt.tick_params(labelsize=t)

        plt.show()


    # reformat, each row is strain for the element
    s_tilde = s_tilde.reshape(nelm,6)

    return s_tilde[:,0],s_tilde[:,1],s_tilde[:,2],s_tilde[:,3],s_tilde[:,4],s_tilde[:,5]


def tikhonov_solve( mesh, directions, strains , omegas, dtys, weights, beam_width, grad_constraint ):
    '''
    '''

    nelm = mesh.shape[0]
    
    A, bad_equations = calc_A_matrix( mesh, directions, omegas, dtys, beam_width )
    
    A = np.delete(A, bad_equations, axis=0)
    strains = np.delete(strains, bad_equations, axis=0)
    weights = np.delete(weights, bad_equations, axis=0)

    W = np.diag( weights )
    WA = np.dot(W,A)
    m = strains
    Wm = np.dot(W,m)
    WATWA = np.dot( WA.T, WA )
    WATWm = np.dot( WA.T, Wm )

    lb,c,ub = constraints(mesh, -grad_constraint, grad_constraint)

    alpha = 0.0
    dalpha = 100.0
    #R = c*alpha
    s_tilde = np.zeros(mesh.shape[0]*6)
    itr=1
    #R = np.eye(WA.shape[0], WA.shape[1])*alpha
    while(np.max(np.abs(np.dot(c,s_tilde)))>grad_constraint or itr==1):
        R = c*alpha
        s_tilde = np.dot(np.dot(np.linalg.inv( WATWA + np.dot(R.T,R) ),WA.T), Wm )
        alpha = alpha + dalpha
        itr+=1
        print("Selecting lambda iteration ", itr)
    alpha = alpha - dalpha
    R = c*alpha
    
    s_tilde = np.dot(np.dot(np.linalg.inv( WATWA + np.dot(R.T,R) ),WA.T), Wm )

    # reformat, each row is strain for the element
    s_tilde = s_tilde.reshape(nelm,6)

    return s_tilde[:,0],s_tilde[:,1],s_tilde[:,2],s_tilde[:,3],s_tilde[:,4],s_tilde[:,5]




def solve( mesh, directions, strains , omegas, dtys, weights, beam_width ):
    '''
    Solve the linear matrix equation, A*m=b, by least squares approach.
    m is the discretised tensor field and b the average strain measurements from
    the various lines/beam illumination regions.
    '''

    W = np.diag( weights )
    #W = np.eye(len(weights),len(weights))
    A = calc_A_matrix( mesh, directions, omegas, dtys, beam_width )
    WA = np.dot(W,A)
    m = strains
    Wm = np.dot(W,m)

    #from scipy.optimize import lsq_linear
    #res = lsq_linear( WA, Wm, bounds=(-30*(10**-4), 30*(10**-4)), verbose=1 )
    #s_tilde = res.x

    s_tilde = np.dot(np.dot(np.linalg.inv( np.dot(WA.T,WA) ),WA.T), Wm )


    # reformat, each row is strain for the element
    nelm = mesh.shape[0]
    # print("nelm",nelm)
    s_tilde = s_tilde.reshape(nelm,6)

    return s_tilde[:,0],s_tilde[:,1],s_tilde[:,2],s_tilde[:,3],s_tilde[:,4],s_tilde[:,5]










# ABP with orientation
#-------------------------------------------------------

def get_areas(om, mesh, dty, beam_width):
    rot_mesh = rotate_mesh( mesh, om )
    rot_trans_mesh = translate_mesh( rot_mesh, 0, dty )
    areas = compute_intersection_areas( rot_trans_mesh, beam_width )
    area_tot = np.sum(areas)
    return areas, area_tot

def omega_mat( om ):
    so = np.sin( np.radians(om) )
    co = np.cos( np.radians(om) )
    return np.array([[co,-so,0],[so,co,0],[0,0,1]])

def assemble( G_omegas, Ghkls, omegas, weights, mesh, dtys, beam_width ):
#-------------------------------------------------------    
    N = mesh.shape[0]
    m = []
    A = []
    W = []
    for hkl, Gw, om, dty, w in zip(Ghkls, G_omegas, omegas, dtys, weights):
        areas, Atot = get_areas(om, mesh, dty, beam_width)
        h,k,l = hkl[0],hkl[1],hkl[2]

        #bad eq. no intersection of grain and beam (due to FBP cutof)
        if Atot==0:
            print("Atot",Atot)
            continue

        
        m1 = np.array([h,k,l,0,0,0,0,0,0])
        m2 = np.array([0,0,0,h,k,l,0,0,0])
        m3 = np.array([0,0,0,0,0,0,h,k,l])
        r1 = []
        r2 = []
        r3 = []
        for frac in (areas/Atot):
            r1.extend(list(m1*frac))
            r2.extend(list(m2*frac))
            r3.extend(list(m3*frac))
    
        A.append( r1 )
        A.append( r2 )
        A.append( r3 )

        W.append( w )
        W.append( w )
        W.append( w )

        m.extend( Gw.flatten() )

    return np.asarray( A ), np.asarray( m ), np.diag( np.asarray(W) )


def solve_with_orient( grain, hkl, G_omegas, omegas, weights, mesh, dtys, beam_width, cell_original ):

    nelm = mesh.shape[0]
    
    x0 = np.asarray( list( grain.ub.flatten() )*nelm )

    c, nconds = get_connections(mesh)
    # f = nonlinear_constraints( c, cell_original, nelm )
    # import time
    # t1 = time.clock()
    # a = f(x0)
    # t2=time.clock()
    # print("time to eval constraint func: ",t2-t1)
    # ang_cond = 0.5
    # strain_cond = 5*(10**-4)

    # lbase = np.ones(9)
    # lbase[0:6]= -strain_cond
    # lbase[6:9]= np.cos( np.radians(ang_cond) )
    # ubase = np.ones(9)
    # ubase[0:6]= strain_cond
    # ubase[6:9]= 1.00001
    # lb = list(lbase)*nconds
    # ub = list(ubase)*nconds

    # constraint = NonlinearConstraint(f,lb,ub,keep_feasible='True')

    A, m, W = assemble( G_omegas, hkl, omegas, weights, mesh, dtys, beam_width )
    WA = np.dot( W, A )
    Wm = np.dot( W, m )
    WATWA = np.dot( WA.T, WA )
    WATWm = np.dot( WA.T, Wm )

    constrains_base = []
    constrains_base.extend( [5*(10**(-4))]*6 )
    constrains_base.extend( [np.radians( 1.0 )]*3 )
    constraints = np.array( constrains_base*int(c.shape[0]/9.0) )

    def func( x ):
        epsang = to_strain_and_angs(x, cell_original, nelm)
        penalty = np.max([0, np.max( np.abs(np.dot(c,epsang)) - constraints)] )**2 
        return 0.5*np.linalg.norm( (np.dot( WA, x ) - Wm) )**2 + penalty
    #def jac( x ): return ( np.dot(WATWA,x) - WATWm )
    #def hess( x ): return WATWA

    def callback( xk, state ):
        out="   {}      {}"
        if state.nit==1: print(out.format("iteration","cost") )
        print( out.format( state.nit, np.round(state.fun,5) ) )
        return state.nit==5

    res = minimize(func, x0, method='trust-constr',\
                    callback=callback, tol=1e-8, \
                    options={'disp': True, 'maxiter':5})
    s_tilde = res.x
    # alpha = 500
    # c = constraints_ub(mesh, 1, 1)
    # R = c*alpha

    # alpha = 0.0
    # dalpha = 5000.0
    # #R = c*alpha
    # s_tilde = x0
    # itr=1
    # #R = np.eye(WA.shape[0], WA.shape[1])*alpha
    # print(to_strain_and_angs(s_tilde, cell_original, nelm)[0:10])
    # print(c.shape)
    # print(constraints[0:10])
    # print(np.dot(c,to_strain_and_angs(s_tilde, cell_original, nelm)).shape)
    # while(np.sum( np.abs( np.dot(c,to_strain_and_angs(s_tilde, cell_original, nelm)) )>constraints )>0 or itr==1):

    #     R = c*alpha
    #     s_tilde = np.dot(np.dot(np.linalg.inv( WATWA + np.dot(R.T,R) ),WA.T), Wm )
    #     alpha = alpha + dalpha
    #     itr+=1
    #     l = np.dot(c,to_strain_and_angs(s_tilde, cell_original, nelm))
    #     p = l[l>constraints]
    #     print(p[0:10])
    #     print("Selecting lambda iteration ", itr, np.sum( np.abs( np.dot(c,to_strain_and_angs(s_tilde, cell_original, nelm)) )>constraints ))
    # alpha = alpha - dalpha
    # R = c*alpha
    # s_tilde = np.dot(np.dot(np.linalg.inv( WATWA + np.dot(R.T,R) ),WA.T), Wm )

    voxel_UBs = s_tilde.reshape(nelm,3,3)
    return voxel_UBs

def to_strain_and_angs(s_tilde, cell, nelm):
    fun_val = []
    ubs = s_tilde.reshape(nelm,3,3)
    for i in range(nelm):
        ubi = np.linalg.inv(ubs[i])
        u, eps = tools.ubi_to_u_and_eps(ubi, cell)
        euler = tools.u_to_euler(u)
        fun_val.extend( eps )
        fun_val.extend( list(euler) )
    return np.asarray(fun_val)



def constraints_ub(mesh, low_bound, high_bound):
    '''
    Limit the difference in UB instacnes between to neighbouring elements
    A neighbour pair is defined as two elements sharing at least one node
    '''
    c = []
    incl = []
    for i,elm in enumerate(mesh):
        indx = find_index_of_neighbors(mesh, elm)
        for j in indx:
            if [i,j] in incl or [j,i] in incl: continue
            for k in range(9):
                row = [0]*mesh.shape[0]*9
                row[(i*9)+k] = 1.
                row[(j*9)+k] = -1.
                c.append( row )
            incl.append([i,j])
    c = np.array(c)
    #lb = np.asarray( list(low_bound)*int(c.shape[0]/9.) )
    #ub = np.asarray( list(high_bound)*int(c.shape[0]/9.) )
    #print("lb",lb.shape)
    #print("c",c.shape)
    return c#lb, c, ub

def get_connections(mesh):
    c = {}
    nconds = 0
    incl = []

    for i,elm in enumerate(mesh):
        c[i]=[]

    for i,elm in enumerate(mesh):
        indx = find_index_of_neighbors(mesh, elm)
        for j in indx:
            if [i,j] in incl or [j,i] in incl: 
                continue
            c[i].append(j)
            incl.append([i,j])
            nconds+=1
    return c, nconds

def nonlinear_constraints( connectivity, cell, nelm ):

    def f(x):
        fun_val = []
        ubs = x.reshape(nelm,3,3)
        for i in range(nelm):
            ubi1 = np.linalg.inv(ubs[i])
            u1,eps1 = tools.ubi_to_u_and_eps(ubi1, cell)
            for indx in connectivity[i]:
                ubi2 = np.linalg.inv(ubs[indx])
                u2,eps2 = tools.ubi_to_u_and_eps(ubi2, cell)
                deps = np.asarray(eps1)-np.asarray(eps2)
                dang1 = np.dot(u1[:,0],u2[:,0])
                dang2 = np.dot(u1[:,1],u2[:,1])
                dang3 = np.dot(u1[:,2],u2[:,2])
                fun_val.extend(deps)
                fun_val.extend([dang1,dang2,dang3])
        return np.asarray(fun_val)
    return f

#-------------------------------------------------------
