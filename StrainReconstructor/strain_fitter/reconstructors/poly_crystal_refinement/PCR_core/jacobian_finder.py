'''
Module for computing the jacobian at a give grain state
optimised to exploit the variable independecy of the voxels.

can be used together with scipy.optimise least_squares()
to generate a callable for the jacobian computations.
'''

import numpy as np
import multiprocessing as mp

class JacobianFinder():

    def __init__(self, func, m, n, C, constraint):
        '''
        '''
        self.C = C
        self.constraint = constraint
        # Constants
        self.run = func
        self.no_cpus = mp.cpu_count() - 1
        self.m = m
        self.n = n
        self.procs = self.get_workload()

        # Variables
        self.result_queue = mp.Queue()
        self.all_jacobian_colons = []

        # def f(x,xp):
        #     A = np.ones(x.shape)
        #     for i,a in enumerate(A):
        #         A[i]=x[0]*x[i]

        #     return A

        # self.run = f
        # self.no_cpus = mp.cpu_count() - 2
        # self.n = 10
        # self.procs = self.get_workload()

        # # Variables
        # self.result_queue = mp.Queue()
        # self.all_jacobian_colons = []


    def function(self,x0,altered_variable):
        '''
        The function to minimise. This is a wrapper of the forward model
        for ease of coding.

        The assumptions of x0 is also here important. 9 variables per voxel
        is assumed.
        '''
        start_voxel = altered_variable//9
        end_voxel = start_voxel + 1
        fvec = self.run(x0, start_voxel=start_voxel, end_voxel=end_voxel, gradient_mode='Yes')
        return fvec
        #return self.run(x0,altered_variable)


    def directional_derivative(self,x0, xp):
        '''
        Find the directional deriviative of the n dimensional function
        in direction xp. The function is supposed to return a m dimensional
        vector of errors. Three point evaluation is used (lot better than one point)

        INPUT:
            x0 -  vector of n variables at state of evaluation
            xp -  index of the p:th variable (direction of derivative)
            function  -  function to be evaluated

        OUTPUT:
            - derivative vector (colon xp of jacobian)
        '''
        h = np.zeros(x0.shape)
        h[xp] = np.sqrt( np.finfo(type(x0[0])).eps )
        #print(max(abs(( self.function(x0 + h, xp) - self.function(x0 - h, xp) ) / (2*h[xp]))))
        return ( self.function(x0 + h, xp) - self.function(x0 - h, xp) ) / (2*h[xp])


    def get_workload(self):
        '''
        distribute n tasks to no_cpus cpus
        '''

        no_vars_per_procs = int( np.ceil(self.n/float(self.no_cpus)) )

        procs = np.zeros((self.no_cpus,2),dtype=np.uint16)
        for i in range(self.no_cpus-1):
            procs[i,0] = i*no_vars_per_procs
            procs[i,1] = procs[i,0] + no_vars_per_procs
        procs[-1,0] = procs[-2,1]
        procs[-1,1] =  self.n

        return procs


    def start_parallel(self,x0):
        '''
        '''

        self.result_queue = mp.Queue()

        def target(x0, start_var, end_var):
            self.result_queue.put( self.find_part_of_jacobian(x0, start_var, end_var) )

        running_procs = []
        for i in range(self.procs.shape[0]):
            start_var = self.procs[i,0]
            end_var = self.procs[i,1]
            running_procs.append(mp.Process(target=target, args=(x0, start_var, end_var)))
            running_procs[-1].start()

        self.all_jacobian_colons = []
        while True:
            try:
                jacobian_colons = self.result_queue.get(False, 0.01)
                self.all_jacobian_colons.append(jacobian_colons)
            except:
                pass
            allExited = True
            for proc in running_procs:
                if proc.exitcode is None:
                    allExited = False
                    break
            if allExited & self.result_queue.empty():
                break

    def find_part_of_jacobian(self,x0, start_var, end_var):
        '''
        '''
        jacobian_colons = {}
        for i in range(start_var, end_var):
            xp = i
            df = self.directional_derivative(x0, xp)
            jacobian_colons[xp] = df
        return jacobian_colons


    def find_jacobian_in_parallel(self,x0):
        '''
        Return the jacobian of the inputted function
        '''
        self.start_parallel(x0)
        Jacobian = np.zeros((self.m,self.n))
        for value in self.all_jacobian_colons:
            for key in value:
                Jacobian[:,key] = value[key][:]
        return Jacobian

    def find_jacobian_one_proc(self,x0):

        Jacobian = np.zeros((self.m,self.n))
        for i in range(self.n):
            xp = i
            df = self.directional_derivative(x0, xp)
            Jacobian[:,xp]=df[:]
        return Jacobian






# x0 = np.ones((10,1))
# jacwack = JacobianFinder()
# J = jacwack.find_jacobian(x0)
# print(J)

#     # def grad_descent(x0,f,args,tol=10**(-8),maxiter=300):
#     #     dx = np.ones(x0.shape)*0.001
#     #     fnew =  f(x0,*args)
#     #     err = 2*tol
#     #     iter=1
#     #     while(err>tol or iter==maxiter):
#     #         iter+=1
#     #         print(iter)
#     #         print(err)
#     #         fold = fnew
#     #         J = find_jacobian(x0,f,args)
#     #         x0 -= np.dot( J, dx )
#     #         fnew = f(x0,*args)
#     #         err = np.linalg.norm(fnew-fold)
#     #     print("solution ", x0)
#     #     print("f(solution) ",fnew)

#     # def f(x,A,B):
#     #     return x*x + A

#     # x0 = np.ones((10,1))*0.9865
#     # A = 1.000023
#     # B = 3.45867
#     # J = find_jacobian(x0,f,(A,B))
#     # #print( "Jacobian ", J)
#     # grad_descent(x0,f,(A,B))
