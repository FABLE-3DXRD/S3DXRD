'''
Welcome to the heart of ModelScanning3DXRD!
Here the forward model is executed and reflections are
computed for. The algorithm is found in the
run() method. The only part of the physics that is not
described in here or called from in here will probably reside
only in reflections.py where structure factors and
miller planes are computed for.
'''

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from xfab import tools,detector
from xfab.structure import int_intensity
from . import variables, cms_peak_compute, convexPolygon, hkl_manager, peak_debug_checker, jacobian_finder
import sys
from shapely.geometry import Polygon
import pylab as pl
import time
import multiprocessing as mp
import copy
from scipy.optimize import least_squares
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

voxel_reflections_id = variables.refarray().voxel_reflections_id

class find_refl_func:
    def __init__(self,param,hkl,voxel_positions,measured_data, unit_cell, initial_guess, ymin, no_yscans,  C, constraint, save_dir=None):
        '''
        A find_refl object is used to hold information
        that otherwise would have to be recomputed for
        during the algorithm execution in run().
        '''
        # print(voxel_positions)
        # raise
        self.C = C
        self.constraint = constraint
        self.count=0
        self.no_yscans = no_yscans
        self.save_dir = save_dir
        self.steplength_termination = 10**(-8)
        self.maxiter = 15

        self.initial_guess = initial_guess
        self.UB_matrices = None


        self.no_voxels = param['no_voxels']
        self.beam_width = param['beam_width']
        self.omega_start = param['omega_start']*np.pi/180
        self.omega_end = param['omega_end']*np.pi/180
        self.omega_step = param['omega_step']
        self.omega_sign = param['omega_sign']
        self.wavelength = param['wavelength']
        self.scale_Gw = self.wavelength/(4.*np.pi)
        self.detz_center = param['detz_center']
        self.dety_center = param['dety_center']
        self.z_size = param['z_size']
        self.y_size = param['y_size']
        self.distance = param['distance']
        self.o11 = param['o11']
        self.o12 = param['o12']
        self.o21 = param['o21']
        self.o22 = param['o22']
        self.dety_size = param['dety_size']
        self.detz_size = param['detz_size']
        self.lorentz_apply = param['lorentz_apply']
        self.tilt_x = param['tilt_x']
        self.tilt_y = param['tilt_y']
        self.tilt_z = param['tilt_z']
        self.wedge = param['wedge']


        self.voxel_positions = voxel_positions  #should be np array
        self.voxel = [None]*self.no_voxels

        # Simple transforms of input and set constants
        self.S = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

        # Detector tilt correction matrix
        self.R = tools.detect_tilt(self.tilt_x,
                                   self.tilt_y,
                                   self.tilt_z)

        # wedge NB! wedge is in degrees
        # The sign is reversed for wedge as the parameter in
        # tools.find_omega_general is right handed and in ImageD11
        # it is left-handed (at this point wedge is defined as in ImageD11)
        self.wy = -1.*self.wedge*np.pi/180.
        self.wx = 0.

        w_mat_x = np.array([[1, 0        , 0         ],
                           [0, np.cos(self.wx), -np.sin(self.wx)],
                           [0, np.sin(self.wx),  np.cos(self.wx)]])
        w_mat_y = np.array([[ np.cos(self.wy), 0, np.sin(self.wy)],
                           [0         , 1, 0        ],
                           [-np.sin(self.wy), 0, np.cos(self.wy)]])
        self.r_mat = np.dot(w_mat_x,w_mat_y)


        self.original_measured_data = copy.deepcopy(measured_data)
        self.measured_data = measured_data # numpy array, first index is dty, then a nx3 list of (sc,fc,omega)


        self.ymin = ymin #microns (same as self.measured_data)

        self.total_number_of_measured_peaks=0
        for dty in self.measured_data:
            self.total_number_of_measured_peaks+=len(dty)
        self.iter = 0

        # Used for merging delta peaks into clusters
        dety_tol, detz_tol, om_tol = self.pick_merge_tolerance()
        self.dety_tol = dety_tol
        self.detz_tol = detz_tol
        self.om_tol = om_tol


        self.nmedian = 5
        self.threshold = self.form_weighted_error([self.dety_tol/2.,self.detz_tol/2.,self.om_tol/2.],[0,0,0])


        self.peak_merger = cms_peak_compute.PeakMerger( param, self.voxel, self.voxel_positions, dety_tol, detz_tol, om_tol, self.ymin )

        # weigth to make all units in units of pixels/voxels in framestack
        # self.dety_weight = self.omega_step / np.degrees( np.arctan( self.y_size/self.distance ) )
        # self.detz_weight = self.omega_step / np.degrees( np.arctan( self.z_size/self.distance ) )
        # self.omega_weight = 1

        self.unit_cell = unit_cell

        self.hkl = hkl
        self.half_pixel_error = self.form_weighted_error([np.sqrt(0.5)]*3,[0]*3)
        if self.save_dir is not None:
            self.write_meta_data(param)
        self.iterations_info = np.zeros( (self.maxiter+1, 7) )
        self.headers = ["iteration", "cost", "gamma","norm of step","worst fitted peak","peaks used (%)","time (min)"]
        self.iteration_file = 'iteration_info'
        self.solutions = 'solutions'
        if self.save_dir is not None:
            with open(self.save_dir +"/"+ self.iteration_file, 'w') as fn:
                for i,head in enumerate(self.headers):
                    fn.write(head.replace(" ","")+"="+str(i)+"\n")
                fn.write("\n")
            os.mkdir(self.save_dir+'/'+self.solutions)
            np.save( self.save_dir+'/'+self.solutions+'/initial_guess', np.asarray(self.initial_guess) )



    def print_and_save_iteration_info(self, x_curr, residuals, Jacobian, gradient, bounds_low, bounds_high, iteration, cost, gamma, norm_step, worst_peak, peaks_used, time  ):

        self.iterations_info[iteration-1,0] = iteration
        self.iterations_info[iteration-1,1] = cost
        self.iterations_info[iteration-1,2] = gamma
        self.iterations_info[iteration-1,3] = norm_step
        self.iterations_info[iteration-1,4] = worst_peak
        self.iterations_info[iteration-1,5] = peaks_used
        self.iterations_info[iteration-1,6] = time

        if self.save_dir is not None:
            with open(self.save_dir +"/"+ self.iteration_file,'a') as fn:
                fn.write("iter_"+str(iteration)+"=[")
                for val in self.iterations_info[iteration-1,:]:
                    fn.write(str(val)+",")
                fn.write("]\n")
            np.save( self.save_dir+'/'+self.solutions+'/solution_iteration_'+str(iteration), np.asarray(x_curr) )
            np.save( self.save_dir+'/'+self.solutions+'/residuals_iteration_'+str(iteration), np.asarray(residuals) )
            np.save( self.save_dir+'/'+self.solutions+'/jacobian_iteration_'+str(iteration), np.asarray(Jacobian) )
            np.save( self.save_dir+'/'+self.solutions+'/gradient_iteration_'+str(iteration), np.asarray(gradient) )

        table = tabulate(self.iterations_info[0:iteration,:], self.headers, tablefmt="fancy_grid")

        print(table)

    def terminate( self, step_size ):

        if self.iter>self.maxiter:
            print("Termination due to: Maximum iteration count")
            return True

        if step_size<self.steplength_termination and self.iter!=1:
            print("Termination due to: step_size = " + str(step_size))
            return True

        return False

    def steepest_descent(self, bounds_low, bounds_high, initial_guess):


        self.bounds_low = bounds_low
        self.bounds_high = bounds_high

        print("Preparing for Steepest Descent minimisation.")
        print("Error size is ", self.half_pixel_error)
        print("This corresponds to corner to center distance in a framstack pixel")
        t1 = time.clock()
        self.iter += 1

        x_curr = initial_guess

        self.setup_hkls(x_curr)

        self.no_peaks = 0
        for dty in self.measured_data:
            self.no_peaks+=len(dty)
        self.no_variables = len(initial_guess)


        jac = jacobian_finder.JacobianFinder(self.run, self.no_peaks, self.no_variables, self.C, self.constraint)
        jacobian_callable = jac.find_jacobian_in_parallel

        print("Evaluating residuals of current state...")
        self.current_error = self.run(x_curr, 0, self.no_voxels,gradient_mode='No',save_dir=self.save_dir)
        #print("shapes ",self.C.shape, x_curr.shape, self.constraint.shape)
        #penalty = np.ones(self.current_error.shape)*((np.max([0, np.max(np.abs(np.dot(self.C,x_curr)) - self.constraint)]) )**2)
        Objective_function = 0.5*np.dot(self.current_error,self.current_error)# + np.dot(penalty,penalty)
        max_residual = np.max(self.current_error)

        print("Evaluating Jacobian of current state...")
        Jac = jacobian_callable(x_curr)
        grad_curr= np.dot(np.transpose(Jac),self.current_error)
        gamma = 0.00001/max(abs(grad_curr))
        x_old = x_curr
        x_curr = x_curr - gamma*grad_curr

        # Apply bounds
        x_curr[ x_curr < bounds_low ]= bounds_low[ x_curr < bounds_low ]
        x_curr[ x_curr > bounds_high ]= bounds_high[ x_curr > bounds_high ]
        if np.sum(x_curr < bounds_low)+np.sum( x_curr > bounds_high)>0:
            print("x_curr > bounds_low",x_curr > bounds_low)
            print("x_curr < bounds_high",x_curr < bounds_high)
            raise
        t2 = time.clock()
        self.print_and_save_iteration_info( x_curr, self.current_error, Jac, grad_curr, bounds_low, bounds_high, self.iter, np.round(Objective_function,5), gamma, np.linalg.norm(x_old-x_curr), max_residual, np.round(100*self.no_peaks/float(self.total_number_of_measured_peaks),4), np.round((t2-t1)/60.,3)  )
        # print("iteration          cost              gamma                      norm step                 worst peak           peaks used       time (min)")
        # print("    ",self.iter,"        ",np.round(Objective_function,5),"    ",gamma,"   ",np.linalg.norm(x_old-x_curr),"   ",max_residual,"         ",np.round(100*self.no_peaks/float(self.total_number_of_measured_peaks),4),"%      ",np.round((t2-t1)/60.,3))
        # print("total no peaks: ", self.total_number_of_measured_peaks)
        # print("no peaks used: ", self.no_peaks)

        # raise
        # no_lost=0
        # for dty in self.lost_peaks:
        #     no_lost+=len(dty)
        # print("total no lost peaks: ", no_lost)
        # raise

        step_size = np.linalg.norm(x_old-x_curr)
        while( not self.terminate( step_size ) ):
        #while(self.theoretical_error<max_residual and self.maxiter>self.iter):
            t1 = time.clock()
            self.iter += 1

            self.setup_hkls(x_curr)

            self.no_peaks = 0
            for dty in self.measured_data:
                self.no_peaks+=len(dty)

            #self.no_variables = len(x_curr)

            jac = jacobian_finder.JacobianFinder(self.run, self.no_peaks, self.no_variables, self.C, self.constraint )
            jacobian_callable = jac.find_jacobian_in_parallel


            print("Evaluating residuals of current state...")
            self.current_error = self.run(x_curr,0,self.no_voxels,gradient_mode='No',save_dir=self.save_dir)
            #penalty = np.ones(self.current_error.shape)*((np.max([0, np.max(np.abs(np.dot(self.C,x_curr)) - self.constraint)]) )**2)
            Objective_function = 0.5*np.dot(self.current_error,self.current_error)# + np.dot(penalty,penalty)
            #Objective_function = 0.5*np.dot(self.current_error,self.current_error)
            max_residual = np.max(self.current_error)


            print("Evaluating Jacobian of current state...")
            Jac = jacobian_callable(x_curr)
            grad_old = grad_curr
            grad_curr = np.dot(np.transpose(Jac),self.current_error)
            gamma = np.dot( np.transpose(x_curr-x_old), (grad_curr-grad_old) )/ np.dot(grad_curr-grad_old,grad_curr-grad_old)
            x_old = x_curr
            x_curr = x_curr - gamma*grad_curr


            # Apply bounds
            x_curr[ x_curr < bounds_low ]= bounds_low[ x_curr < bounds_low ]
            x_curr[ x_curr > bounds_high ]= bounds_high[ x_curr > bounds_high ]
            if np.sum(x_curr < bounds_low)+np.sum( x_curr > bounds_high)>0:
                print("x_curr > bounds_low",x_curr > bounds_low)
                print("x_curr < bounds_high",x_curr < bounds_high)
                raise
            t2 = time.clock()
            step_size = np.linalg.norm(x_old-x_curr)
            self.print_and_save_iteration_info(x_curr, self.current_error, Jac, grad_curr, bounds_low, bounds_high, self.iter, np.round(Objective_function,5), gamma, step_size, max_residual, np.round(100*self.no_peaks/float(self.total_number_of_measured_peaks),4), np.round((t2-t1)/60.,3)  )
            #print("    ",self.iter,"        ",np.round(Objective_function,5),"    ",gamma,"   ",np.linalg.norm(x_old-x_curr),"   ",max_residual,"         ",np.round(100*self.no_peaks/float(self.total_number_of_measured_peaks),4),"%      ",np.round((t2-t1)/60.,3))

            # print("--------")
            # print("theoretical_error ",self.theoretical_error)
            # print("max_residual ", max_residual)
            # print("iteration: ", self.iter)

        return x_old

    def setup_hkls(self, grain_state):
        print("Setting up peak sets to match...")
        self.lost_peaks = None
        self.UB_matrices = self.get_ub_matrices_from_strain( grain_state )
        self.forward_model(0, self.no_voxels, self.UB_matrices)

        merged_reflections = self.peak_merger.analyse_images_as_clusters(self.iter, save_dir=None)

        simulated_data = self.sort_merged_reflections(merged_reflections)

        #Set things back in order:
        for i in range(len(self.voxel)):
            self.voxel[i] = None

        self.measured_data = copy.deepcopy(self.original_measured_data)
        #print("Forward modeled simulated data set for current state...")

        #print("Running peak set matching algorithm...")

        self.measured_data, simulated_data, self.lost_peaks = hkl_manager.match_all_data( self.measured_data,\
                                                                                              simulated_data,\
                                                                                              self.form_weighted_error,\
                                                                                              self.nmedian,\
                                                                                              self.threshold )

        all_ok = peak_debug_checker.check_peaks(self.measured_data,\
                                                    simulated_data,\
                                                    ymin=self.ymin,\
                                                    no_yscans=self.no_yscans,\
                                                    beam_width=self.beam_width,\
                                                    lost_peaks=self.lost_peaks,\
                                                    tot_no_peaks=self.total_number_of_measured_peaks)

        if not all_ok:
            print("Failed to match measured and simulated peak sets")
            raise
        else:
            print("All seems to be in order...")



    # def setup_hkls(self, grain_state):
    #     '''
    #     Construct a list of all peaks which are missing from
    #     the simulated peak set compared to the inputted measured one
    #     the missing peaks are removed from the measured data set during
    #     inverse reconstruction.
    #     '''

    #     #print("Setting up hkls for inverse reconstruction")
    #     #Forward model and get the data
    #     self.lost_peaks = None
    #     UB_matrices = self.get_ub_matrices_from_strain( grain_state )
    #     self.forward_model(0, self.no_voxels, UB_matrices)
    #     # print("no_voxels",self.no_voxels)
    #     # print("UB_matrices",UB_matrices)
    #     # print("beam_width",self.beam_width)
    #     # print("voxel[0].refs",self.voxel[0].refs)
    #     # print("voxel[1].refs",self.voxel[1].refs)
    #     # print("hkls")
    #     merged_reflections = self.peak_merger.analyse_images_as_clusters()
    #     # for dty in merged_reflections:
    #     #     print(len(dty))
    #     # print(self.hkl[0])
    #     # print(self.voxel_positions)
    #     # raise
    #     simulated_data = self.sort_merged_reflections(merged_reflections)

    #     #Set things back in order:
    #     #self.voxel = [None]*self.no_voxels
    #     for i in range(len(self.voxel)):
    #         self.voxel[i] = None

    #     #Define missmatch between peaks sets
    #     self.measured_data = copy.deepcopy(self.original_measured_data)
    #     self.lost_peaks = hkl_manager.find_missing_peaks(simulated_data, self.measured_data, self.hkl)


    #     # for peak in self.lost_peaks:
    #     #     print(peak)
    #     # print(self.measured_data[9])
    #     # print("--")
    #     # print(simulated_data[9])
    #     #raise
    #     hkl_manager.remove_missing_peaks(self.measured_data, self.lost_peaks)
    #     hkl_manager.remove_missing_peaks(simulated_data, self.lost_peaks)


    #     peak_debug_checker.check_peaks( self.hkl, \
    #                                     self.measured_data,\
    #                                     simulated_data,\
    #                                     ymin=self.ymin,\
    #                                     no_yscans=self.no_yscans,\
    #                                     beam_width=self.beam_width,\
    #                                     lost_peaks=self.lost_peaks,\
    #                                     tot_no_peaks=self.total_number_of_measured_peaks)
    #     before=0
    #     after=0
    #     for dty in self.lost_peaks:
    #         before +=len(dty)
    #     print("before",before)
    #     hkl_manager.remove_outliners(5, self.form_weighted_error,self.dety_tol,self.detz_tol,self.om_tol, simulated_data, self.measured_data, self.lost_peaks)
    #     for dty in self.lost_peaks:
    #         after +=len(dty)
    #     print("after",after)
    #     hkl_manager.remove_missing_peaks(self.measured_data, self.lost_peaks)
    #     hkl_manager.remove_missing_peaks(simulated_data, self.lost_peaks)
    #     # print(simulated_data[60])
    #     # print("")
    #     # print(self.lost_peaks[60])
    #     # raise

    #     # for dtym,dtys in zip(self.measured_data,simulated_data):
    #     #     print(dtym.shape,dtys.shape)

    #     peak_debug_checker.check_peaks( self.hkl, \
    #                                     self.measured_data,\
    #                                     simulated_data,\
    #                                     ymin=self.ymin,\
    #                                     no_yscans=self.no_yscans,\
    #                                     beam_width=self.beam_width,\
    #                                     lost_peaks=self.lost_peaks,\
    #                                     tot_no_peaks=self.total_number_of_measured_peaks)

    #     # for i in range(len(self.voxel)):
    #     #     self.voxel[i] = None
    #     # UB_matrices_1 = self.get_ub_matrices_from_strain(self.initial_guess)
    #     # self.forward_model(0, self.no_voxels, UB_matrices_1)
    #     # merged_reflections_1 = self.peak_merger.analyse_images_as_clusters()
    #     # simulated_data = self.sort_merged_reflections(merged_reflections)
    #     # error = self.compute_error(simulated_data)
    #     # print("Scalar error : ",np.dot(error,error))
    #     # print("Worst fitted peak: ", np.max(error))

    #     # for i in range(len(self.voxel)):
    #     #     self.voxel[i] = None
    #     # UB_matrices_2 = self.get_ub_matrices_from_strain(self.initial_guess*234.234+1000000000)
    #     # self.forward_model(0, self.no_voxels, UB_matrices_2)
    #     # merged_reflections_2 = self.peak_merger.analyse_images_as_clusters()
    #     # simulated_data = self.sort_merged_reflections(merged_reflections)
    #     # error = self.compute_error(simulated_data)
    #     # print("Scalar error : ",np.dot(error,error))
    #     # print("Worst fitted peak: ", np.max(error))

    #     # # for i,val in enumerate(UB_matrices_1):
    #     # #     print("1",val)
    #     # #     print("2",UB_matrices_2[i])
    #     # for dty1,dty2 in zip(merged_reflections_1,merged_reflections_2):
    #     #     for p1,p2 in zip(dty1,dty2):
    #     #         if abs(p1[0]-p2[0]+p1[1]-p2[1])>0:
    #     #             print(abs(p1[0]-p2[0]+p1[1]-p2[1]))
    #     # raise





    def pick_merge_tolerance(self,safety_factor_det=0.2,safety_factor_om=0.2, no_discarded=0):
        '''
        Make a simple iteration to find a suitable merging tolerance
        '''

        om_tol=3*self.omega_step/safety_factor_om
        dety_tol=100./safety_factor_det
        detz_tol=100./safety_factor_det

        # print("type(self.measured_data[0])",type(self.measured_data[0]))
        # print("self.measured_data[0]",self.measured_data[0])
        def eval_tol(om_tol,dety_tol,detz_tol):
            for k,dty in enumerate(self.measured_data):
                for i in range(len(dty)-1):
                    peak = dty[i]
                    for j in range(i+1,len(dty)):
                        p = dty[j]
                        dety_diff = abs(peak[0]-p[0])
                        detz_diff = abs(peak[1]-p[1])
                        om_diff = abs(peak[2]-p[2])
                        if om_diff*safety_factor_om<om_tol:
                            if dety_diff*safety_factor_det<dety_tol:
                                if detz_diff*safety_factor_det<detz_tol:
                                    # print("p1: ",peak)
                                    # print("p2: ",p)
                                    # print("")
                                    return False, k, i, j
            return True, None, None, None

        success, dty, peak_1, peak_2 = eval_tol(om_tol,dety_tol,detz_tol)
        while( not success ):
            if dety_tol<15 and detz_tol<15 and om_tol<1.5:
                print("Some peaks are too close together, they risk being merged together")
                print("dety_tol,detz_tol,om_tol")
                print(dety_tol,detz_tol,om_tol)
                print("Trying to remove peak pair at dty ", dty," and rerun: ")
                print("Peak pair:")
                print("peak_1: ",self.measured_data[dty].pop(peak_2))
                print("peak_2: ",self.measured_data[dty].pop(peak_1))
                return self.pick_merge_tolerance( safety_factor_det=0.2, safety_factor_om=0.2, no_discarded=no_discarded+1 )
                #raise KeyboardInterrupt

            if om_tol>1.5:
                om_tol = om_tol*0.95
            if dety_tol>15:
                dety_tol = dety_tol*0.95
            if detz_tol>15:
                detz_tol = detz_tol*0.95

            success, dty, peak_1, peak_2 = eval_tol(om_tol,dety_tol,detz_tol)

        print("")
        print("Computed tolerances as:")
        print("dety tol: ",dety_tol)
        print("detz tol: ",detz_tol)
        print("omega tol: ",om_tol)
        print("discarded ", no_discarded ,"peaks to achive this")
        print("")

        return dety_tol,detz_tol,om_tol

    def cell_and_euler_to_ub(self,cell,euler):
        '''
        cell = [a,b,c,alpha,beta,gamma]
        euler= [phi1,PHI,phi2]
        '''

        U = tools.euler_to_u(euler[0],euler[1],euler[2])
        B = tools.form_b_mat(cell)
        return np.dot(U,B)
        # UBI = tools.u_to_ubi(U,cell)
        # return np.linalg.inv(UBI)

    def crystal_to_omega(self,grain_state):
        omega = []
        eps = np.zeros((3,3))
        for i in range(len(grain_state)//9):
            low = i*9
            mid = low+6
            high = low + 9
            euler = grain_state[mid:high]
            #[e11, e12, e13, e22, e23, e33] 
            e = grain_state[low:mid]
            eps[0,0] = e[0]
            eps[0,1] = e[1]
            eps[1,0] = e[1]
            eps[0,2] = e[2]
            eps[2,0] = e[2]
            eps[1,1] = e[3]
            eps[1,2] = e[4]
            eps[2,1] = e[4]
            eps[2,2] = e[5]           
            U = tools.euler_to_u(euler[0],euler[1],euler[2])
            eps_om = np.dot(np.dot(U.T,eps),U)
            eps_om_list = [eps[0,0], eps[0,1], eps[0,2], eps[1,1], eps[1,2], eps[2,2]]
            omega.extend( eps_om_list )
            omega.extend( euler )
        return np.asarray( omega )

    def strain_and_euler_to_ub(self,strain,euler):
        '''
        cell = [a,b,c,alpha,beta,gamma]
        euler= [phi1,PHI,phi2]
        '''

        U = tools.euler_to_u(euler[0],euler[1],euler[2])
        B = tools.epsilon_to_b(strain, self.unit_cell)
        return np.dot(U,B)

    def get_ub_matrices_from_cell(self,grain_state):
        UB_matrices = []
        for i in range(len(grain_state)//9):
            low = i*9
            mid = low+6
            high = low + 9
            UB = self.cell_and_euler_to_ub(grain_state[low:mid],grain_state[mid:high])
            UB_matrices.append(UB)
        return UB_matrices

    def get_ub_matrices_from_strain(self,grain_state):

        UB_matrices = []
        for i in range(len(grain_state)//9):
            low = i*9
            mid = low+6
            high = low + 9
            UB = self.strain_and_euler_to_ub(grain_state[low:mid],grain_state[mid:high])
            UB_matrices.append(UB)
        return UB_matrices

    def get_single_ub_from_strain(self, grain_state, voxel_nbr ):
        low = voxel_nbr*9
        mid = low+6
        high = low + 9
        UB = self.strain_and_euler_to_ub(grain_state[low:mid],grain_state[mid:high])
        return UB


    def sort_merged_reflections(self,merged_reflections):

        simulated_data = []
        for i,dty in enumerate(self.measured_data):
            simulated_data.append([])

        banned=0
        for dty in range(len(merged_reflections)):
            if len(merged_reflections[dty])>0:
                dty_curr = []
                for reflection in merged_reflections[dty]:

                    dety = reflection[voxel_reflections_id['dety']]
                    detz = reflection[voxel_reflections_id['detz']]
                    om = reflection[voxel_reflections_id['omega']]*(180./np.pi)
                    y = reflection[voxel_reflections_id['dty']]
                    h = reflection[voxel_reflections_id['h']]
                    k = reflection[voxel_reflections_id['k']]
                    l = reflection[voxel_reflections_id['l']]
                    peak = [dety,detz,om,y,h,k,l]
                    if hkl_manager.is_banned(peak, dty, self.lost_peaks) :
                        banned+=1
                    else:
                        dty_curr.append(peak)
                simulated_data[dty] = dty_curr
        #print("banned ", banned)
        simulated_data = hkl_manager.sort_data( simulated_data )
        return simulated_data


    def run(self, grain_state, start_voxel=None, end_voxel=None, gradient_mode='No', save_dir=None):
        '''
        functionalised run method for find_refl_func
        '''

        if start_voxel==None or end_voxel==None:
            if gradient_mode!='No':
                print('Decide, are you evaluating gradients or not...')
                raise ValueError
            start_voxel = 0
            end_voxel = self.no_voxels

        #UB_matrices = self.get_ub_matrices_from_cell(grain_state)


        if gradient_mode!='No':
            old_UB = copy.copy( self.UB_matrices[ start_voxel ] )
            self.UB_matrices[ start_voxel ] = self.get_single_ub_from_strain(grain_state, start_voxel)
        else:
            self.UB_matrices = self.get_ub_matrices_from_strain(grain_state)


        if gradient_mode!='No':
            path = None
            voxel_reflections, voxel_nbr = self.forward_model(start_voxel, end_voxel, self.UB_matrices, gradient_mode='Yes')
            old_refl = copy.copy(self.voxel[voxel_nbr].refs)
            self.voxel[voxel_nbr].refs = voxel_reflections
        else:
            #self.iter+=1
            path = save_dir
            self.forward_model(start_voxel, end_voxel, self.UB_matrices, gradient_mode)

        # Merge delta peak result
        if gradient_mode!='No':
            merged_reflections = self.peak_merger.analyse_images_as_clusters_gradient_mode( voxel_reflections, voxel_nbr)
        else:
            merged_reflections = self.peak_merger.analyse_images_as_clusters(self.iter, save_dir=path)
        sorted_merged = self.sort_merged_reflections(merged_reflections)
        error = self.compute_error(sorted_merged, gradient_mode)

        # put things back in order
        if gradient_mode!='No':
            self.voxel[ voxel_nbr ].refs = old_refl
            self.UB_matrices[ start_voxel ] = old_UB

        if gradient_mode=='No':
            self.plot_error(sorted_merged, self.iter, nbr_plots=5, show=False, savedir=self.save_dir)

        #scale = 100.
        #om_state = self.crystal_to_omega(grain_state)
        #error += (1/scale)*np.ones(error.shape)*(np.max([ 0, np.max( np.abs(np.dot(self.C,om_state)) - self.constraint)] )**2 )
        #error += (1/scale)*np.ones(error.shape)*(np.max([ 0, np.max( np.abs(np.dot(self.C,om_state)) - self.constraint)] )**2 )

        #if gradient_mode=='No':
        #    print("constraint violation: ",(1/scale)*np.max([ 0, np.max( np.abs(np.dot(self.C,om_state)) - self.constraint)] )**2 )
        return error


    def compute_error(self,simulated_data, gradient_mode):
        e=[]

        for i,(measured,simulated) in enumerate(zip(self.measured_data,simulated_data)):
            if len(measured)!=len(simulated):
                print("DIFFERENT SHAPES!")
                print("---------------------")
                print("measured at",i," ", len(measured))
                #print(measured)
                print("simulated at",i," ", len(simulated))
                print("simulated peaks at ",i," : ",simulated)
                #print(simulated)
                print("")
                print("lost peaks at ",i," : ",self.lost_peaks[i])

                print("--------------------")
                peak_debug_checker_new.check_peaks( self.measured_data,\
                                        simulated_data,\
                                        ymin=self.ymin,\
                                        no_yscans=self.no_yscans,\
                                        beam_width=self.beam_width,\
                                        lost_peaks=self.lost_peaks)
                raise KeyboardInterrupt

            for i,(measured_peak,simulated_peak) in enumerate(zip(measured,simulated)):

                if gradient_mode=='No':
                # when evaluating the fnction this is a bit of a debug safety check
                    indx_dety = 0
                    indx_detz = 1
                    indx_om = 2
                    index_dty = 3
                    indx_h = 4
                    indx_k = 5
                    indx_l = 6
                    if abs(measured_peak[indx_h]-simulated_peak[indx_h])>0.01 or \
                    abs(measured_peak[indx_k]-simulated_peak[indx_k])>0.01 or \
                    abs(measured_peak[indx_l]-simulated_peak[indx_l])>0.01:
                        print("hkls do not match!")
                        print("-----------------------------")
                        print("measured_peak ",measured_peak)
                        print("simulated_peak ",simulated_peak)
                        print()
                        print("measured")
                        for m in measured[i-3:i+4]:
                            print(m)
                        print()
                        print("simulated")
                        for m in simulated[i-3:i+4]:
                            print(m)
                        print("-----------------------------")
                        raise

                err = self.form_weighted_error(measured_peak,simulated_peak)
                e.append(err)

        return np.asarray(e)


    def form_weighted_error(self,measured_peak,simulated_peak):
        '''
        Produce a scalar errror based on peak difference of modeled and true peak.
        '''

        dety_err = (measured_peak[0]-simulated_peak[0])
        detz_err = (measured_peak[1]-simulated_peak[1])
        om_err = (measured_peak[2]-simulated_peak[2])

        error = np.sqrt(dety_err*dety_err+detz_err*detz_err+om_err*om_err)

        return error


    def plot_error(self, simulated_data, iteration, nbr_plots=5, show=True, savedir=None):
        '''
        Create and show error plots of nbr_plots dty scans.
        Also crate a plot of the mean error for all dtys.

        The function fills out a 3x2 grid of plots, so maximum
        number of plots is 5 and the sixth is then used for
        avergae error display.
        '''

        if nbr_plots>5:
            print('need to implement support for more plots')
            nbr_plots=5

        subplots = []
        for i in range(nbr_plots):
            subplots.append(321+i)

        avg_errors = []
        dtys = []
        props=dict(boxstyle='round',facecolor='wheat',alpha=0.4)
        fig = plt.figure()

        # for dty in simulated_data:
        #     print(dty)
        n=0
        out_simul='simulated = ['
        out_measured='measured = ['
        out_error='error = ['
        for measured,simulated in zip(self.measured_data,simulated_data):
            if len(simulated)>0:
                if n<nbr_plots:
                    ax = fig.add_subplot( subplots[n] )

                    ax.set_xlabel(r'$\omega$')
                    ax.set_ylabel('error')
                    ax.grid()
                errors_dty=0
                j = 0
                dtys.append(simulated[0][3])
                for i,(measured_peak,simulated_peak) in enumerate(zip(measured,simulated)):
                    err = self.form_weighted_error(measured_peak,simulated_peak)
                    errors_dty+=err
                    j+=1
                    if n<nbr_plots:
                        out_simul+=str(simulated_peak)+','
                        out_measured+=str(measured_peak)+','
                        out_error+=str(err)+','
                        ax.scatter(simulated_peak[2],err)
                if n<nbr_plots:
                    ax.text(0.85,0.95,'y-scan nbr: '+str(n),transform=ax.transAxes,verticalalignment='top',bbox=props)
                avg_errors.append( errors_dty/j )
                n+=1
        out_simul+=']'
        out_measured+=']'
        out_error+=']'
        ax = fig.add_subplot(subplots[-1]+1)
        ax.scatter(dtys, avg_errors)
        ax.set_xlabel('y-scan number')
        ax.set_ylabel('average error')
        ax.text(0.65,0.95,'average from all y-scans',transform=ax.transAxes,verticalalignment='top',bbox=props)
        ax.grid()

        if iteration==1:
            self.high = []
            for i,ax in enumerate(fig.get_axes()[0:-1]):
                l,h = ax.get_ylim()
                self.high.append(h)
            ax = fig.get_axes()[-1]
            l,h = ax.get_ylim()
            self.high.append(h)

        yline = self.half_pixel_error
        for ax in fig.get_axes()[0:-1]:
            ax.axhline(yline,color='red',label='Theoretical uncertainty')
            legend = ax.legend(loc='upper left', shadow=False, fontsize='small',framealpha=0.4)
            # Put a nicer background color on the legend.
            legend.get_frame().set_facecolor('wheat')

        for ax in fig.get_axes()[0:-1]:
            ax.set_ylim(0,max(self.high[0:-1])*1.15)
        ax = fig.get_axes()[-1]
        ax.set_ylim(0,self.high[-1]*1.25)

        fig.subplots_adjust(hspace=0.35)
        fig.suptitle(r'Error analysis as function of beam y-setting and $\omega$ turntable rotation: Iteration '+str(iteration),fontsize=15)


        if show:
            plt.show()

        if savedir!=None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            filename = "error_iteration_"+str(iteration)+"_("+str(time.clock())+").png"
            fig.savefig(savedir+"/"+filename)
            with open(savedir+'/error_iteration_'+str(iteration)+'.txt','w') as f:
                #peak = [dety,detz,om,y,h,k,l] (y is in microns)
                f.write('iteration = '+str(iteration)+'\n')
                f.write('# The index for the peaks follows:\n')
                f.write('dety = '+str(0)+'\n')
                f.write('detz = '+str(1)+'\n')
                f.write('omega = '+str(2)+'\n')
                f.write('dty = '+str(3)+'\n')
                f.write('h = '+str(4)+'\n')
                f.write('k = '+str(5)+'\n')
                f.write('l = '+str(6)+'\n')
                f.write('\n')
                f.write(out_measured)
                f.write('\n')
                f.write(out_simul)
                f.write('\n')
                f.write(out_error)
                f.write('\n')
        plt.close('all')




    def forward_model(self, start_voxel, end_voxel, UB_matrices, gradient_mode='No', log=True, parallel=None):
        '''
        This is the main algorithmic loop that will calculate the diffraction pattern
        based on a per-voxel definition of the sample. The algorithm deals with one voxel at the
        time, computing all reflections that would be generated for that voxel in a single go.
        Therefore, the run time is linear with voxel number regardless of the number of y-scans.

        Input: start_voxel: Used for running in parrallel, equal to 0 for 1 cpu
               end_voxel:   Used for running in parrallel, equal to number of voxels for 1 cpu
               log:         used to determine of this process should print to stdout or be quiet.
               parallel:    Flag to indicate parallel mode. Such a mode requires that we pass the result
                            of the run to the calling process.
        '''


        for voxel_nbr in range(start_voxel, end_voxel):

            voxel_reflections = []
            if gradient_mode=='No':
                self.voxel[voxel_nbr]=variables.voxel_cont()
            UB = UB_matrices[voxel_nbr]

            # print(UB.ravel())
            # raise
            #UB = UBs_true[voxel_nbr]
            voxel_pos = self.voxel_positions[voxel_nbr] #should be np array

            # th_min=90
            # th_max=0

            for hkl in self.hkl[0]:
                Gw = np.dot( self.S , np.dot( UB, hkl[0:3] ) )
                tth = tools.tth2(Gw,self.wavelength)

                costth = np.cos(tth)
                (Omega, Eta, Omega_mat, Gts) = self.find_omega_general(Gw*self.scale_Gw,self.scale_Gw,tth)

                # if hkl[0]==1 and hkl[1]==0 and hkl[2]==-3:
                #     print(Omega*180./np.pi)
                #     raise
                for omega,eta,Om,Gt in zip(Omega, Eta, Omega_mat, Gts):
                    if  self.omega_start < omega and omega < self.omega_end:

                        # Calc crystal position at present omega
                        [tx,ty,tz]= np.dot(Om,voxel_pos)

                        #The 3 beam y positions that could illuminate the voxel
                        beam_centre_1 = np.round(ty/self.beam_width)*self.beam_width
                        beam_centre_0 = beam_centre_1+self.beam_width
                        beam_centre_2 = beam_centre_1-self.beam_width

                        #Compute precentual voxel beam overlap for the three regions
                        overlaps_info = convexPolygon.compute_voxel_beam_overlap(tx,ty,omega,beam_centre_1, self.beam_width)

                        regions = [ beam_centre_0, beam_centre_1,beam_centre_2]
                        for beam_centre_y,info in zip(regions,overlaps_info):

                            intensity_modifier = info[0]

                            if intensity_modifier==0:
                                continue

                            #Lorentz factor
                            if self.lorentz_apply == 1:
                                if eta != 0:
                                    L=1/(np.sin(tth)*abs(np.sin(eta)))
                                else:
                                    L=np.inf;
                            else:
                                L = 1.0

                            #intensity = L*intensity_modifier*hkl[3]
                            intensity = intensity_modifier

                            # If no unit cells where activated we continue


                            dty = beam_centre_y*1000. # beam cetre in microns
                            cx = info[1]
                            cy = info[2]

                            # Compute peak based on cms of illuminated voxel fraction
                            tx = cx
                            ty = cy - beam_centre_y # The sample moves not the beam!
                            tz = 0
                            (dety, detz) = self.det_coor(Gt,
                                                        costth,
                                                        self.wavelength,
                                                        self.distance,
                                                        self.y_size,
                                                        self.z_size,
                                                        self.dety_center,
                                                        self.detz_center,
                                                        self.R,
                                                        tx,ty,tz)

                            # if abs(hkl[0]-5)<0.01 and abs(hkl[1]+6)<0.01 and abs(hkl[2]+3)<0.01:
                            #     print("hkl",hkl)
                            #     print(tth*180/np.pi)
                            #     print(dty)
                            #     raise

                            #If shoebox extends outside detector exclude it
                            if (-0.5 > dety) or\
                            (dety > self.dety_size-0.5) or\
                            (-0.5 > detz) or\
                            (detz > self.detz_size-0.5):
                                continue

                            dty_as_index = self.get_index_of_dty( dty )

                            voxel_reflections.append([dety,detz,intensity,omega,dty,hkl[0],hkl[1],hkl[2],voxel_nbr,dty_as_index])


            if gradient_mode=='No':
                self.voxel[voxel_nbr].refs = voxel_reflections
            else:
                return (voxel_reflections, voxel_nbr)


        #     if parallel and log==True:
        #         progress = int(100*(voxel_nbr+1-start_voxel)/float(end_voxel-start_voxel))
        #         print('\rDone approximately %3i percent' %progress, end=' ')
        #         sys.stdout.flush()
        #     elif log==True:
        #         print('\rDone approximately %3i voxel(s) of %3i' %(voxel_nbr+1,self.no_voxels), end=' ')
        #         sys.stdout.flush()

        # if log==True:
        #     print('\n')


        # if we are running in parrallel the memory cannot be shared (easily)
        # so wee return our result in order to merge all results
        # into a single find_refl.py Object.
        # print("found theta bounds")
        # print(th_min)
        # print(th_max)
        # raise

        # if parallel:

        #     result = {"start_voxel": start_voxel,
        #               "end_voxel": end_voxel,
        #               "refl_list": self.voxel}
        #     return result

        # else:
        #     pass


    def get_index_of_dty(self, dty_in_microns):
        '''
        assumes self.ymin in microns and self.beam_width in mm
        '''
         
        # if np.round( ( dty_in_microns-self.ymin )/( 1000.*self.beam_width ) ).astype(int)<0:
        #     print("get_index_of_dty",np.round( ( dty_in_microns-self.ymin )/( 1000.*self.beam_width ) ).astype(int))
        #     print("dty_in_microns",dty_in_microns)
        #     print("self.ymin",self.ymin)
        #     print("self.beam_width",self.beam_width)
        #     raise
        return np.round( ( dty_in_microns-self.ymin )/( 1000.*self.beam_width ) ).astype(int)



    def find_omega_general(self, g_w, scale_Gw, twoth):
        """
        For gw find the omega rotation (in radians) around an axis
        tilted by w_x radians around x (chi) and w_y radians around y (wedge)
        Furthermore find eta (in radians).

        Output: Omega, Eta, Omega rotation matrix, Gt (G vector in lab frame)

        A similar method can be found in the xfab module, here it has been modified for
        computational efficency. This also means that it could be hard to read, see xfab
        for more easily read code.
        """

        a = g_w[0]*self.r_mat[0][0] + g_w[1]*self.r_mat[0][1]
        b = g_w[0]*self.r_mat[1][0] - g_w[1]*self.r_mat[0][0]
        c = - np.dot(g_w, g_w) - g_w[2]*self.r_mat[0][2]
        d = a*a + b*b - c*c

        omega = []
        eta = []
        omega_mat = []
        Gt = []

        if d < 0:
            pass
        else:
            # Compute numbers that are reused twice for speed
            #-------------------------------------------------
            sq_d = np.sqrt(d)
            sinfact = 2/np.sin(twoth)
            ac = a*c
            bc = b*c
            bsqd = b*sq_d
            asqd = a*sq_d

            # Find Omega number 0
            #-------------------------------------------------
            omega.append(np.arctan2((bc - asqd), (ac + bsqd)))
            if omega[0] > np.pi:
                omega[0] = omega[0] - 2*np.pi

            cosOm = np.cos(omega[0])
            sinOm = np.sin(omega[0])
            omega_mat.append(np.dot(self.r_mat,np.array([[cosOm, -sinOm,  0],
                                                       [sinOm,  cosOm,  0],
                                                       [  0,  0  ,  1]])))
            g_t = np.dot(omega_mat[0], g_w)
            sineta = -g_t[1]*sinfact
            coseta =  g_t[2]*sinfact
            eta.append(np.arctan2(sineta, coseta))
            Gt.append(g_t/scale_Gw)

            #Find omega number 1
            #-------------------------------------------------
            omega.append(np.arctan2((bc + asqd), (ac - bsqd)))
            if omega[1] > np.pi:
                omega[1] = omega[1] - 2*np.pi
            cosOm = np.cos(omega[1])
            sinOm = np.sin(omega[1])
            omega_mat.append(np.dot(self.r_mat,np.array([[cosOm, -sinOm,  0],
                                                       [sinOm,  cosOm,  0],
                                                       [  0,  0  ,  1]])))
            g_t = np.dot(omega_mat[1], g_w)
            sineta = -g_t[1]*sinfact
            coseta =  g_t[2]*sinfact
            eta.append(np.arctan2(sineta, coseta))
            Gt.append(g_t/scale_Gw)
            #-------------------------------------------------

        return np.array(omega), np.array(eta), omega_mat, Gt


    def det_coor(self, Gt, costth, wavelength, distance, y_size, z_size,
             dety_center, detz_center, R_tilt, tx,ty,tz):
        """
        Calculates detector coordinates dety,detz

        INPUT:
        Gt is the g-vector
        y_size and z_size are the detector pixel size in y, and z (in microns)
        (dety_center, detz-center) is the beam center of the detector (in pixels)
        R_tilt is the rotation matrix of the detector
        (tx, ty, tz) is the position of the voxel at the present omega

        OUTPUT:
        [dety, detz]
        """

        # Unit directional vector for reflection
        v = np.array([costth, wavelength/(2*np.pi)*Gt[1], wavelength/(2*np.pi)*Gt[2]])
        t = (R_tilt[0, 0]*distance - np.dot(R_tilt[:, 0],np.array([tx, ty, tz])))/np.dot(R_tilt[:, 0],v)
        Ltv = np.array([tx-distance, ty, tz])+ t*v
        dety = np.dot(R_tilt[:, 1],Ltv)/y_size + dety_center
        detz = np.dot(R_tilt[:, 2],Ltv)/z_size + detz_center
        return [dety, detz]


    def write_meta_data(self,param):
        with open(self.save_dir+'/meta_data','w') as f:
            f.write('\n')
            f.write('Beam width (microns): '+str(self.beam_width*1000.)+'\n')
            f.write('Number of voxels: '+str(self.no_voxels)+'\n')
            f.write('Maximum iterations: '+str(self.maxiter)+'\n')
            f.write('Steplength termniation size: '+str(self.steplength_termination)+'\n')
            f.write('dety tolerance for merging: '+str(self.dety_tol)+' pixels\n')
            f.write('detz tolerance for merging: '+str(self.detz_tol)+' pixels\n')
            f.write('omega tolerance for merging: '+str(self.om_tol)+' frames\n')
            f.write('number of medians outliners: '+str(self.nmedian)+' times media error\n')
            f.write('threshold for outliners: '+str(self.threshold)+' pixels\n')
            f.write('\n')
            for key in param:
                f.write(key+': '+str(param[key])+'\n')


