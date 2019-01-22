#!/usr/bin/env python
'''
gomodelscanning3DXRD.py is an administrative script used to start
the forward model which is found in find_refl.py. This script handles for instance:
parallel runs, profiling, input verification and file writing initialization.
'''
from __future__ import absolute_import
from __future__ import print_function
import sys,os
import numpy as n
import multiprocessing as mp
from modelscanning3DXRD import check_input
from modelscanning3DXRD import file_io
from modelscanning3DXRD import find_refl
from modelscanning3DXRD import generate_voxels
from modelscanning3DXRD import make_image
from modelscanning3DXRD import make_imagestack
from modelscanning3DXRD import reflections
from modelscanning3DXRD import help_input
import cProfile, pstats, StringIO
from optparse import OptionParser

def get_options(parser):
    '''
    Parses the input given from terminal execution of ModelScanning3DXRD.py.
    If you want to add extra input options to ModelScanning3DXRD.py this is the
    place to do so. Don't forget to also add your extra options to the arguments
    of ModelScanning3DXRD.py main() and gomodelscanning3DXRD.run().
    '''

    parser = OptionParser()
    parser.add_option("-i", "--input", action="store",
                      dest="filename", type="string",
                      help="Name of the file containing the input parameters")
    parser.add_option("-d", "--debug", action="store_true",
                      dest="debug",default =False,
                      help="Run in debug mode")
    parser.add_option("-p", "--print", action="store_true",
                      dest="print_input",default =False,
                      help="Show input parameters and syntax")
    parser.add_option("-k","--killfile", action="store",
                      dest="killfile", default=None, type="string",
                      help="Name of file to create halt ModelScanning3DXRD")
    parser.add_option("-t", "--profile", action="store",
                      dest="profile", default=None, type="string",
                      help="Run and profile main algorithmic loop")
    parser.add_option("-c", "--parallel", action="store_true",
                      dest="parallel",default =False,
                      help="Run in parallel for speed, good for large number of voxels")

    options , args = parser.parse_args()

    return options

def run(print_input,filename,killfile,profile,debug,parallel):
    '''
    The gomodelscanning3DXRD.run() method will intalize the voxels from the supplied
    input file and determine if all is good to go. This method is the administrator
    of find_refl.py and will handle things like parallel runs and profiling. From this
    method find_refl.run() will be started in one way or another, actually running the
    forward diffraction model.
    '''

    # Check for print_input
    try:
        print_input
    except:
        print_input = None
    if print_input:
        print(help_input.show_input())
        sys.exit()

    # Check if filename is specified
    try:
        filename
    except:
        filename = None
    if filename == None:
        print("\nNo input file supplied [-i filename]\n")
        sys.exit()

    # Check killfile does not exist
    try:
        killfile
    except:
        killfile = None
    if killfile is not None and os.path.exists(killfile):
        print("The purpose of the killfile option is to create that file")
        print("only when you want ModelScanning3DXRD to stop")
        print("If the file already exists when you start ModelScanning3DXRD, it is")
        print("stopped immediately")
        raise ValueError("Your killfile "+killfile+" already exists")

    # Read and check input
    print('Reading input\n')
    # create input object
    myinput = check_input.parse_input(input_file=filename)
    # read input from file
    myinput.read()
    print('Checking input\n')
    # check validity of input
    myinput.check()
    check_input.interrupt(killfile)

    if len(myinput.errors) > 0:
        myinput.show_errors()
        sys.exit()

    print('Initialize parameters etc\n')
    # if ok initialize
    myinput.initialize()
    check_input.interrupt(killfile)

    # Generate reflections
    hkl = []

    #TODO: implement and test for multiphase samples
    for phase in myinput.param['phase_list']:
        if  ('structure_phase_%i' %phase) in myinput.param:
            xtal_structure = reflections.open_structure(myinput.param,phase)
            print('Generating miller indices')
            hkl_tmp = reflections.gen_miller(myinput.param,phase)
            if myinput.param['structure_factors'] != 0:
                print('Structure factor calculation')
                hkl.append(reflections.calc_intensity(hkl_tmp,
                                                      xtal_structure,
                                                      killfile))
            else:
                hkl.append(reflections.add_intensity(hkl,myinput.param))
                print('No structure factor calculation')
        else:
            hkl_tmp = reflections.gen_miller(myinput.param,phase)
            hkl.append(reflections.add_intensity(hkl_tmp,myinput.param))

        check_input.interrupt(killfile)

    # generate random voxel quanteties of so specified in input file
    generate_voxels.generate_voxels(myinput.param)
    check_input.interrupt(killfile)

    # Save the true state of the sample to file, these files can
    # be usefull for plotting the true state of the sample
    print('Write voxels file')
    file_io.write_voxels(myinput.param)
    check_input.interrupt(killfile)
    print('Write res file')
    file_io.write_res(myinput.param)
    check_input.interrupt(killfile)

    if '.hkl' in myinput.param['output']:
        print('Write hkl file')
        file_io.write_hkl(myinput.param,hkl)
    if '.fcf' in myinput.param['output']:
        print('Write fcf file')
        file_io.write_fcf(myinput.param,hkl)
    if '.ubi' in myinput.param['output']:
        print('Write UBI file')
        file_io.write_ubi(myinput.param)
    if '.par' in myinput.param['output']:
        print('Write detector.par file')
        file_io.write_par(myinput.param)
    check_input.interrupt(killfile)

    # Initiate a find_refl object, this object will later
    # be used to execute the forward model upon
    voxeldata = find_refl.find_refl(myinput.param,hkl,killfile)
    voxeldata.frameinfo = myinput.frameinfo

    # It is now time to execute the forward model upon voxeldata as
    # voxeldata.run(). There exists three cases: either we are in profiler mode,
    # or we are in parallel mode, or we are in normal mode. Depending on mode
    # different setup is needed.

    if profile and parallel:
        print("profiling must not be done in a parallel run")
        print("The keywords --parallel and --profile are not compatible")
        raise

    print('Determine reflections positions')
    if profile:
        # PROFILER MODE

        print('-----------------')
        print('')
        print('In profiler mode!')
        print('')
        print('-----------------')
        function_to_profile = voxeldata.run
        profileFile = profile
        #Run simulation and profile it. Save report to profileFile and print it in stdout.
        log = True
        pr = cProfile.Profile()
        pr.enable()
        voxeldata.run(0,myinput.param['no_voxels'], log)
        pr.disable()
        report = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=report).strip_dirs().sort_stats(sortby)
        ps.print_stats()
        #show report
        print(report.getvalue())
        #write report to text file
        with open(profileFile, 'w+') as f:
            f.write(report.getvalue())
    else:
        if parallel:
            # PARALLEL MODE

            # Grab all but two cores of the computer.
            # Maybe you want to use all cpus, then change to
            # cpus = mp.cpu_count(). This may cause laggy
            # behaviour for other processes though.
            cpus = mp.cpu_count() - 2

            #Distribute voxels over active cpus as evenly as possible
            procs = []
            voxels_per_cpu = myinput.param['no_voxels']/float(cpus)

            print("Distributing voxels over cpus")
            print("Average number of voxels per cpu: ",voxels_per_cpu)

            if voxels_per_cpu.is_integer():
                # if the voxels can be evenly distributed
                voxels_per_cpu = int(voxels_per_cpu)
                for i in range(cpus):
                    start_voxel = i*voxels_per_cpu
                    end_voxel = start_voxel + voxels_per_cpu
                    procs.append([start_voxel,end_voxel])
            else:
                # if the voxels cannot be evenly distributed
                for i in range(cpus-1):
                    start_voxel = int( n.round( i*n.round(voxels_per_cpu) ) )
                    end_voxel = int( n.round( start_voxel + n.round(voxels_per_cpu) ) )
                    if end_voxel>=myinput.param['no_voxels']:
                        break
                    procs.append([start_voxel,end_voxel])
                start_voxel = procs[-1][1]
                end_voxel = myinput.param['no_voxels']
                procs.append([start_voxel,end_voxel])

            def startproc(the_voxeldata,start_voxel, end_voxel, log, result_queue):
                result_queue.put(the_voxeldata.run(start_voxel, end_voxel, log, parallel))

            #Start a parallel proccess for each active cpu
            running_procs = []
            result_queue = mp.Queue()
            for proc in procs:
                start_voxel = proc[0]
                end_voxel = proc[1]
                if proc==procs[-1]:
                    log = True
                else:
                    log = False
                running_procs.append(mp.Process(target=startproc, args=(voxeldata,start_voxel, end_voxel, log, result_queue,)))
                print("Started paralell process for voxel number: ",start_voxel," to ",end_voxel)
                running_procs[-1].start()

            # Wait for all processes to finish
            # Collect results as they finish when done
            results = []
            while True:
                try:
                    result = result_queue.get(False, 0.01)
                    results.append(result)
                except:
                    pass
                allExited = True
                for proc in running_procs:
                    if proc.exitcode is None:
                        allExited = False
                        break
                if allExited & result_queue.empty():
                    break

            # Merge the results to the voxeldata object
            for result in results:
                start_voxel = result["start_voxel"]
                end_voxel = result["end_voxel"]
                refl_list = result["refl_list"]
                for voxel_nbr in range(start_voxel,end_voxel):
                    voxeldata.voxel[voxel_nbr] = refl_list[voxel_nbr]
        else:
            # NORMAL MODE

            log = True
            voxeldata.run(0,myinput.param['no_voxels'], log)

        # Save the diffraction results to file
        if not os.path.exists(myinput.param['direc']):
            os.makedirs(myinput.param['direc'])
        if '.ref' in myinput.param['output']:
            print('Write reflection file')
            voxeldata.save()
        if '.gve' in myinput.param['output']:
            print('Write g-vector file')
            voxeldata.write_gve()
        if '.ini' in myinput.param['output']:
            print('Write voxelSpotter ini file - Remember it is just a template')
            voxeldata.write_ini()
        if '.flt' in myinput.param['output']:
            print('Write filtered peaks file')
            voxeldata.write_flt()

        #OBS: Not yet adapted for ModelScanning3DXRD!
        if myinput.param['make_image'] == 1:
            if  myinput.param['peakshape'][0] == 2:
                image = make_imagestack.make_image(voxeldata,killfile)
                image.setup_odf()
                image.make_image_array()
                image.make_image()
                image.correct_image()
            else:
                image = make_image.make_image(voxeldata,killfile)
                image.make_image()
