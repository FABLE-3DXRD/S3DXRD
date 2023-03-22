import numpy as np
from . import peak_debug_checker
import copy
import time
import random

def covert_to_list( data ):
    data_as_list=[]
    for dty in data:
        dty_updated=[]
        for peak in dty:
            dty_updated.append( list(peak) )
        data_as_list.append(dty_updated)
    return data_as_list

def sort_data( data ):
    '''
    Sort measured and simulated based on
    h, k ,l , omega, detz, dety
    in that order of priority
    '''
    data_sorted=[]
    for dty in data:
        dty_sorted = sorted( dty, key=lambda x: (x[4], x[5], x[6], x[2], x[1], x[0]) )
        data_sorted.append(dty_sorted)

    return data_sorted




def remove_outliners(measured, simulated, lost_peaks, error_function, nmedian, threshold):
    '''
    Requiers sorted and matched sets
    '''

    measured_updated = []
    simulated_updated = []
    lost_peaks_updated = copy.deepcopy( lost_peaks )
    errors = []
    for dtym,dtys in zip(measured,simulated):
        for m,s in zip(dtym,dtys):
            errors.append( error_function(m,s) )
    if len(errors)==0:
        print(measured)
        print(simulated)
        raise
    median_error = np.median(errors)
    print("Looking for outliners based on ",nmedian," times the median error value firstly")
    print("and based on thershold, ",threshold," secondly.")
    outliners_m=[]
    outliners_s=[]
    for i,(dtym,dtys) in enumerate(zip(measured,simulated)):
        measured_updated.append([])
        simulated_updated.append([])
        for m,s in zip(dtym,dtys):
            error = error_function(m,s)
            if error>(nmedian*median_error) and error>threshold:
                outliners_m.append( copy.copy(m) )
                outliners_s.append( copy.copy(s) )
                lost_peaks_updated[i].append( copy.copy(m) )
                lost_peaks_updated[i].append( copy.copy(s) )
            else:
                measured_updated[i].append(m)
                simulated_updated[i].append(s)
    print("Found and removed ouliners: ",len(outliners_m))
    for om,os in zip(outliners_m,outliners_s):
        print("outliner measured",om)
        print("utliner simulated",os)
        print("")
    lost_peaks_updated = sort_data( lost_peaks_updated )
    return measured_updated, simulated_updated, lost_peaks_updated






def match_all_data(measured, simulated, error_function, nmedian, threshold):
    '''
    Match the peak sets from all y-scan settings
    '''
    measured_updated = []
    simulated_updated = []
    lost_peaks=[]
    for dty_m,dty_s in zip(measured, simulated):
        discarded_peaks=[]
        if len(dty_m)>0 or len(dty_s)>0:
            dty_m, dty_s, discarded_peaks = match_single_dty(dty_m, dty_s)
        lost_peaks.append(discarded_peaks)
        measured_updated.append(dty_m)
        simulated_updated.append(dty_s)
    measured_updated = sort_data( measured_updated )
    simulated_updated = sort_data( simulated_updated )
    return remove_outliners(measured_updated, simulated_updated, lost_peaks, error_function, nmedian, threshold)



def match_single_dty(dty_1, dty_2):
    '''
    Match the peak sets of a single y-scan setting
    '''
    dty_1_updated = []
    dty_2_updated = []
    discarded_peaks = []
    for hkl in get_hkls(dty_1, dty_2):
        hkl_1 = get_peaks_for_hkl(dty_1, hkl[0],hkl[1],hkl[2] )
        hkl_2 = get_peaks_for_hkl(dty_2, hkl[0],hkl[1],hkl[2] )
        while( len(hkl_1)!=len(hkl_2) ):
            hkl_1, hkl_2, discarded_peak = discard_worst_fit(hkl_1, hkl_2)
            discarded_peaks.append(discarded_peak)
        for peak in hkl_1:
            dty_1_updated.append(peak)
        for peak in hkl_2:
            dty_2_updated.append(peak)

    return dty_1_updated, dty_2_updated, discarded_peaks

def get_hkls(dty_1, dty_2):
    hkls = []
    for peak in dty_1:
        hkls.append([peak[4],peak[5],peak[6]])
    for peak in dty_2:
        hkls.append([peak[4],peak[5],peak[6]])
    hkls = list(np.unique(hkls,axis=0))
    for i,hkl in enumerate(hkls):
        hkls[i] = list(hkl)
    return hkls

def get_peaks_for_hkl(peaks, h, k, l):
    matching = []
    for peak in peaks:
        if peak[4]==h and peak[5]==k and peak[6]==l:
            matching.append(peak)
    return matching


def discard_worst_fit(peaks_1, peaks_2):
    '''
    Take two lists of peaks and discard one peak
    from the longest list which is the worst match
    between the sets.

    assumes that peaks_1 and peaks_2 is of a single hkl
    '''
    len_peaks_1 = len(peaks_1)
    len_peaks_2 = len(peaks_2)
    discarded_peak = None
    if len_peaks_1>len_peaks_2:
        best_om_diffs = np.ones((len_peaks_1,))
        for i,peak in enumerate(peaks_1):
            best_om_diffs[i] = find_closest_match(peak, peaks_2)
        discarded_peak = peaks_1.pop( np.argmax(best_om_diffs) )
    elif len_peaks_2>len_peaks_1:
        best_om_diffs = np.ones((len_peaks_2,))
        for i,peak in enumerate(peaks_2):
            best_om_diffs[i] = find_closest_match(peak, peaks_1)
        discarded_peak = peaks_2.pop( np.argmax(best_om_diffs) )
    else:
        raise

    return peaks_1, peaks_2, discarded_peak

def find_closest_match(peak, peak_set):
    '''
    find the  closest match in peak_set to peak
    in omega coordinates. Return the value of the
    omega diff.
    '''
    best_match = 180.
    for p in peak_set:
        om_diff = abs(p[2]-peak[2])
        if om_diff<best_match:
            best_match = om_diff
    return best_match


def hkl_equal(peak1, peak2):
    if peak1[4]==peak2[4] and peak1[5]==peak2[5] and peak1[6]==peak2[6]:
        return True
    else:
        return False

def almost_equal(peak1, peak2):
    '''
    Check if two peaks are almost equal.
    identifies banned peaks which are drifted
    sligthly in Jacobian evaluation.
    '''

    # Check that hkls match exactly
    if peak1[4]!=peak2[4] or peak1[4]!=peak2[4] or peak1[4]!=peak2[4]:
        return False

    # Check that the peak detector coordinates is within the same pixel
    if abs(peak1[0]-peak2[0])>0.5 or abs(peak1[1]-peak2[1])>0.5:
        return False

    # Check that the omega coordinate is in the same frame
    if abs(peak1[2]-peak2[2])>0.5:
        return False

    return True


def is_banned(peak, dty_index, lost_peaks):

    if lost_peaks==None:
        return False

    for lost_peak in lost_peaks[dty_index]:
        if almost_equal(lost_peak, peak):
            return True
    return False


def print_peaks(peaks):
    for i,dty in enumerate(peaks):
        print("y index number",i)
        for p in dty:
            print(p)
        if len(dty)==0:
            print("empty")
        print("")











# TESTS
#-------------------------------------------------------------------------------
def generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length):
    data = []
    for i in range(no_yscans):
        no_peaks = np.round(max_peaks_per_y*np.random.rand()+1).astype(int)
        dty=[]
        for j in range(no_peaks):
            peak = create_peak(i,ymin,step_length)
            dty.append(peak)
        data.append(dty)
    return data

def create_peak(dty_index, ymin, step_length):
    peak = [None]*7
    peak[0] = np.random.rand()*2048.
    peak[1] = np.random.rand()*2048.
    peak[2] = np.random.rand()*180.
    peak[3] = ymin + dty_index*step_length
    peak[4] = np.round(sign()*5*np.random.rand()).astype(int)
    peak[5] = np.round(sign()*5*np.random.rand()).astype(int)
    peak[6] = np.round(sign()*5*np.random.rand()).astype(int)
    return peak

def sign():
    if np.random.rand()>0.5:
        return 1.
    else:
        return -1.

def add_extra_peaks( data , ymin, step_length):
    new_data = copy.deepcopy( data )
    for i,dty in enumerate(new_data):
        if np.random.rand()>0.4:
            for j in range(0,np.random.randint(0,10)):
                peak = create_peak(i, ymin, step_length)
                dty.append( peak )
    for dty in new_data:
        random.shuffle( dty )
    return new_data


def duplicate_hkls_as_extra_peaks( data, ymin, step_length ):
    new_data = copy.deepcopy( data )
    for i,dty in enumerate(new_data):
        if len(dty)>0:
            if np.random.rand()>0.4:
                for j in range(0,np.random.randint(0,5)):
                    peak = create_peak(i, ymin, step_length)
                    peak[4] = dty[0][4]
                    peak[5] = dty[0][5]
                    peak[6] = dty[0][4]
                    dty.append( peak )
    for dty in new_data:
        random.shuffle( dty )
    return new_data

def duplicate_hkls_equal_lengths( data, ymin, step_length ):
    new_data = copy.deepcopy( data )
    for i,dty in enumerate(new_data):
        if len(dty)>2:
            if np.random.rand()>0.4:
                dty[1][4] = dty[0][4]
                dty[1][5] = dty[0][4]
                dty[1][6] = dty[0][4]
    for dty in new_data:
        random.shuffle( dty )
    return new_data

def outliners(data, ymin, step_length ):
    new_data = copy.deepcopy( data )
    for i,dty in enumerate(new_data):
        if len(dty)>2:
            index = np.round(np.random.rand()).astype(int)
            dty[index][0] = np.random.rand()*2048.
            dty[index][1] = np.random.rand()*2048.
            dty[index][2] = np.random.rand()*180.
    for dty in new_data:
        random.shuffle( dty )
    return new_data



def test_extra_peaks(no_tests, error_function, nmedian, threshold):
    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        measured = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        simulated = add_extra_peaks( measured, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Extra peaks in simulated: FAILURE!")
            raise

    for i in range(no_tests+1):
            no_yscans = np.round(120*np.random.rand()+2).astype(int)
            step_length = 2*np.random.rand()
            max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
            ymin = -np.round(no_yscans/2.)*step_length
            simulated = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
            measured = add_extra_peaks( simulated, ymin, step_length )
            measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
            status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
            if status==False:
                print("Extra peaks in simulated: FAILURE!")
                raise

    print("Extra random peaks in simulated: OK!")


def test_duplicate_hkls(no_tests, error_function, nmedian, threshold):
    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        measured = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        simulated = duplicate_hkls_as_extra_peaks( measured, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Extra hkl duplicates in simulated: FAILURE!")
            raise

    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        simulated = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        measured = duplicate_hkls_as_extra_peaks( simulated, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Extra hkl duplicates in simulated: FAILURE!")
            raise
    print("Extra hkl duplicates in simulated: OK!")


def test_duplicate_hkls_equal_lengths(no_tests, error_function, nmedian, threshold):
    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        measured = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        simulated = duplicate_hkls_equal_lengths( measured, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Equal number peaks with hkl duplicates simulated: FAILURE!")
            raise

    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        simulated = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        measured = duplicate_hkls_equal_lengths( simulated, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Equal number peaks with hkl duplicates simulated: FAILURE!")
            raise
    print("Equal number peaks with hkl duplicates simulated: OK!")


def test_outliners(no_tests, error_function, nmedian, threshold):
    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        measured = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        simulated = outliners( measured, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Outliners test: FAILURE!")
            raise

    for i in range(no_tests+1):
        no_yscans = np.round(120*np.random.rand()+2).astype(int)
        step_length = 2*np.random.rand()
        max_peaks_per_y = np.round(50*np.random.rand()+2).astype(int)
        ymin = -np.round(no_yscans/2.)*step_length
        simulated = generate_test_data(no_yscans, max_peaks_per_y, ymin, step_length)
        measured = outliners( simulated, ymin, step_length )
        measured, simulated, lost_peaks = match_all_data(measured, simulated, error_function, nmedian, threshold)
        status = peak_debug_checker.check_peaks(measured, simulated, ymin=ymin, no_yscans=no_yscans, beam_width=step_length/1000., lost_peaks=lost_peaks)
        if status==False:
            print("Outliners test: FAILURE!")
            raise
    print("Outliners test: OK!")


def error_function(peak1,peak2):
    dety_err = abs(peak1[0]-peak2[0])
    detz_err = abs(peak1[1]-peak2[1])
    om_err = abs(peak1[2]-peak2[2])
    return np.sqrt(dety_err**2 + detz_err**2 + om_err**2)

# nmedian=5
# threshold=20*error_function([0,0,0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5,0.5,0.5])
# no_tests=1
# test_extra_peaks(no_tests, error_function, nmedian, threshold)
# test_duplicate_hkls(no_tests, error_function, nmedian, threshold)
# test_duplicate_hkls_equal_lengths(no_tests, error_function, nmedian, threshold)
# test_outliners(no_tests, error_function, nmedian, threshold)
