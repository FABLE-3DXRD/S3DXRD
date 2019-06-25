


import numpy as np


def check_peaks(measured, simulated, ymin=-15., no_yscans=121., beam_width=0.00025, lost_peaks=None,tot_no_peaks=None):
    '''

    Debug function to controll what is (or is not) wrong in the peak sets.

    If lost_peaks is specified and not None the measured and simulated are assumed
    to have been filtered based upon lost_peaks contents.

    '''

    #Check correct number of yscans
    if len(simulated)!=no_yscans or len(measured)!=no_yscans:
        print("---------------------------------------")
        print("Incorrect number of y-scans")
        print("Excpected ",no_yscans," y-scans got:")
        print("No scans in simulated: ", len(simulated))
        print("No scans in measured: ", len(measured))
        return False

    #Check no mixing among dty in y-scan peaks of measured data
    tol = 0.0001
    for i,dty in enumerate(measured):
        for j,peak in enumerate(dty):
            if abs(peak[3]-dty[0][3])>tol:
                print("---------------------------------------")
                print("MIXED y-scans in MEASURED peaks set!")
                print("at y-scan index ", i," first peak was")
                print("showing dty ",dty[0][3]," while")
                print("peak ",j," has dty of ", peak[3])
                return False

    #Check no mixing among dty in y-scan peaks of simulated data
    for i,dty in enumerate(simulated):
        for j,peak in enumerate(dty):
            if abs(peak[3]-dty[0][3])>tol:
                print("---------------------------------------")
                print("MIXED y-scans in SIMULATED peaks set!")
                print("at y-scan index ", i," first peak was")
                print("showing dty ",dty[0][3]," while")
                print("peak ",j," has dty of ", peak[3])
                return False

    #Check correct mapping of dty index and dty value of measured data
    for i,dty in enumerate(measured):
        for j,peak in enumerate(dty):
            index = np.round( ( peak[3]-ymin )/( 1000.*beam_width ) ).astype(int)
            if index!=i:
                print("---------------------------------------")
                print("Incorrect MAPPING of dty value to index in MEASURED")
                print("found index, ", i," at what should be index", index )
                print("corresponding dty value is: ", peak[3])
                print("used ymin: ", ymin ," microns")
                print("used beam width: ", 1000.*beam_width ," microns")
                return False

    #Check correct mapping of dty index and dty value of simulated data
    for i,dty in enumerate(simulated):
        for j,peak in enumerate(dty):
            index = np.round( ( peak[3]-ymin )/( 1000.*beam_width ) ).astype(int)
            if index!=i:
                print("---------------------------------------")
                print("Incorrect MAPPING of dty value to index in SIMULATED")
                print("found index, ", i," at what should be index", index )
                print("corresponding dty value is: ", peak[3])
                print("used ymin: ", ymin ," microns")
                print("used beam width: ", 1000.*beam_width ," microns")
                return False

    if lost_peaks!=None:
        if len(lost_peaks)!=no_yscans:
            print("---------------------------------------")
            print("Incorrect number of y-scans in LOST_PEAKS")
            print("Excpected ",no_yscans," y-scans got:")
            print("No scans in lost_peaks: ", len(lost_peaks))
            return False

        #Check no mixing among dty in y-scan peaks of lost peaks
        for i,dty in enumerate(measured):
            for j,peak in enumerate(dty):
                if abs(peak[3]-dty[0][3])>tol:
                    print("---------------------------------------")
                    print("MIXED y-scans in LOST_PEAKS peaks set!")
                    print("at y-scan index ", i," first peak was")
                    print("showing dty ",dty[0][3]," while")
                    print("peak ",j," has dty of ", peak[3])
                    return False

        #Check correct mapping of dty index and dty value of lost peaks
        for i,dty in enumerate(lost_peaks):
            for j,peak in enumerate(dty):
                index = np.round( ( peak[3]-ymin )/( 1000.*beam_width ) ).astype(int)
                if index!=i:
                    print("---------------------------------------")
                    print("Incorrect MAPPING of dty value to index in LOST_PEAKS")
                    print("found index, ", i," at what should be index", index )
                    print("corresponding dty value is: ", peak[3])
                    print("used ymin: ", ymin ," microns")
                    print("used beam width: ", 1000.*beam_width ," microns")
                    return False

        # Check that no illegal peaks got in to the measured peak set
        for i,dty in enumerate(lost_peaks):
            for peak in dty:
                for m in measured[i]:
                    if np.linalg.norm(np.asarray(m)-np.asarray(peak))<0.001:
                        print("---------------------------------------")
                        print("ILLEGAL peaks in MEASURED")
                        print("found hkl: ",m[4],m[5],m[6])
                        print("in measured at dty: ", m[3])
                        print("Which is also in lost peak index")
                        print("Index used: ", i)
                        return False

        # Check that no illegal peaks got in to the simulated peak set
        for i,dty in enumerate(lost_peaks):
            for peak in dty:
                for s in simulated[i]:
                    if np.linalg.norm(np.asarray(s)-np.asarray(peak))<0.001:
                        print("---------------------------------------")
                        print("ILLEGAL peaks in SIMULATED")
                        print("found hkl: ",s[4],s[5],s[6])
                        print("in simulated at dty: ", s[3])
                        print("Which is also in lost peak")
                        print("Index used: ", i)
                        return False

        # Check that the number of peaks match
        for i,(dtym,dtys) in enumerate(zip(measured,simulated)):
            if len(dtym)!=len(dtys):
                print("---------------------------------------")
                print("The number of peks in MEASURED and SIMULATED")
                print("do not match at index,", i)
                print("len(dtym) ", len(dtym))
                print("len(dtys)", len(dtys))
                print("Measured: ")
                print(dtym)
                print("")
                print("Simulated: ")
                print(dtys)
                print("")
                print("Lost peaks: ")
                print(lost_peaks[i])
                return False


        for i,(dtym,dtys) in enumerate(zip(measured,simulated)):
            for p1,p2 in zip(dtym,dtys):
                if p1[4]!=p2[4] or p1[5]!=p2[5] or p1[6]!=p2[6]:
                    print("---------------------------------------")
                    print("hkls do not match!")
                    print("Measured peak: ",p1)
                    print("Simulated peak: ",p2)
                    print("lost_peaks", lost_peaks[i])
                    return False
                if abs(p1[0]-p2[0])+abs(p1[1]-p2[1])+abs(p1[2]-p2[2])>100:
                    print("---------------------------------------")
                    print("Peak error is large")
                    print("Measured peak: ",p1)
                    print("Simulated peak: ",p2)
                    print("lost_peaks", lost_peaks[i])
                    print("Measured peaks: ", measured[i])
                    print("Simulated peaks: ", simulated[i])
                    return False
                # print(p1,p2)

        # Check that the data is somewhat reasonable
        percentage = 50.
        no_banned_peaks = 0
        no_measured_peaks_after_banning = 0
        no_simulated_peaks_after_banning = 0
        for i,dty in enumerate(lost_peaks):
            no_banned_peaks+=len(dty)
        for i,dty in enumerate(measured):
            no_measured_peaks_after_banning+=len(dty)
        for i,dty in enumerate(simulated):
            no_simulated_peaks_after_banning+=len(dty)

        if tot_no_peaks!=None:
            if (1-(no_measured_peaks_after_banning/float(tot_no_peaks)))>(percentage/100.):
                print("---------------------------------------")
                print("More than ",percentage,"% of the peaks in")
                print("MEASURED where banned. This is not strictly wrong")
                print("but return Falses serious doubt.")
                print("no_banned_peaks: ", no_banned_peaks)
                print("no_measured_peaks_after_banning: ", no_measured_peaks_after_banning)
                print("no_simulated_peaks_after_banning: ", no_simulated_peaks_after_banning)
                return False
    return True
