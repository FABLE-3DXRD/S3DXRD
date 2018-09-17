

from __future__ import print_function, division

import glob, os

HOME = "/data/visitor/ma4200/id11/ImageD11_peaksearches"
THRESHOLDS = [ int(100*pow(2,i)) for i in range(9) ]


splinefiles = {
    "10241024" : "/data/id11/3dxrd/inhouse/Frelon21/frelon21_mar16_2x2.spline",
    "20482048" : "/data/id11/3dxrd/inhouse/Frelon21/frelon21_mar16.spline"
    }            

def read_info_file( infofile ):
    """ get the information from the info file for peaksearch"""
    with open( infofile, "r" ) as f:
        info = {}
        for line in f.readlines():
            key, val = line.split("=")
            info[key] = val.lstrip().rstrip()
    command = info["Comment"]
    items = command.split()
    # 0               1       2 3   4  5    6
    # ftomointerlaced silicon 0 180 90 0.08 1
    if items[0] == "ftomointerlaced":
        info[ "interlaced" ] = True
        if items[6] == "1":
            info[ "iflip" ] = True
        # interlaced so items[2] == range
        info["step"] = float(items[3])/int(items[4])/2 # interlaced
        info["start"] = float( items[2] ) + info["step"]/2.
        info["last"] = int(items[4])-1
        info["first"] = 0
    if items[0] == "ftomoscan":
        # not interlaced so items[3] == range
        info["step"] = float(items[3])/int(items[4]) # interlaced
        info["start"] = float( items[2] ) + info["step"]/2.
        info["last"] = int(items[4])-1
        info["first"] = 0
    return info

def do_bg( info, N=5 ):
    if info.has_key( "interlaced" ):
        info[ "bgstem" ] = info["Prefix"]+"0_"
    else:
        info[ "bgstem" ] = info["Prefix"]
    cmd = []
    for i in range(N):
        cmd += [ "bgmaker.py",
                "--namestem={Directory}/{bgstem}",
                " -F .edf "
                "--first=%d"%(i+int(info["first"])),
                "--last=%d"%(int(info["last"]) - N - 1),
                "--outfile={outfolder}/{Prefix}_b%04d.edf"%(i),
                "--step=%d"%(N),
            "\n" ]
    cmd += ["median.py -d",
            "-i {outfolder}/{Prefix}_b0000.edf",
            "-f 0 -l 5",
            "-o {outfolder}/{Prefix}_median",
            "\n"    ]
    for i in range(N):
        cmd.append( "rm {outfolder}/{Prefix}_b%04d.edf\n"%(i) )
    # skip gzip as it gives annoying messages
    # cmd.append( "gzip -q -1 {outfolder}/{Prefix}_median.edf\n" )
    # writes to 
    return " ".join( cmd ).format( **info )
        

def do_peaksearch( info ):
    cmd = [do_bg(info),
           "peaksearch.py",
             " -F .edf ", 
            "--namestem={Directory}/{Prefix}",
            "--outfile={outfolder}/{Prefix}.spt",
            "--darkfile={outfolder}/{Prefix}_median.edf"
            ]
    for option in ("flood","first","last","ndigits"):
        if info.has_key( option ):
            cmd.append( "--%s={%s}"%(option, option ) )
    for option in ( "interlaced", "iflip" ):
        if info.has_key( option ):
            cmd.append( "--"+option )
    # side effects  :start needs step and override
    if info.has_key( "start" ):
        cmd.append( "--step={step}" )
        cmd.append( "--start={start}" )
        cmd.append( "--OmegaOverRide" )
    # spline or not
    if info.has_key( "splinefile" ):
        cmd += [ "--splinefile={splinefile}", "--perfect_images=N" ]
    else:
        cmd.append("--perfect_images=Y")
    for t in info["thresholds"]:
        cmd.append("--threshold=%d"%(t))
    cmdline = " ".join( cmd ).format( **info )
    return cmdline
    

shfiles = []
for line in open("scans_to_process.txt","r").readlines():
    line = line.rstrip()
    if len(line) == 0 or line[0] == "#":
        continue
    folders = sorted(glob.glob( line ))
    for folder in folders:
        assert os.path.isdir( folder )
        ftomofolder, scanfolder = os.path.split( folder )
        dataroot, ftfolder = os.path.split( ftomofolder )
        outfolder = os.path.join( HOME, scanfolder )
        print( "# ",folder )
        print( "#\t",outfolder )
        if not os.path.exists( outfolder ) :
            os.makedirs( outfolder )
        info = read_info_file( os.path.join( folder, scanfolder+".info" ) )
        info[ "outfolder" ] = outfolder
        info[ "thresholds" ] = THRESHOLDS
        # print(info)
        info[ "splinefile" ] = splinefiles[ info["Dim_1" ]+info["Dim_2" ] ]
        cmdline =  do_peaksearch( info )
        with open( os.path.join( outfolder, "dopks.sh" ), "w" ) as shfile:
            shfile.write("# folder %s\n"%(folder))
            shfile.write(cmdline)
            shfiles.append(  os.path.join( outfolder ) )
            

todo = []
for folder in shfiles:
    if os.path.exists( os.path.join(folder, "dopks.log")):
        print("Done",folder)
    else:
        print("Todo:",folder)
        #continue
        todo.append( "sh " + os.path.join( folder, "dopks.sh") + " >> " +
                         os.path.join( folder, "dopks.log") )

import multiprocessing
p=multiprocessing.Pool(multiprocessing.cpu_count())
for r in p.imap_unordered( os.system, todo ):
    pass

