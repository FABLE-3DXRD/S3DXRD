
from __future__ import print_function, division
import glob, os
import sys
from numpy import arange, ones, concatenate
from ImageD11.columnfile import columnfile, colfile_to_hdf
from ImageD11.parameters import read_par_file

help = """Merges flt files from a series of rotation scans

Args: parfile y-motor outfile.hdf [flts]

It gets the y-motor from the corresponding spt file

Adds a column in the output called dty with y motor position
"""


def get_dty( flt, motor='dty' ):
    with open( flt[:-3]+"spt",  "r") as sptfile:
        vals = None
        names = None
        for line in sptfile.readlines():
            if line.find( "motor_pos" ) > 0:
                vals = line.split()
                if names is not None:
                    return float(vals[ names.index( motor ) ])
            if line.find( "motor_mne" ) > 0:
                names = line.split()
                if vals is not None:
                    return float(vals[ names.index( motor ) ])
    print( "Missed finding dty for",flt)
    return 0
                    


def merge_flts( p, flts, motor ):
    ars = []
    for f in flts:
        try:
            c = columnfile(f)
        except:
            print(f,"Empty file")
            continue
        # num = int(f.split("_")[5])
        dty = get_dty( f, motor=motor )
        print(f, dty, c.nrows)
        try:
            c.addcolumn( ones(c.nrows)*dty , "dty")
        except:
            print (dty, c.nrows, type(dty), type( c.nrows))
            print (ones( c.nrows) * dty)
            raise
        c.updateGeometry( p )
        ars.append( c.bigarray )
    c.bigarray = concatenate( ars, axis=1 )
    c.nrows = c.bigarray.shape[1]
    c.set_attributes()
    return c


if __name__=="__main__":
    try:
        p   = read_par_file( sys.argv[1] )
        motor = sys.argv[2]
        out = sys.argv[3]

        if os.path.exists( out ):
            print(help)
            print("File exists already, try again")
        
        sys.exit()
    except:
        print(help)
        sys.exit()
        
    c = merge_flts( p, sorted(sys.argv[4:]) )
    colfile_to_hdf( c, out )
    
