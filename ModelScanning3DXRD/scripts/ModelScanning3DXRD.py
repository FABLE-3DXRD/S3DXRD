#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
from modelscanning3DXRD import gomodelscanning3DXRD

import logging
logging.basicConfig(level=logging.INFO,format='\n%(levelname)s: %(message)s')

def main(print_input,filename,killfile,profile,debug,parallel):
    gomodelscanning3DXRD.run(print_input,filename,killfile,profile,debug,parallel)


if __name__=="__main__":
    options = None
    try:
        from optparse import OptionParser
        parser = OptionParser()
        options  = gomodelscanning3DXRD.get_options(parser)
        print(options)
        main(options.print_input,options.filename,options.killfile,options.profile,options.debug,options.parallel)
    except:
        if options != None:
            parser.print_help()
        raise 
