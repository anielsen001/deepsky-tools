#!/usr/bin/env python3
"""
The Android app DeepSkyCamera creates files with names formatted like:
Lights_0001_YEARMMDD_HHMMSS.dng
The second field is the file count in the series of recorded images
which I prefer to have at the end fo the file name. This script renames
the files to my preferrred format.
"""

import sys
import os

# rename
def rename( origname ):
    basename = os.path.basename( origname )
    rootname, ext = os.path.splitext( basename ) 
    dirname = os.path.basename( basename )
    stem,count,date,time = rootname.split('_')

    newname = '_'.join([ stem, date, time, count ] ) + ext
    
    return newname

if __name__=='__main__':

    fnames = sys.argv[1:]
    for fn in fnames:
        os.rename( fn, rename(fn) )
        
