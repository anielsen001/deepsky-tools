#!/usr/bin/env python3
"""
convert a fits file with a Bayer matrix to a super pixel by 
summing the 4 bayer pixels to single value. The output files 
have super_ prepended to the file name. 
"""
import sys, os

import numpy as np
from scipy.signal import convolve2d
from astropy.io import fits

def debayer_file( filein ):

    fpath, fname = os.path.split( filein )
    if len( fpath ) == 0:
        fpath = '.'
    fname_out = 'super_' + fname
    fileout = os.path.sep.join( [ fpath, fname_out ] )

    # read the input data
    bayer_hdu = fits.open( filein )
    bayer_data = bayer_hdu[0].data
    bayer_header = bayer_hdu[0].header

    # create a convolution filter, convolve and downsample
    bayer_mat = np.ones( [2,2], dtype = bayer_data.dtype )
    super_data = convolve2d( bayer_data, bayer_mat, mode = 'valid' )[0::2,0::2]

    # write the output file
    super_hdu = fits.PrimaryHDU(super_data)
    super_hdu.writeto( fileout, overwrite = True )
    
if __name__=='__main__':

    # input is a list of files to convert
    bayer_files_in = sys.argv[1:]

    # output files will be left in the same place
    # with super_ prepended to the file name

    for f in bayer_files_in:
        debayer_file( f )
