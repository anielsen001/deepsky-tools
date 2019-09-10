import numpy as np
import rawpy
import glob
from astropy.io import fits
from matplotlib import pylab as plt
plt.ion()

bias_list = glob.glob( '/home/apn/data/stars/biases-2019-06-02/iso800/Bias*.dng' )

dark_list = glob.glob( '/home/apn/data/stars/darks-2019-05-29/iso800/Darks*.dng' )

# allocate memory to hold the raw images
# read one image to get sizes
raw0 = rawpy.imread( bias_list[0] )
sz = raw0.raw_image.shape

stk = np.zeros( [sz[0], sz[1], len(bias_list) ],
                dtype = raw0.raw_image.dtype )

# read each frame and put it into the stack
for i,b in enumerate( bias_list ):
    raw = rawpy.imread( b )
    stk[:,:,i] = raw.raw_image

stkmed = np.median( stk, axis = 2 )

# write stacked file as a fits file
# this appears to convert the numerical types to
# double precision floating point
hdu = fits.PrimaryHDU( stkmed )
hdul = fits.HDUList( [hdu] )
hdul.writeto( 'test.fits' )

# load the stacked fits file
# fits.open retuns a list of hdus in the file
# hdu = header descriptor unit
hdul = fits.open( 'test.fits' ) 

# for my images, there is only a primary hdu, the first in the list
hdui = hdul[0]
img = hdui.data

#

class StackerError( Exception ):
    pass

class Stacker( object ):
    """
    a generic stacker object
    """
    _method = None

    _raw_frms = None

    _stack = None
    
    def __init__( self,
                  frms,
                  method = np.median ):
        
        self.method = method

        if type( frms ) is np.ndarray:
            self._raw_frms = frms
        elif type( frms ) is list:
            self._raw_frms = self.read_list( frms )
        else:
            raise StackerError( 'Type not understood' )
            
    def read_list( self, frm_list ):
        """
        read a list of files specifying frames into 
        memory
        """
        
        raw0 = rawpy.imread( frm_list[0] )
        sz = raw0.raw_image.shape

        stk = np.zeros( [sz[0], sz[1], len(frm_list) ],
                        dtype = raw0.raw_image.dtype )

        # read each frame and put it into the stack
        for i,b in enumerate( frm_list ):
            raw = rawpy.imread( b )
            stk[:,:,i] = raw.raw_image

        return stk

    def stack( self, method = None ):
        if method is None:
            method = self.method
        stk = method( self._raw_frms, axis = 2 )
        self._stack = stk

    def get_stack( self ):
        if _stack is None:
            self.stack()

        return self._stack
        
    def write( self, filename ):
        """
        write the stack to a fits file 
        """
        hdu = fits.PrimaryHDU( self.stack )
        hdul = fits.HDUList( [hdu] )
        hdul.writeto( filename )

    
class BiasStacker( Stacker ):
    """
    Stack the bias frames to produce a clean bias frame
    """
    method = None

    def __init__( self,
                  stk_frms,
                  method = np.median ):
        """ 
        stk_frms is a list of frame files to stack or an np.ndarray
                 object that is N x M x nframes
        """

        self.method = method
        
        
        
