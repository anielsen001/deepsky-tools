import numpy as np
import rawpy
import glob
from astropy.io import fits
from matplotlib import pylab as plt
from scipy import signal
plt.ion()
from tqdm import tqdm

import logging
logger = logging.getLogger( 'stacker' )
logger.setLevel( logging.DEBUG )
ch = logging.StreamHandler()
logger.addHandler( ch )

class StackerError( Exception ):
    pass

class Stacker( object ):
    """
    a generic stacker object
    """
    _method = None
    _frame_list = None
    _raw_frames = None
    _stack = None
    
    def __init__( self,
                  frms,
                  method = np.median ):
        
        self.method = method

        if type( frms ) is np.ndarray:
            self._raw_frames = frms
        elif type( frms ) is list:
            self._frame_list = frms
            self._raw_frames = self.read_list( frms )
        else:
            raise StackerError( 'Type not understood' )
            
    def read_list( self, frm_list ):
        """
        read a list of files specifying frames into 
        memory
        """
        logger.debug('Stacker:read_list')
        
        raw0 = rawpy.imread( frm_list[0] )
        sz = raw0.raw_image.shape

        stk = np.zeros( [sz[0], sz[1], len(frm_list) ],
                        dtype = raw0.raw_image.dtype )

        # read each frame and put it into the stack
        for i,b in enumerate( tqdm(frm_list) ):
            raw = rawpy.imread( b )
            stk[:,:,i] = raw.raw_image

        return stk

    def stack( self, method = None ):

        logger.debug('Stacker:stack')
        
        if method is None:
            method = self.method

        # preprocess the frames for stacking
        pp_stk = self.preprocess()
            
        stk = method( pp_stk, axis = 2 )
        self._stack = stk

    def get_stack( self ):
        
        logger.debug('Stacker:get_stack')
        
        if self._stack is None:
            self.stack()

        return self._stack
        
    def write( self, filename ):
        """
        write the stack to a fits file 
        """
        logger.debug('Stacker:write')
        
        hdu = fits.PrimaryHDU( self.get_stack() )
        hdul = fits.HDUList( [hdu] )
        hdul.writeto( filename )

    def preprocess( self ):
        """
        Define this preprocess method to apply to the raw
        frames before stacking.

        This generic method just returns the original raw frames. 
        Override it if you need to do something else.
        """
        
        logger.debug('Stacker:preprocess')
        
        return self._raw_frames

    def postprocess( self ):
        """
        define a post-process method to apply after stacking to the 
        stacked frame, this default just returns the stack.
        """
        logger.debug('Stacker:postprocess')
        
        return self._stack
        
    def debayer( self, bayer_frame ):
        """
        remove the bayer mosaic from the raw FPA data. This method sums
        the 2x2 area to a single pixel value. 
        """
        logger.debug('Stacker:debayer')
        
        # convolve and downsample to sum into 2x2 blocks
        kernel = np.ones([2,2])
        fscds = signal.convolve2d( bayer_frame, kernel, 'same' )[::2, ::2]
        # fscds is the debayed frame

        # an alternative for multiple frames to debayer is like
        # frmc = signal.convolve( frm, kernel[:,:,np.newaxis])[::2, ::2, : ]
        return fscds
    
class BiasStacker( Stacker ):
    """
    Stack the bias frames to produce a clean bias frame
    """

    def __init__( self,
                  bias_frms,
                  method = np.median ):
        """ 
        stk_frms is a list of frame files to stack or an np.ndarray
                 object that is N x M x nframes

        the np.median method creates float64 as output type
        """
        super().__init__( bias_frms, method = method )
        
class DarkStacker( Stacker ):
    """
    Stack the dark frames to produce a clean dark frame
    """
    # bias_stack is an object of the BiasStacker class
    _bias_stack = None
    
    def __init__( self,
                  dark_frames,
                  bias_stack = None,
                  method = np.median ):
        
        self._bias_stack = bias_stack
        super().__init__( dark_frames, method = method )
        
    def preprocess( self ):
        """
        remove the bias frame from each dark frame if the bias
        frame is configured
        """
        logger.debug('DarkStacker:preprocess')
        
        if self._bias_stack is not None:
            bstk_frm = self._bias_stack.get_stack()
            pp_frames = self._raw_frames - bstk_frm[:,:,np.newaxis]
        else:
            pp_frames = self._raw_frames

        return pp_frames

class FlatStacker( Stacker ):
    """
    stack the flat frames to produce a flat frame

    the stacked flat frame needs to be set to unity gain, but
    we need to account for the Bayer mosaic before we do that.
    """
    # these are objects of BiasStacker, DarkStacker and FlatStacker
    _bias_stack = None
    _dark_stack = None
    _flat = None

    def __init__( self,
                  flat_frames,
                  bias_stack = None,
                  dark_stack = None,
                  method = np.median ):

        self._bias_stack = bias_stack
        self._dark_stack = dark_stack

        super().__init__( flat_frames, method = method )

    def preprocess( self ):
        """
        remove the bias and dark frame stack from each frame
        """

        logger.debug('FlatStacker:preprocess')

        pp_frames = self._raw_frames
        
        # remove bias if input as option
        if self._bias_stack is not None:
            pp_frames = pp_frames - self._bias_stack.get_stack()[:,:,np.newaxis]
        
        # remove dark if input as option
        if self._dark_stack is not None:
            pp_frames = pp_frames - self._dark_stack.get_stack()[:,:,np.newaxis]
           
        # return - this still has the bayer mosaic
        return pp_frames

    def make_flat_frame( self ):
        """
        generate the flat frame to use. Does not return the flat frame.
        """
        
        logger.debug('FlatStacker:make_flat_frame')
        
        # debayer the image - now it's downsample 2x2 in each direction
        dimg = self.debayer( self.get_stack() )

        # normalize the dmin to unity gain
        dimg /= np.sum( dimg.flatten() )

        self._flat = dimg

    def get_flat( self ):
        """
        return the flat frame as a numpy array
        """

        logger.debug('FlatStacker:get_flat')
        
        if self._flat is None:
            self.make_flat_frame()

        return self._flat
        
    
class LightStacker( Stacker ):
    """
    stack the light frame - the light frames need a registration algorithm
    """
    # these are objects of BiasStacker, DarkStacker and FlatStacker
    _bias_stack = None
    _dark_stack = None
    _flat_stack = None

    method = None # stacking method
    reg_method = None # registration method to register the frames

    def __init__( self,
                  light_frames,
                  bias_stack = None,
                  dark_stack = None,
                  flat_stack = None,
                  method = np.sum,
                  reg_method = None ):

        self._bias_stack = bias_stack
        self._dark_stack = dark_stack
        self._flat_stack = flat_stack

        super().__init__( light_frames, method = method )

    def preprocess( self ):
        """
        remove dark frame stack from each frame
        """
        logger.debug('LightStacker:preprocess')

        pp_frames = self._raw_frames
        
        # remove bias if input as option
        if self._bias_stack is not None:
            pp_frames = pp_frames - self._bias_stack.get_stack()[:,:,np.newaxis]
        
        # remove dark if input as option
        if self._dark_stack is not None:
            pp_frames = pp_frames - self._dark_stack.get_stack()[:,:,np.newaxis]
           
        if self._flat_stack is not None:
            # in order to flatten the frames, we must debayer them
            kernel = np.ones( [2,2] )
            db_pp = signal.convolve( pp_frames, kernel[:,:,np.newaxis] ,'same')[::2,::2,:]

            # flatten each
            flt = self._flat_stack.get_flat()
            dpf = db_pp / flt[:,:,np.newaxis]
            
        return dpf
        
            
if __name__=='__main__' and False:
    
    bias_list = glob.glob( '/data/stars/biases-2019-06-02/iso800/Bias*.dng' )
    dark_list = glob.glob( '/data/stars/darks-2019-05-29/iso800/Darks*.dng' )
    flat_list = glob.glob( '/data/stars/2019-08-29/f1/Lights*.dng' )
    light_list = glob.glob( '/data/stars/2019-08-29/s1/Lights*.dng' )

    bias_stacker = BiasStacker( bias_list )
    
    
    dark_stacker = DarkStacker( dark_list,
                                bias_stack = bias_stacker )
    
    flat_stacker = FlatStacker( flat_list[0:10] ,
                                bias_stack = bias_stacker,
                                dark_stack = dark_stacker )

    light_stacker = LightStacker( [ light_list[0] ],
                                  bias_stack = bias_stacker,
                                  dark_stack = dark_stacker,
                                  flat_stack = flat_stacker,
                                  method = np.sum,
                                  reg_method = None )
    
    frm = light_stacker.preprocess() 
