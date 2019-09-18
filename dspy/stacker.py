import glob # not needed here, but convenient for now
from matplotlib import pylab as plt
plt.ion()

import numpy as np
import rawpy
from astropy.io import fits
from scipy import signal
from tqdm import tqdm

import logging
logger = logging.getLogger( 'stacker' )
logger.setLevel( logging.DEBUG )
ch = logging.StreamHandler()
logger.addHandler( ch )

from functools import wraps
# this works as a decorator for a class method and will log the
# name of the class and the method being called.
def log_debug( f ):
    @wraps( f )
    def wrapper( *args, **kwargs ):
        logger.debug( args[0].__class__.__name__ +':'+f.__name__ )
        return f( *args, **kwargs )
    return wrapper

class StackerError( Exception ):
    pass

class Stacker( object ):
    """
    a generic stacker object
    """
    # method used to stack frames default is np.median
    _method = None

    # list of raw files containing each frame to put into the stack
    _frame_list = None

    # 3-D array of each input frame to the stack
    _raw_frames = None

    # 2-D array of output from stacking process
    _stack = None     
    
    def __init__( self,
                  frms,
                  method = np.median ):
        
        self._method = method

        if type( frms ) is np.ndarray:
            # input is a 3D np.ndarray of frames to stack
            # Mpixel x Npixel x Nframes
            self._raw_frames = frms
        elif type( frms ) is list:
            # input is a list of strings referring to file names
            self._frame_list = frms
            # don't read the raw frames until needed
            #self._raw_frames = self.load_raw_frames( frms )
        else:
            raise StackerError( 'Type not understood' )

    @log_debug
    def load_raw_frames( self, frm_list ):
        """
        read a list of files specifying frames into 
        memory
        """
        
        raw0 = rawpy.imread( frm_list[0] )
        sz = raw0.raw_image.shape

        self._raw_frames = np.zeros( [sz[0], sz[1], len(frm_list) ],
                        dtype = raw0.raw_image.dtype )

        # read each frame and put it into the stack
        for i,b in enumerate( tqdm(frm_list) ):
            with rawpy.imread( b ) as raw:
                self._raw_frames[:,:,i] = raw.raw_image.copy()

    @log_debug            
    def get_raw_frames( self ):
        """
        return the raw frames. If they are not in memory, load them
        """

        if self._raw_frames is None:
            self.load_raw_frames( self._frame_list )

        return self._raw_frames

    @log_debug
    def del_raw_frames( self ):
        """
        remove the raw frames from memory
        """
        self._raw_frames = None

    @log_debug
    def stack( self, method = None ):

        if method is None:
            method = self._method

        logger.debug('Stacker:stack with: ' + self._method.__name__ )
            
        # preprocess the frames for stacking
        pp_stk = self.preprocess()
            
        self._stack = method( pp_stk, axis = 2 )

    @log_debug
    def get_stack( self ):
        
        if self._stack is None:
            self.stack()

        return self._stack
    
    @log_debug    
    def write( self, filename, dtype = None, overwrite = False ):
        """
        write the stack to a fits file 
        
        dtype specifies the data type to write out. It should be one of numpy's
        acceptable types. The default is None which will use the type of the 
        array returned from self.get_stack()

        """
        stk = self.get_stack()

        self.write_frame_to_fits( filename, stk, dtype = dtype, overwrite = overwrite )
        
    @log_debug
    def write_preprocess( self, filepattern = 'pp_', dtype = None, overwrite = False ):
        """
        write out each frame after applying the preprocessing corrections
        file pattern is a pattern to use when writing the frames 
        
        The default filepattern pp_ follows the siril convention
        """

        pp_frms = self.preprocess()

        nframes = self._raw_frames.shape[2]

        for iframe in tqdm( range( nframes ) ):
            fname = filepattern + str(iframe).rjust(5,'0') + '.fits'
            
            hdu = fits.PrimaryHDU( np.squeeze( pp_frms[:,:,iframe] ) )

            self.write_frame_to_fits( fname,
                                      np.squeeze( pp_frms[:,:,iframe]),
                                      dtype = dtype,
                                      overwrite = overwrite )

    def write_frame_to_fits( self, filename, frame, dtype = None, overwrite = False ):
        """
        write out a 2D frame of data to a fits file, casting to a particular
        numpy dtype beforehand if necessary
        """
        # siril appears to have some restrictions on the data types
        # you can write out and use for partial reading in the registration
        # algorithms.
        #
        # see src/core/siril.h for the type defintions:
        #  /* bitpix can take the following values:
        # * BYTE_IMG     (8-bit byte pixels, 0 - 255)
        # * SHORT_IMG    (16 bit signed integer pixels)  
        # * USHORT_IMG   (16 bit unsigned integer pixels)        (used by Siril, quite off-standard)
        # * LONG_IMG     (32-bit integer pixels)
        # * FLOAT_IMG    (32-bit floating point pixels)
        # * DOUBLE_IMG   (64-bit floating point pixels)
        # * http://heasarc.nasa.gov/docs/software/fitsio/quick/node9.html
        # */
        #
        # and src/io/image_format_fits.c for registration error message:
        # if (fit->bitpix != SHORT_IMG && fit->bitpix != USHORT_IMG
	# && fit->bitpix != BYTE_IMG) {
	# siril_log_message(
	# _("Only Siril FITS images can be used with partial image reading.\n"));
	# return -1;
	# }
        #
        # options to handle this with astropy are hdu.scale or
        # cast using numpy's astype before writing
        #
        # the acceptable siril types are all integers so some rounding
        # will occur when going from float64. we'll do some checking on the
        # dynamic range of the data to see how it will fit.
        
        if dtype is None:
            # if not specified, use the type of the frame
            dtype = frame.dtype

        elif dtype != frame.dtype:
            # check if the specified type matches the stack
            # requested to cast the data, so determine if the data
            # will fit in the requested type
            if ( frame.max() >= np.iinfo( dtype ).max or\
                 frame.min() <= np.iinfo( dtype ).min ):
                # data is outside range supported by datatype
                raise StackError('Cannot cast to requested type, data ouside range')
            
        # else: the are the same type, so we don't have to do anything

        # this cast will potentially round floating point numbers to integers
        if not np.can_cast( frame, dtype, casting = 'safe' ):
            logger.warning('Casting from %s to %s will reduce precision'%(str(frame.dtype), str(dtype)))

        hdu = fits.PrimaryHDU( frame.astype( dtype ) )
            
        # add some informaton to the header
        hdr = hdu.header
        # name of stacking method
        hdr['method'] = self._method.__name__ 
        # 
        
        hdul = fits.HDUList( [hdu] )
        hdul.writeto( filename, overwrite = True )
        
    @log_debug    
    def preprocess( self ):
        """
        Define this preprocess method to apply to the raw
        frames before stacking.

        This generic method just returns the original raw frames. 
        Override it if you need to do something else.
        """
        
        return self.get_raw_frames()

    @log_debug
    def postprocess( self ):
        """
        define a post-process method to apply after stacking to the 
        stacked frame, this default just returns the stack.
        """
                
        return self.get_stack()

    @log_debug
    def debayer( self, bayer_frame ):
        """
        remove the bayer mosaic from the raw FPA data. This method sums
        the 2x2 area to a single pixel value. 
        """
        
        # convolve and downsample to sum into 2x2 blocks
        kernel = np.ones([2,2])

        # how many dimensions does the bayer_frame have?
        ndims = len( bayer_frame.shape )

        if ndims == 2:
            fscds = signal.convolve2d( bayer_frame, kernel, 'valid' )[::2, ::2]
            # fscds is the debayed frame
        elif ndims == 3:
            # an alternative for multiple frames to debayer is like
            fscds = signal.convolve( bayer_frame,
                                     kernel[:,:,np.newaxis],
                                     mode = 'valid' )[::2, ::2, : ]
            
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

    @log_debug
    def preprocess( self ):
        """
        remove the bias frame from each dark frame if the bias
        frame is configured
        """
        
        if self._bias_stack is not None:
            bstk_frm = self._bias_stack.get_stack()
            pp_frames = self.get_raw_frames() - bstk_frm[:,:,np.newaxis]
        else:
            pp_frames = self.get_raw_frames()

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
    _flat_mean = None

    def __init__( self,
                  flat_frames,
                  bias_stack = None,
                  dark_stack = None,
                  method = np.median ):

        self._bias_stack = bias_stack
        self._dark_stack = dark_stack

        super().__init__( flat_frames, method = method )

    @log_debug
    def preprocess( self ):
        """
        remove the bias and dark frame stack from each frame
        """

        pp_frames = self.get_raw_frames()
        
        # remove bias if input as option
        if self._bias_stack is not None:
            pp_frames = pp_frames - self._bias_stack.get_stack()[:,:,np.newaxis]
        
        # remove dark if input as option
        if self._dark_stack is not None:
            pp_frames = pp_frames - self._dark_stack.get_stack()[:,:,np.newaxis]
           
        # return - this still has the bayer mosaic
        return pp_frames

    @log_debug
    def make_flat_frame( self ):
        """
        generate the flat frame to use. Does not return the flat frame.
        """
        
        # debayer the image - now it's downsample 2x2 in each direction
        dimg = self.debayer( self.get_stack() )

        # normalize the dmin to unity gain
        ### this step scales all the flat values to very small numbers
        #dimg /= np.sum( dimg.flatten() )

        # the resulting flat image should be an image that can be divided
        # into raw frames
        self._flat_mean = np.mean( dimg[:] )
        
        self._flat = dimg / self._flat_mean

    @log_debug
    def get_flat( self ):
        """
        return the flat frame as a numpy array
        """

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

    @log_debug
    def preprocess( self ):
        """
        remove bias frame stack from each frame
        remove dark frame stack from each frame
        debayer each frame
        """

        pp_frames = self._raw_frames
        
        # remove bias if input as option
        if self._bias_stack is not None:
            pp_frames = pp_frames - self._bias_stack.get_stack()[:,:,np.newaxis]
        
        # remove dark if input as option
        if self._dark_stack is not None:
            pp_frames = pp_frames - self._dark_stack.get_stack()[:,:,np.newaxis]
           
        if self._flat_stack is not None:
            # in order to flatten the frames, we must debayer them
            # kernel = np.ones( [2,2] )
            # db_pp = signal.convolve( pp_frames, kernel[:,:,np.newaxis] ,'same')[::2,::2,:]
            db_pp = self.debayer( pp_frames )
            
            # flatten each
            flt = self._flat_stack.get_flat()
            pp_frames = db_pp / flt[:,:,np.newaxis]
            
        return pp_frames
        
            
if __name__=='__main__' and False:

    if False: 
        bias_list = glob.glob( '/data/stars/biases-2019-06-02/iso800/Bias*.dng' )
        dark_list = glob.glob( '/data/stars/darks-2019-05-29/iso800/Darks*.dng' )
        flat_list = glob.glob( '/data/stars/2019-08-29/f1/Lights*.dng' )
        light_list = glob.glob( '/data/stars/2019-08-29/s1/Lights*.dng' )

    if True:
        bias_list = glob.glob( '/home/apn/data/stars/biases-2019-06-02/iso800/Bias*.dng' )
        dark_list = glob.glob( '/home/apn/data/stars/darks-2019-05-29/iso800/Darks*.dng' )
        flat_list = glob.glob( '/home/apn/data/stars/flats-2019-06-01/iso800/Flats*.dng' )
        light_list = glob.glob( '/home/apn/data/stars/lights-2019-06-02/ursa_major/Lights*.dng' )

    bias_stacker = BiasStacker( bias_list[0:10] )
    
    dark_stacker = DarkStacker( dark_list[0:10],
                                bias_stack = bias_stacker )
    
    flat_stacker = FlatStacker( flat_list[0:10] ,
                                bias_stack = bias_stacker,
                                dark_stack = dark_stacker )

    light_stacker = LightStacker( light_list[0:10] ,
                                  bias_stack = bias_stacker,
                                  dark_stack = dark_stacker,
                                  flat_stack = flat_stacker,
                                  method = np.sum,
                                  reg_method = None )
    
    frm = light_stacker.preprocess() 
