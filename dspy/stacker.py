
import numpy as np
import rawpy
from astropy.io import fits
from scipy import signal
from tqdm import tqdm, trange
import json
import exifread
import socket
import datetime

from numba import jit,prange

import logging
logger = logging.getLogger( 'stacker' )
logger.setLevel( logging.DEBUG )
ch = logging.StreamHandler()
logger.addHandler( ch )

from .metadata import Metadata

from functools import wraps
# this works as a decorator for a class method and will log the
# name of the class and the method being called.
def log_debug( f ):
    @wraps( f )
    def wrapper( *args, **kwargs ):
        logger.debug( args[0].__class__.__name__ +':'+f.__name__ )
        return f( *args, **kwargs )
    return wrapper

# use as a decorator to warn that a method or function is deprecated
def warn_deprecated( f ):
    @wraps( f )
    def wrapper( *args, **kwargs ):
        logger.warn( f.__name__ + ' is deprecated and may be removed.' )
        return f( *args, **kwargs )
    return wrapper
    

class StackerError( Exception ):
    pass

class Stacker( object ):
    """
    a generic stacker object
    """
    # method used to stack frames default is np.median
    _method = np.median

    # list of raw files containing each frame to put into the stack
    _frame_list = None

    # 3-D array of each input frame to the stack
    _raw_frames = None

    # 2-D array of output from stacking process
    _stack = None

    # file saved to disk containing a 2-D stacked set fo frames
    _stack_file = None

    # metadata object to hold associated metadata
    metadata = Metadata()

    # need per-frame metadata also - this is a list of Metadata objects
    _frm_meta = list()
    
    def __init__( self,
                  frms,
                  method = np.median ):

        # set the method 
        self.method = method

        if isinstance( frms, np.ndarray ): 
            # input is a 3D np.ndarray of frames to stack
            # Mpixel x Npixel x Nframes
            self._raw_frames = frms
        elif isinstance( frms, list ): 
            # input is a list of strings referring to file names
            self._frame_list = frms
            # don't read the raw frames until needed
            #self._raw_frames = self.load_raw_frames( frms )

            if not frms:
                # input list is empty
                raise StackerError( 'Input list of file names is empty' )
            
        elif isinstance( frms, str ): 
            # input is a file containing already stacked data
            self.stack_file = frms
        else:
            raise StackerError( 'Type not understood' )

        # populate metadata
        # we are going to be lazy about populating metadata until we start doing things
        # put the class name in the metadata
        self.metadata['stacker']  = ( self.__class__.__name__, 'stacker class name' ) 


    # @log_debug
    def load_raw_frames( self, frm_list ):
        """
        read a list of files specifying frames into 
        memory
        """
        if not frm_list:
            # frm_list is empty, raise an exception
            raise StackerError( 'Frames to stack must not be empty' )

        # read the first frame in the list and get the image size
        # use this image size to allocate memory for holding all
        # the memory
        raw0 = rawpy.imread( frm_list[0] )
        sz = raw0.raw_image.shape

        # read the EXIF metadata from the first file and use this to populate
        # the meta data for the stack
        # md0 is metadata of file 0
        # definitions may be found here:
        # https://www.exif.org/Exif2-2.PDF
        # https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
        with open( frm_list[0], 'rb' ) as f:
            md0 = exifread.process_file( f )

        # get the camera make and model
        self.metadata['camera'] = ' '.join( [ md0['Image Make'].values, md0['Image Model'].values ] )

        # get the date/time of the image capture - string type
        self.metadata['collect datetime'] = ( md0['Image DateTimeOriginal'].values,
                                              'collection date and time' )

        # get the camera imaging parameters
        # F number - ratio type
        self.metadata['f_number'] = ( md0['Image FNumber'].values[0].num /\
                                      md0['Image FNumber'].values[0].den,
                                      'F number of camera collection' )
        # exposure time in seconds, convert from ratio to floating point
        self.metadata['exposure_time'] = ( md0['Image ExposureTime'].values[0].num / \
                                           md0['Image ExposureTime'].values[0].den,
                                           'Exposure time in seconds' )
        # ISO - integer type
        self.metadata['ISO'] = ( md0['Image ISOSpeedRatings'].values[0],
                                 'ISO setting of camera' )
        # focal length - ratio type
        self.metadata['focal_length'] = ( md0['Image FocalLength'].values[0].num /\
                                          md0['Image FocalLength'].values[0].den,
                                          'Focal length of camera' )
        
        # allocate array/memory to hold the raw data from each frame
        self._raw_frames = np.zeros( [sz[0], sz[1], len(frm_list) ],
                        dtype = raw0.raw_image.dtype )

        # initialize the list of per-frame metadata
        self._frm_meta = [ Metadata() for _i in frm_list ]

        # read each frame and put it into the stack

        # we may encounter a bad frame, track it with this list
        bad_frm_idx = list() 
        t_enum_list = trange(  len( frm_list ), desc = 'Reading Frames:', leave = True ) 
        #for i,b in enumerate( tqdm(frm_list,desc = 'Reading frames: ') ):
        for i in t_enum_list:
            # b is the file name from the list
            # i is the index in the list
            b = frm_list[i]
            t_enum_list.set_description( 'Reading frame (%s)'%b )
            
            # populate overall metadata with frame name
            kw = 'frm' + str(i).rjust(5,'0')
            self.metadata[ kw ] = b
            
            # read the exif meta data from the frame
            try:
                with open( b, 'rb' ) as f:
                    mdf = exifread.process_file ( f )

                # assign metadata for this frame
                # image collect time
                self._frm_meta[i]['img_time'] = ( mdf['Image DateTimeOriginal'].values,
                                                  'image date and time' )
                # exposure time in seconds, convert from ratio to floating point
                self._frm_meta[i]['exp_time'] = ( mdf['Image ExposureTime'].values[0].num / \
                                                  mdf['Image ExposureTime'].values[0].den,
                                                  'Exposure time in seconds' )
                self._frm_meta[i]['origname'] = (b, 'Original file name ')
                
            except:
                # an error occured getting meta data, treat as bad file
                bad_frm_idx.append( i )

                # skip reading binary image data if a bad frame
                continue

            # read the image data
            try: 
                with rawpy.imread( b ) as raw:
                    self._raw_frames[:,:,i] = raw.raw_image.copy()
            except:
                # an error occured reading the file, track this as a bad file
                bad_frm_idx.append( i )

        if bad_frm_idx:
            # if any bad frames were encountered, remove them from the stack

            # set diff to get the good frames
            good_frm_idx = np.setdiff1d( np.arange( len(frm_list ) ), bad_frm_idx )

            # keep only the good frames in the stack
            self._raw_frames = self._raw_frames[ :, :, good_frm_idx ]

            # clean up the per-frame meta data as well
            # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
            for bad_idx in sorted( bad_frm_idx, reverse = True ):
                del self._frm_meta[ bad_idx ]
                
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

    # raw_frames = property( get_raw_frames, load_raw_frames, del_raw_frames )

    @property
    def stack_file( self ):
        """ return the stack file name """
        return self._stack_file

    @stack_file.setter
    def stack_file( self, filename ):
        """ set the stack file name """
        if type( filename ) is not str:
            raise StackerError( 'stack file name must be a string' )
        self._stack_file = filename
    
    @log_debug
    @jit( parallel = True )
    def make_stack( self, method = None ):

        if method is None:
            method = self.method

        # write out debug information
        logger.debug('Stacker:stack with: ' + self.method.__name__ )

        # when we stack update the metadata to hold the method used for
        # creating the stack
        self.metadata['method'] = ( self.method.__name__, 'method used for stacking' )

        # add the hostname where stacking occurred
        self.metadata['hostname'] = ( socket.gethostname(), 'hostname for stacking' )

        # add the date/time of stacking
        self.metadata['datetime'] = ( str( datetime.datetime.now() ) , 'date and time of stacking' )
        
        # preprocess the frames for stacking
        prep_stk = self.preprocess()

        # stack the frames
        self._stack = method( prep_stk, axis = 2 )

        # postprocess the stack after stacking
        post_stk = self.postprocess()

        # set the stack frame property
        self._stack = post_stk

    # can't decorate a property?
    @property
    def stack( self ):            
        
        if self._stack is None:
            if self.stack_file is None:
                # if no stack file is listed, then generate a stack
                self.make_stack()
            else:
                # read from the stack file
                self.read_stack_from_fits()

        return self._stack

    @stack.setter
    def stack( self, stk ):
        """ set the stack, must be a 2D numpy array """
        self._stack = stk

    @property
    def method( self ):
        return self._method

    @method.setter
    def method( self, val ):
        # set the method 
        self._method = val
    
    @log_debug    
    def write( self, filename, dtype = None, overwrite = False ):
        """
        write the stack to a fits file 
        
        dtype specifies the data type to write out. It should be one of numpy's
        acceptable types. The default is None which will use the type of the 
        array returned from self.stack

        """
        stk = self.stack
        
        self.stack_file = filename

        self.write_frame_to_fits( filename, stk, dtype = dtype, overwrite = overwrite )
        
    # @log_debug
    def write_preprocess( self, filepattern = 'pp_', dtype = None, overwrite = False ):
        """
        write out each frame after applying the preprocessing corrections
        file pattern is a pattern to use when writing the frames 
        
        The default filepattern pp_ follows the siril convention
        """

        # this can fail is the preprocess method needs too much memory
        pp_frms = self.preprocess()

        nframes = self._raw_frames.shape[2]

        for iframe in tqdm( range( nframes ), desc = 'Write pp frames: ' ):
            fname = filepattern + str(iframe).rjust(5,'0') + '.fits'
            
            hdu = fits.PrimaryHDU( np.squeeze( pp_frms[:,:,iframe] ) )

            self.write_frame_to_fits( fname,
                                      np.squeeze( pp_frms[:,:,iframe]),
                                      dtype = dtype,
                                      overwrite = overwrite,
                                      extrameta = self._frm_meta[ iframe ] )

    def write_frame_to_fits( self, filename, frame, dtype = None, overwrite = False, extrameta = None ):
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
                raise StackerError('Cannot cast to requested type, data ouside range')
            
        # else: the are the same type, so we don't have to do anything

        # this cast will potentially round floating point numbers to integers
        if not np.can_cast( frame, dtype, casting = 'safe' ):
            logger.warning('Casting from %s to %s will reduce precision'%(str(frame.dtype), str(dtype)))

        hdu = fits.PrimaryHDU( frame.astype( dtype ) )

        #
        # add some informaton to the header
        #
        hdr = hdu.header

        hdr.extend( self.metadata )

        if extrameta:
            hdr.extend( extrameta )
                
        hdul = fits.HDUList( [hdu] )
        hdul.writeto( filename, overwrite = overwrite )

    def read_stack_from_fits( self ):
        """
        read a 2-D stack from a fits file into memory
        requires that self._stack_file be set
        """

        hdul = fits.open( self.stack_file )
        
        self.stack = hdul[0].data
        
        # this will be a string name stored in the file, not the function
        # that should be passed when creating
        self.method = hdul[0].header['method']

        # put this method into the metadata attribute as well
        self.metadata['method'] = ( self.method, 'method used for stacking' )
        

    @warn_deprecated
    def get_meta_json( self ):
        """
        write the metadata of this object to a dictionary, at the base class Stacker
        level. This dictionary will be written to a json file for saving information.
        This writes out the basic information that should be needed to 
        recreate the stack results. Stacker objects that require other data to stack
        """

        # put the class name in the metadata
        metadata = { 'stacker' : self.__class__.__name__ } 

        # put the hostname into the metadata
        metadata['hostname'] = socket.gethostname()

        # put the date into the metadata
        metadata['datetime'] = str( datetime.datetime.today() )
        
        try:
            metadata[ 'method' ] = self._method.__name__
        except AttributeError:
            # there is no __name__ attribute
            # convert whatever exists to a string
            metadata[ 'method' ] = str( self._method )

        if self._frame_list is not None:
            metadata[ 'files' ] = self._frame_list

        return metadata

    @warn_deprecated
    def write_meta_json( self, jsonname ):
        """
        write a json file contaning the meta data requried to recreate the stack. This calls
        the get_meta_json method to convert the meta data to a dictionary. By breaking up the 
        function calls this way, subclasses can override the get_meta_json method and also 
        call the super class get_meta_json if desired. 
        """
        jsonstr = json.dumps( self.get_meta_json() )
        
        with open( jsonname, 'w' ) as f:
            f.write( jsonstr )
        
    # @log_debug    
    def preprocess( self ):
        """
        Define this preprocess method to apply to the raw
        frames before stacking.

        This generic method just returns the original raw frames. 
        Override it if you need to do something else.
        """
        
        return self.get_raw_frames()

    # @log_debug
    def postprocess( self ):
        """
        define a post-process method to apply after stacking to the 
        stacked frame, this default just returns the stack.
        """
                
        return self._stack

    # @log_debug
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

    The dark stack does not need to have the bias frame removed. Track where 
    the bias is removed to make sure that it's not removed twice.
    """
    # bias_stack is an object of the BiasStacker class
    _bias_stack = None
    
    def __init__( self,
                  dark_frames,
                  bias_stack = None,
                  method = np.median ):
        
        self._bias_stack = bias_stack
        super().__init__( dark_frames, method = method )

    # @log_debug
    def preprocess( self ):
        """
        remove the bias frame from each dark frame if the bias
        frame is configured
        """
        
        if self._bias_stack is not None:
            bstk_frm = self._bias_stack.stack
            pp_frames = self.get_raw_frames() - bstk_frm[:,:,np.newaxis]
        else:
            pp_frames = self.get_raw_frames()

        return pp_frames

    # @log_debug
    def get_meta_json( self ):
        """
        the Dark stacker meta data should include some information about the 
        bias stack if it was used
        """

        # get the metadata from the parent class
        metadata = super().get_meta_json()

        # add meta data based on this subclass instance
        if self._bias_stack is not None:
            # a bias stack was usd, so add it to the metadata dictionary
            metadata[ 'bias stack' ] = self._bias_stack.get_meta_json()

        return metadata

class FlatStacker( Stacker ):
    """
    stack the flat frames to produce a flat frame

    The stacked flat frame needs to be set to a gain, but
    we need to account for the Bayer mosaic before we do that.

    If you use a dark stack here, it must be generated using dark frames
    with the same exposure time as the flat frames. Typically the flat
    frames have much shorter exposure times than the light frames, so the 
    integrated dark current will be different in both cases.

    Examples of exposure times from cell phone camera Motorola Moto X4
    flat frames : 0.05 sec
    bias frames : 1.06e-5 sec
    light frames (maximum possible) : 0.3 sec
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

    # @log_debug
    def preprocess( self ):
        """
        Remove the bias and dark frame stack from each frame

        """

        pp_frames = self.get_raw_frames()
        
        # remove bias if input as option
        if self._bias_stack is not None:
            pp_frames = pp_frames - self._bias_stack.stack[:,:,np.newaxis]
        
        # remove dark if input as option
        if self._dark_stack is not None:
            pp_frames = pp_frames - self._dark_stack.stack[:,:,np.newaxis]
           
        # return - this still has the bayer mosaic
        return pp_frames

    @property
    def flat( self ):
        if self._flat is None:
            self.make_flat_frame()

        return self._flat

    # @log_debug
    def make_flat_frame( self ):
        """
        generate the flat frame to use. Does not return the flat frame.
        This frame is debayered.
        """
        
        # debayer the image - now it's downsample 2x2 in each direction
        dimg = self.debayer( self.stack )

        # normalize the dmin to unity gain
        ### this step scales all the flat values to very small numbers
        #dimg /= np.sum( dimg.flatten() )

        # the resulting flat image should be an image that can be divided
        # into raw frames
        self._flat_mean = np.mean( dimg[:] )
        
        self._flat = dimg / self._flat_mean

        
    # @log_debug
    def get_flat( self ):
        """
        return the flat frame as a numpy array
        """
        return self.flat

    # @log_debug
    def get_meta_json( self ):
        """
        the Dark stacker meta data should include some information about the 
        bias stack if it was used
        """

        # get the metadata from the parent class
        metadata = super().get_meta_json()

        # add meta data based on this subclass instance
        if self._bias_stack is not None:
            # a bias stack was usd, so add it to the metadata dictionary
            metadata[ 'bias stack' ] = self._bias_stack.get_meta_json()

        if self._dark_stack is not None:
            # a dark stack was used, so add it to the metadata dictionary
            metadata[ 'dark stack' ] = self._dark_stack.get_meta_json()

        return metadata

        
    
class LightStacker( Stacker ):
    """
    stack the light frame - the light frames need a registration algorithm
    """
    # these are objects of BiasStacker, DarkStacker and FlatStacker
    _bias_stack = None
    _dark_stack = None
    _flat_stack = None

    _reg_method = None # registration method to register the frames

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

    # @log_debug
    def pre_proc( self ):
        """
        remove bias frame stack from each frame
        remove dark frame stack from each frame
        debayer each frame
        """

        pp_frames = self.get_raw_frames()
        
        # remove bias if input as option
        if self._bias_stack is not None:
            pp_frames = pp_frames - self._bias_stack.stack[:,:,np.newaxis]
        
        # remove dark if input as option
        if self._dark_stack is not None:
            pp_frames = pp_frames - self._dark_stack.stack[:,:,np.newaxis]
           
        if self._flat_stack is not None:
            # in order to flatten the frames, we must debayer them
            # kernel = np.ones( [2,2] )
            # db_pp = signal.convolve( pp_frames, kernel[:,:,np.newaxis] ,'same')[::2,::2,:]
            # this will run out of memory if there are too many frames
            db_pp = self.debayer( pp_frames )
            
            # flatten each
            flt = self._flat_stack.get_flat()
            pp_frames = db_pp / flt[:,:,np.newaxis]
            
        return pp_frames

    def pre_proc_single( self ):
        """
        preprocess each frame separately in a loop, to conserve memory

        assumes, dark frame and flat frame correction only
        """
        # get teh frames to preprocess
        pp_frames = self.get_raw_frames()

        L,M,N = pp_frames.shape

        # get the flat frame
        flt = self._flat_stack.get_flat()

        pp_out = np.zeros( [ int(L/2), int(M/2), N ] ) 

        for ifrm in trange( N, desc = 'Pre-processing: ', leave = True ):
            
            # remove bias
            ppf = np.squeeze(pp_frames[:,:,ifrm]) - self._dark_stack.stack

            # debayer
            ppfd = self.debayer( ppf )

            # flatten
            ppfdf = ppfd / flt

            pp_out[ :, :, ifrm ] = ppfdf

        return pp_out
                    
    # this allows for easy swapping of pre-process method
    preprocess = pre_proc_single

    @log_debug
    def post_proc( self ):
        """
        remove bias from final stack
        remove dark from final stack
        debayer final stack
        flatten final stack
        """
        # remove bias if input as option
        if self._bias_stack is not None:
            self._stack = self._stack - self._bias_stack.stack
        
        # remove dark if input as option
        if self._dark_stack is not None:
            self._stack = self._stack - self._dark_stack.stack

        if self._flat_stack is not None:
            # in order to flatten the frames, we must debayer them
            # kernel = np.ones( [2,2] )
            # db_pp = signal.convolve( pp_frames, kernel[:,:,np.newaxis] ,'same')[::2,::2,:]
            self._stack = self.debayer( self._stack )
            
            # flatten each
            flt = self._flat_stack.get_flat()
            self._stack = self._stack / flt

        return self._stack

    @log_debug
    def get_meta_json( self ):
        """
        the Dark stacker meta data should include some information about the 
        bias stack if it was used
        """

        # get the metadata from the parent class
        metadata = super().get_meta_json()

        # add meta data based on this subclass instance
        if self._bias_stack is not None:
            # a bias stack was usd, so add it to the metadata dictionary
            metadata[ 'bias stack' ] = self._bias_stack.get_meta_json()

        if self._dark_stack is not None:
            # a dark stack was used, so add it to the metadata dictionary
            metadata[ 'dark stack' ] = self._dark_stack.get_meta_json()

        if self._flat_stack is not None:
            # a dark stack was used, so add it to the metadata dictionary
            metadata[ 'flat stack' ] = self._flat_stack.get_meta_json()
        
        return metadata


