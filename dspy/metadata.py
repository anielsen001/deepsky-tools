"""
metadata.py

wrapper to hold and manage metadata associated with workflow

"""

from astropy.io.fits import Header

class Metadata( Header ):
    """
    based on astropy.fits.Header class with some customizations
    """
    
    def __init__(  self, *args, **kwargs ):
        super().__init__( *args, **kwargs )
