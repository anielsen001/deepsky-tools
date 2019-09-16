import numpy as np
import glob

from matplotlib import pylab as plt
from scipy import signal
plt.ion()

from stacker import *

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

bias_stacker = BiasStacker( bias_list )
bias_stacker.write('test/bias_test.fits')

dark_stacker = DarkStacker( dark_list,
                            bias_stack = bias_stacker )
dark_stacker.write('test/dark_test.fits')

flat_stacker = FlatStacker( flat_list ,
                            bias_stack = bias_stacker,
                            dark_stack = dark_stacker )
flat_stacker.write('test/flat_test.fits')

light_stacker = LightStacker( light_list[0:10] ,
                              bias_stack = bias_stacker,
                              dark_stack = dark_stacker,
                              flat_stack = flat_stacker,
                              method = np.mean,
                              reg_method = None )

frm = light_stacker.write_preprocess('test/pp_test_') 
