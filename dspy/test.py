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

bias_stacker = BiasStacker( bias_list[0:2] )
bias_stacker.write('test/bias_test.fits', overwrite = True)
bias_stacker.write_meta_json('test/bias_test.json')
bias_stacker.del_raw_frames()

dark_stacker = DarkStacker( dark_list[0:2],
                            bias_stack = bias_stacker )
dark_stacker.write('test/dark_test.fits', overwrite = True)
dark_stacker.write_meta_json('test/dark_test.json')
dark_stacker.del_raw_frames()

flat_stacker = FlatStacker( flat_list[0:2] ,
                            bias_stack = bias_stacker,
                            dark_stack = dark_stacker )
flat_stacker.write('test/flat_test.fits', overwrite = True)
flat_stacker.write_meta_json('test/flat_test.json')
flat_stacker.del_raw_frames()

light_stacker = LightStacker( light_list[0:2] ,
                              bias_stack = bias_stacker,
                              dark_stack = dark_stacker,
                              flat_stack = flat_stacker,
                              method = np.mean,
                              reg_method = None )

frm = light_stacker.write_preprocess('test/pp_test_',dtype='int16',overwrite=True) 
