import numpy as np
import glob

from matplotlib import pylab as plt
from scipy import signal
plt.ion()

from stacker import *


bias_list = glob.glob( '/data/stars/biases-2019-06-02/iso800/Bias*.dng' )
dark_list = glob.glob( '/data/stars/darks-2019-05-29/iso800/Darks*.dng' )
flat_list = glob.glob( '/data/stars/2019-08-29/f1/Lights*.dng' )

bias_stacker = BiasStacker( bias_list )
bias_stacker.write('test/bias_test.fits', overwrite = True)
bias_stacker.write_meta_json('test/bias_test.json')
bias_stacker.del_raw_frames()

dark_stacker = DarkStacker( dark_list,
                            bias_stack = bias_stacker )
dark_stacker.write('test/dark_test.fits', overwrite = True)
dark_stacker.write_meta_json('test/dark_test.json')
dark_stacker.del_raw_frames()

flat_stacker = FlatStacker( flat_list ,
                            bias_stack = bias_stacker,
                            dark_stack = dark_stacker )
flat_stacker.write('test/flat_test.fits', overwrite = True)
flat_stacker.write_meta_json('test/flat_test.json')
flat_stacker.del_raw_frames()

# construct light lists and loop over
light_dirs = [ 's1', 's2', 's3', 's4' ] # directories with light files
#light_list = glob.glob( '/data/stars/2019-08-29/' )
light_head_dir = '/data/stars/2019-08-29/'

for ld in light_dirs:
    light_list = glob.glob( os.sep.join([ light_head_dir, ld, '*.dng' ] ))
    #print(light_list)


    light_stacker = LightStacker( light_list[0:10] ,
                                  bias_stack = bias_stacker,
                                  dark_stack = dark_stacker,
                                  flat_stack = flat_stacker,
                                  method = np.mean,
                                  reg_method = None )
    try:
        
        frm = light_stacker.write_preprocess(os.sep.join( ['test',ld,'pp_test_'] ),
                                         dtype='int16',
                                         overwrite=True) 
    except StackerError:
        continue
