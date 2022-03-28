import qoi
import numpy as np
import imageio
import sys
import os
import glob

input_path = sys.argv[1]
low_byte_paths = glob.glob(os.path.join(input_path,'*-LBITS.qoi'))
high_byte_paths = glob.glob(os.path.join(input_path,'*-UBITS.qoi'))

output_dir = os.path.join(input_path, 'converted')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i,img in enumerate(low_byte_paths):
    
    lbyte = np.flipud(np.fliplr(qoi.read(low_byte_paths[i])))
    
    hbyte = np.flipud(np.fliplr(qoi.read(high_byte_paths[i])))
    
    lout = np.zeros((lbyte.shape[0]*2, lbyte.shape[1]*2), dtype=np.uint16)
    hout = np.zeros((hbyte.shape[0]*2, hbyte.shape[1]*2), dtype=np.uint16)
    
    # fill in the image with the respective channels
    lout[1::2, 0:-1:2] = lbyte[:,:,0]
    lout[0:-1:2, 1::2] = lbyte[:,:,1]
    lout[1::2, 1::2] = lbyte[:,:,2]
    lout[0:-1:2, 0:-1:2] = lbyte[:,:,3]
    
    hout[1::2, 0:-1:2] = hbyte[:,:,0]
    hout[0:-1:2, 1::2] = hbyte[:,:,1]
    hout[1::2, 1::2] = hbyte[:,:,2]
    hout[0:-1:2, 0:-1:2] = hbyte[:,:,3]

    output_path = os.path.join(output_dir, os.path.basename(img)[:-10] + '.tif')
    
    imageio.imwrite(output_path, np.bitwise_or(lout, 256*hout))
    
    print(output_path)