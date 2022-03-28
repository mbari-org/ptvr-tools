import os
import sys
import cv2
import struct
import datetime
import imageio
import numpy as np
from loguru import logger

def read_file_header(file_path, header_length=128, fmt=''):
    
    data = np.fromfile(file_path, dtype=np.int32, count=int(header_length/4))
    if len(data) != header_length/4:
        logger.error("Could not head full length of header")
        return
    
    return data

def read_frame(file_path, frame_number, header_length=128, frame_header_length=40, width=5504, height=3648, bpp=16):

    of = header_length + frame_number*(frame_header_length + bpp/8 * width * height) 

    #logger.info("Frame offset: " + str(of))
    
    header_data = np.fromfile(file_path, dtype=np.int64, count=int(frame_header_length/8), offset=int(of))
    if len(header_data) != frame_header_length/8:
        logger.error("Could not head full length of header")
        return

    #logger.info("Image data offset: " + str(int(of)+int(frame_header_length/8)))

    image_data = np.fromfile(file_path, dtype='<u2', count=width*height, offset=int(of)+int(frame_header_length))

    return header_data, image_data

def export_raw_frames(file_path, export_path):

    header_data = read_file_header(file_path)

    width = header_data[5]
    height = header_data[6]
    nframes = header_data[2]
    frame_header_length = header_data[1]

    # handle case of vc frame with height of 1
    if height == 1:
        width = 5504
        height = 3648

    # create the output dir
    export_subdir_1 = os.path.join(export_path, os.path.basename(file_path)[:-4] + '-01')
    export_subdir_2 = os.path.join(export_path, os.path.basename(file_path)[:-4] + '-02')
    if not os.path.exists(export_subdir_1):
        os.makedirs(export_subdir_1)
    if not os.path.exists(export_subdir_2):
        os.makedirs(export_subdir_2)

    for i in range(0,nframes,1):
        hdata, idata = read_frame(file_path, i, 128, frame_header_length, width, height, 16)
        dt = datetime.datetime.utcfromtimestamp(float(hdata[2])/1000000)
        timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S-%f")  
        img = np.reshape(idata, (height, width))
        #color_img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), 0, 0, cv2.INTER_AREA).astype(float)
        
        if i % 2 == 0:
            img = 4*255*(img - np.min(img))/np.max(img)
            export_path = os.path.join(export_subdir_1, timestamp + "-{:06d}".format(hdata[1]) + '.tif')
        else:
            img = 2*255*(img - np.min(img))/np.max(img)
            export_path = os.path.join(export_subdir_2, timestamp + "-{:06d}".format(hdata[1]) + '.tif')

        img[img > 255] = 255

        #if np.mean(color_img[:,:,1]) < np.mean(color_img[:,:,2]):
        #    color_img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
        imageio.imwrite(export_path, img.astype(np.uint8))
        logger.info('Exported: ' + export_path)


if __name__=="__main__":

    last_time = 0

    if len(sys.argv) < 3:
        logger.error('Usage: python raw_frame_tools.py [raw_file_path] [export_path]')

    export_raw_frames(sys.argv[1], sys.argv[2])

