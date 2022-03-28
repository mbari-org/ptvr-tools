import os
import cv2
import sys
import glob
import imageio
import numpy as np
from loguru import logger

def compress_image(filepath, quality=10):

    im = cv2.imread(filepath)
    dim = (int(im.shape[1]/2), int(im.shape[0]/2))
    im2 = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(filepath+'.q'+str(quality)+'.jpg',im2,[int(cv2.IMWRITE_JPEG_QUALITY), quality])

def open_video(data_path, fps=10):
    # open video file
    video_path = os.path.join(data_path, os.path.basename(data_path) + '.mp4')

    video_format = cv2.VideoWriter_fourcc(*'H264')

    out = imageio.get_writer(video_path,
                              format='FFMPEG',
                              mode='I',
                              fps=fps,
                              output_params=['-movflags', '+faststart'],
                              quality=9.5)

    return out

def merge_jpegs(low_img, high_img):

    low_img[low_img == 255] = 0

    output = np.bitwise_or(low_img.astype(np.uint16), 256 * high_img.astype(np.uint16))
    output = (output/4).astype(np.uint8)

    return output


def merge_directory(data_path, cam_name='cam00', make_video=True):

    # get path to jpegs
    cam_path = glob.glob(os.path.join(data_path,cam_name+'*'))

    if len(cam_path) < 1:
        logger.error("No data directory found with the matching camera name")
        return
    
    output_dir = os.path.join(cam_path[0],'data','jpegs_merged')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    low_imgs = sorted(glob.glob(os.path.join(cam_path[0],'data','jpegs','*-L.jpg')))
    high_imgs = sorted(glob.glob(os.path.join(cam_path[0],'data','jpegs','*-H.jpg')))
    
    for i,img in enumerate(low_imgs):
        
        l = cv2.imread(low_imgs[i])
        h = cv2.imread(high_imgs[i])
        
        merged = merge_jpegs(l, h)
        
        output_path = os.path.join(output_dir,os.path.basename(low_imgs[i])[0:-6]+'_merged.jpg')
        imageio.imwrite(output_path, merged)
        
        logger.info('Merged: ' + output_path)
        
        
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        logger.error("Useage: python image_tools.py [path]")
        exit(1)
        
    merge_directory(sys.argv[1], cam_name='cam02')