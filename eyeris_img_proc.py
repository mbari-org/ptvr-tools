import sys
import glob
import os
import cv2
import skimage
import argparse
import time
import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy import ndimage
from skimage import filters, morphology, measure, color


def sbd_proc(img_path): 
    
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10;
    params.maxThreshold = 200;

    params.filterByArea = True
    params.minArea = 50

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = False
    params.minConvexity = 0.87

    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    
    # load
    img = cv2.imread(img_path)

    # make image gray scale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    return keypoints, img


def canny_proc(img_path, can_min=150, can_max=225, se_size=5):
    """
    process image with canny edge detector
    :param img_path: absolute path to image [str]
    :param can_min: min for hysteris threshold [int]
    :param can_max: max for hysteris threshold [int]
    :param se_size: size of square structuring element [int]
    :return mask: binary mask
    """

    # load
    img = cv2.imread(img_path)

    # make image gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # trim black bar in image
    img = img[:,0:-60]

    # run canny
    cn = cv2.Canny(img, can_min, can_max)

    # clean up the mask
    mask = morphology.closing(cn, morphology.square(se_size))

    return mask, img


def get_regions(masked, img, conn=2, props=None):
    """
    get region props from binary mask
    :param masked: binary mask [hh, ww]
    :param img: gray scale image [hh, ww]
    :param conn: type of connectivity to consider [int]
    :param props: region props to extract [list]
    :return prop_df: data frame of properties of all detected regions
    """

    # get the props
    if props is None:
        props = ['bbox', 'centroid', 'equivalent_diameter']

    # generate label image
    label_img = morphology.label(masked, connectivity=conn, background=0)
    
    # measure the region properties
    prop_df = measure.regionprops_table(label_img, img, properties=props)
    
    # join bouding boxes that would overlap 
    if len(prop_df['area']) > 0:
        for ind in range(0,len(prop_df['area'])):
            cv2.rectangle(masked * 0,(prop_df['bbox-1'][ind], prop_df['bbox-0'][ind]) , (prop_df['bbox-3'][ind], prop_df['bbox-2'][ind]),255,-1)
    
    # generate label image again
    label_img = morphology.label(masked, connectivity=conn, background=0)
    
    # measure the region properties
    prop_df = measure.regionprops_table(label_img, img, properties=props)       
    


    return prop_df


if __name__ == '__main__':

    # instantiate parser
    parser = argparse.ArgumentParser(description='process a directory of images from AyeRIS')
    parser.add_argument('img_dir', help='absolute path to images directory')
    parser.add_argument('--out_dir', default=None, help='path to output [default=new subdir in parent of img_dir] ')
    parser.add_argument('--img_format', default='png', help='image file format')
    parser.add_argument('--edge_detector', default='canny', help='edge detector to use [canny,]')
    parser.add_argument('--output_format', default='csv', help='output file format [csv, json]')
    parser.add_argument('--save_boxed_image', default=True, help='Save an output image with boxes on detections')

    # parse
    args = parser.parse_args()
    img_dir = args.img_dir
    img_format = args.img_format
    edge_detect = args.edge_detector
    out_format = args.output_format
    save_boxed_image = args.save_boxed_image

    # create the output directory if needed
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_path = os.path.split(img_dir)[0]
        out_name = os.path.split(img_dir)[1]
        out_dir = os.path.join(out_path, f"{out_name}_processed_{int(time.time())}")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    # get list of images to process
    imgs = glob.glob(os.path.join(img_dir, f"*.{img_format}"))

    if edge_detect == 'canny':
        for im in progressbar.progressbar(imgs):
            
            mm, gray = canny_proc(im)  # the edge detector
            df = get_regions(mm, gray, props=['area','bbox', 'centroid', 'equivalent_diameter'])  # get the regions as dict
            df = pd.DataFrame(df)  # make output df

            # save the dataframe
            outpath = os.path.join(out_dir, f"{os.path.basename(im).split('.')[0]}.{out_format}")

            if out_format == 'csv':
                df.to_csv(outpath)
            elif out_format == 'json':
                df.to_json(outpath)
                
            # save boxed image if requested
            if save_boxed_image:
                color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                for ind in range(0,len(df['area'])):
                    if df['area'][ind] > 1:
                        cv2.rectangle(color_img,(df['bbox-1'][ind], df['bbox-0'][ind]) , (df['bbox-3'][ind], df['bbox-2'][ind]),(0,255,0),3)
                
                imgpath = os.path.join(out_dir, f"{os.path.basename(im).split('.')[0]}.jpg")
                cv2.imwrite(imgpath, color_img)
    elif edge_detect == 'sbd':
        for im in progressbar.progressbar(imgs):
            keypoints, img = sbd_proc(im)
            # color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Keypoints", im_with_keypoints)
            cv2.waitKey(0)

    else:
        # space here to add other options for edge detectors
        pass