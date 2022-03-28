import sys
import os
import numpy as np
import cv2
import shutil
import glob
import time
import datetime
from loguru import logger

LOG_DIR='/NVMEDATA/publogs'
LOG_FILE='/NVMEDATA/publogs/sa_pub.log'
COURIER_PREFIX='Courier9'

def batch_transfer(img_dir, period=1800, rescale=4, downsamp=2):

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    img_counter = 0
    with open(os.path.join(LOG_DIR, 'pub_counter'), "r") as f:
        img_counter = int(f.readline())

    if img_counter > 1000:
        img_counter = 0

    logger.info("Found " + str(len(imgs)) + " images, transfering largest")

    # Send the largest of the images to shore only so we get one per
    # exec

    loaded_imgs = []

    max_size_ind = 0

    img_size = 0

    for i, img in enumerate(imgs):

        # load image and resample to lower resolution, color depth
        im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if im is not None:
            im = im/rescale
            im = cv2.resize(im, (int(im.shape[1]/downsamp), int(im.shape[0]/downsamp)), 0, 0, cv2.INTER_AREA)

            loaded_imgs.append(im)

            if np.prod(im.shape) > img_size:
                max_size_ind = i
                img_size = np.prod(im.shape)

    for i, img in enumerate(loaded_imgs):

        new_name = imgs[i] + '.jpg'
        cv2.imwrite(new_name, im)

        logger.info("Processing: " + new_name)

        if i == max_size_ind:
            # Copy the image to Courier name
            courier_name = COURIER_PREFIX + "{:03d}".format(img_counter)
            shutil.copy(new_name, os.path.join(img_dir, courier_name))

            # Transfer to the vehicle
            cmd = "rsync "
            cmd += os.path.join(img_dir, courier_name)
            cmd += " galene:/LRAUV/Logs/latest/"
            os.system(cmd)

            logger.info(cmd)

            img_counter += 1

        # move the image to the log
        logger.info("Moving: " + imgs[i])
        shutil.move(imgs[i], os.path.join(LOG_DIR, os.path.basename(imgs[i])))
        shutil.move(new_name, os.path.join(LOG_DIR, os.path.basename(new_name)))

        #time.sleep(period)

    # tar the contents of the log dir
    if len(imgs) > 0:

        with open(os.path.join(LOG_DIR, 'pub_counter'), "w") as f:
            f.writelines([str(img_counter) + '\n'])

        tar_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".tar.gz"
        cmd = "cd " + LOG_DIR + "; " + "tar cvzf " + tar_name + " *.tif* --remove-files"
        os.system(cmd)


if __name__=="__main__":

    #print("test only")
    #exit(0)

    if len(sys.argv) < 5:
        print('Usage: python sa_pub.py [IMG_DIR] [PUB_PERIOD] [RESCALE] [DOWNSAMP]')
        exit(1)

    logger.add(LOG_FILE, rotation="500 MB")

    img_dir = sys.argv[1]
    period = int(sys.argv[2])
    rescale = int(sys.argv[3])
    downsamp = int(sys.argv[4])

    batch_transfer(img_dir, period=period, rescale=rescale, downsamp=downsamp)
