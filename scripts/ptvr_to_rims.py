import os
import sys 
import glob
import time
import shutil
import tarfile
import cProfile
import datetime
import configparser
from concurrent import futures
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import Process, Queue, cpu_count
import pystache
import numpy as np

import django

# setup django env
sys.path.append('../libs')
sys.path.append('../../rims-ptvr')


from loguru import logger
from raw_image import RawImage
from log_parser import DualMagLog
from roi_tools import ROI, ptvr_to_rims_filename
from lrauv_data import LRAUVData

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rims-ptvr.settings')
django.setup()

from rois.file_name_formats import FileNameFmt, ChitonFileNameFmt
from rois.models import Image, ProcSettings, LabelSet, TagSet, Tag, Label

camera_name_map = {}
camera_name_map['low_mag_cam_rois'] = {}
camera_name_map['high_mag_cam_rois'] = {}
camera_name_map['low_mag_cam_rois']['V1']= 'PTVR01LM'
camera_name_map['high_mag_cam_rois']['V1']= 'PTVR01HM'
camera_name_map['low_mag_cam_rois']['V2']= 'PTVR02LM'
camera_name_map['high_mag_cam_rois']['V2']= 'PTVR02HM'
camera_name_map['low_mag_cam_rois']['V3']= 'PTVR03LM'
camera_name_map['high_mag_cam_rois']['V3']= 'PTVR03HM'

roi_paths = ['low_mag_cam_rois', 'high_mag_cam_rois']
video_paths = []

output_path = ''

def delete_from_disk(roi_item):

    try:
        if os.path.exists(os.path.join(roi_item['output_path'],roi_item['image_id']+".jpeg")):
            os.remove(os.path.join(roi_item['output_path'],roi_item['image_id']+".jpeg"))
        if os.path.exists(os.path.join(roi_item['output_path'],roi_item['image_id']+".png")):
            os.remove(os.path.join(roi_item['output_path'],roi_item['image_id']+".png"))
        if os.path.exists(os.path.join(roi_item['output_path'],roi_item['image_id']+"_binary.png")):
            os.remove(os.path.join(roi_item['output_path'],roi_item['image_id']+"_binary.png"))
        if os.path.exists(os.path.join(roi_item['output_path'],roi_item['image_id']+"_rawcolor.jpeg")):
            os.remove(os.path.join(roi_item['output_path'],roi_item['image_id']+"_rawcolor.jpeg"))
        if os.path.exists(os.path.join(roi_item['output_path'],roi_item['image_id']+"_rawcolor.png")):
            os.remove(os.path.join(roi_item['output_path'],roi_item['image_id']+"_rawcolor.png"))
    except IOError as e:
        logger.error(e)

def bb_intersection_over_union(roiA, roiB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(roiA.left, roiB.left)
	yA = max(roiA.top, roiB.top)
	xB = min(roiA.left+roiA.width, roiB.left+roiB.width)
	yB = min(roiA.top+roiA.height, roiB.top+roiB.height)
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = roiA.width * roiA.height
	boxBArea = roiB.width * roiB.height
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = float(interArea) / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def threaded_video_proc(raw_image):
    raw_image.export_as_tiff(flat_field=False)
    raw_image.export_8bit_jpegs(thumbnails=True, flat_field=False, gamma=0.5)
    return raw_image
    
def threaded_roi_proc(roi):
    roi.process(save_to_disk=True)
    return roi

def process_bundle_list(bundle_queue,output_queue):

    while True:
        try:
            output_queue.put(threaded_subdir_proc(bundle_queue.get()))
        except:
            time.sleep(0.04*np.random.rand())

def threaded_subdir_proc(subdir_pack):

    extracted_path = None

    proc_time = time.time()

    subdir = subdir_pack['subdir']
    platform = subdir_pack['platform']
    deployment = subdir_pack['deployment']
    camera = subdir_pack['camera']
    proc_settings = subdir_pack['proc_settings']
    timestamp_delta = subdir_pack['timestamp_delta']

    if subdir[-3:] == 'tar':
        extracted_path = subdir + ".unpacked"
        if not os.path.exists(extracted_path):
            try:
                os.makedirs(extracted_path)
            except:
                logger.warning("Path exists, won't create a new one")
        with tarfile.open(subdir) as archive:
            for member in archive.getmembers():
                if member.isreg():  # skip if the TarInfo is not files
                    member.name = os.path.basename(member.name) # remove the path by reset it
                    archive.extract(member,extracted_path) # extract

        rois = sorted(glob.glob(os.path.join(extracted_path, '*.tif')))
    else:
        rois = sorted(glob.glob(os.path.join(subdir, '*.tif')))

    raw_rois = []
    for roi in rois:
        roi_filename = os.path.basename(roi)
        roi_filepath = os.path.dirname(roi)

        # Convert to RIMS file name conventions
        rims_name = ptvr_to_rims_filename(roi_filename, platform, deployment, camera, timestamp_delta=timestamp_delta)
        
        # Check for valid image
        # skip images that have zero file size
        if os.path.getsize(roi) <= 0:
            logger.warning(roi + " file size is <= 0.")
            return
        
        # Check if image is already in db
        #print "Importing " + image_name
        im = Image.objects.filter(image_id = rims_name)
        if not im.exists():
            im = Image(image_id=rims_name)
        else:
            im = im[0]
            #print rims_name + " already exists in db, will reprocess..."
            logger.debug(rims_name + " already exists in db, skipping...")
            continue     
        
        # Import into RIMS
        if im.import_image(roi_filepath, roi_filename, proc_settings):
            raw_rois.append(im)
            im.save()
            # Add Clipped Image tag is the image may be clipped
            if (im.is_clipped):
                ci = Tag.objects.filter(name='Clipped Image')[0]
                im.tags.add(ci)
                im.save()

            # Save the image to the DB
            

        

    # remove the extracted dir
    if extracted_path is not None and os.path.exists(extracted_path):
        shutil.rmtree(extracted_path)

    logger.info('Processed : ' + subdir + ' @ ' + str(len(raw_rois)/(time.time() - proc_time + 1)) + ' roi/s')

    return raw_rois

if __name__=="__main__":

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    with logger.catch():

        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()

        frame_increment = [1,1]

        if len(sys.argv) != 5 and len(sys.argv) != 7:
            logger.exception('Usage: python ptvr_to_rims.py [data_dir] [camera_version] [platform] [deployment]')
            logger.exception('Example: python python ptvr_to_rims.py 2023-10-27-12-07-27.309869056 V2 LRAH 7')
            exit(1)

        # Check to make sure paths are okay
        dl = os.listdir(sys.argv[1])
        for roi_path in roi_paths:
            if not roi_path in dl:
                logger.warning("Could not find: " + roi_path)
        for video_path in video_paths:
            if not video_path in dl:
                logger.warning("Could not find: " + video_path)
        
        
        log_file = glob.glob(os.path.join(sys.argv[1], '*.log'))
        if len(log_file) != 1:
            logger.error("Missing or multiple log files")
            exit(1)
        else:
            log_file = log_file[0]

        try:

            # create output directory and populate
            output_dir = os.path.join(sys.argv[1],'processed-'+os.path.basename(sys.argv[1]))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                pass
                shutil.rmtree(output_dir)
                time.sleep(2)
                os.makedirs(output_dir)

        except IOError as e:
            logger.error(e)
            exit(1)
    
        # # Process and export log file
        # Do not parse logs for RIMS import
        #dml = DualMagLog(log_file)
        #dml.parse_lines()
        #dml.export(os.path.join(output_dir, os.path.basename(log_file)[:-4] + '.csv'))

        # Process ROIs
        total_rois = []
        index = 0
        for roi_path in roi_paths:

            if not os.path.exists(os.path.join(sys.argv[1], roi_path)):
                continue

            all_rois = []

            # create output path is needed
            output_path = os.path.join(output_dir, roi_path)
            
            logger.info(output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # loop over tars, untar and then process subdirs
            subdirs = sorted(glob.glob(os.path.join(sys.argv[1], roi_path, '*')))
            
            subdir_packs = []
            #bundle_queue = Queue()
            for subdir in subdirs:
                subdir_pack = {}
                subdir_pack['subdir'] = subdir
                subdir_pack['output_path'] = output_path
                subdir_pack['platform'] = sys.argv[3]
                subdir_pack['deployment'] = sys.argv[4]
                subdir_pack['camera'] = camera_name_map[roi_path][sys.argv[2]]
                subdir_pack['proc_settings'] = '/home/rimsadmin/software/rims-ptvr/rois/default_proc_settings.json'
                if len(sys.argv) == 7:
                    subdir_pack['timestamp_delta'] = datetime.timedelta(days=int(sys.argv[5]), seconds=int(sys.argv[6]))
                else:
                    subdir_pack['timestamp_delta'] = None
                #bundle_queue.put(subdir_pack)
                subdir_packs.append(subdir_pack)

            roi_per_sec_timer = time.time()

            n_threads = int(cpu_count()/2) - 3
            if n_threads < 1:
                n_threads = 1

            # Single threaded import
            #for sd in subdir_packs:
            #    threaded_subdir_proc(sd)

            # Multithreaded import
            with Pool(processes=n_threads) as p:
                all_subdirs = p.map(threaded_subdir_proc, subdir_packs)

            logger.info('ROIs per second: ' + str(len(all_rois)/(time.time()-roi_per_sec_timer + 1)))

            index += 1

        pr.disable()
        pr.dump_stats("profile_result.txt")
        
