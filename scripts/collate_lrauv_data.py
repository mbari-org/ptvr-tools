import os
import sys
import h5py
import glob
import pytz
import pandas as pd
import numpy as np
from loguru import logger
import datetime

import django

# setup django env
sys.path.append('../libs')
sys.path.append('../../rims-ptvr')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rims-ptvr.settings')
django.setup()

from rois.file_name_formats import FileNameFmt, ChitonFileNameFmt
from rois.models import Image, ProcSettings, LabelSet, TagSet, Tag, Label

from lrauv_data import LRAUVData

def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.datetime.utcfromtimestamp(timestamp)


        
if __name__=="__main__":

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    with logger.catch():
        
        if len(sys.argv) != 2:
            logger.error("Usage: python collate_lrauv_data.py [path-to-lrauv-data]")
            exit()
            
        # Load all of the LRAUV data
        lr = LRAUVData(sys.argv[1])
        lr.load_all_logs(sys.argv[1])
        
        query_count = 0
        image_count = 0
        
        for index in range(0,len(lr.full_df['timestamp'])-1):
            d1 = to_datetime(lr.full_df['timestamp'].values[index]).replace(tzinfo=pytz.UTC)
            d2 = to_datetime(lr.full_df['timestamp'].values[index+1]).replace(tzinfo=pytz.UTC)

            # Find images with matching timestamp range
            imgs = Image.objects.filter(timestamp__range=[d1,d2])
            image_count += imgs.count()
            # Loop over images and update fields from LRAUV Data
            for img in imgs:
                img.depth = np.mean(lr.full_df['depth'].values[index:index+2])
                img.latitude = np.mean(lr.full_df['latitude'].values[index:index+2])
                img.longitude = np.mean(lr.full_df['longitude'].values[index:index+2])
                img.temperature = np.mean(lr.full_df['sea_water_temperature'].values[index:index+2])
                img.salinity = np.mean(lr.full_df['sea_water_salinity'].values[index:index+2])
                img.chlorophyll = np.mean(lr.full_df['mass_concentration_of_chlorophyll_in_sea_water'].values[index:index+2])
                img.save()
                
            query_count += 1
            
            if query_count % 100 == 0:
                logger.info('collated ' + str(image_count) + ' images in ' + str(query_count) + ' queries.')