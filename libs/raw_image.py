# -*- coding: utf-8 -*-
import os
import cv2
import json
import struct
import datetime
import numpy as np
from tifffile import TiffWriter
from psutil import virtual_memory
from loguru import logger

__author__ = "Paul Roberts"
__copyright__ = "Copyright 2020 Guatek"
__credits__ = ["Guatek"]
__license__ = "MIT"
__maintainer__ = "Paul Roberts"
__email__ = "proberts@guatek.com"
__doc__ = '''

Handle loading and exporting raw bin files from pcam-like cameras

@author: __author__
@status: __status__
@license: __license__
'''


class RawImage:
    """Load, process and export raw bin files
    """

    def __init__(self, filepath, output_path=None):

        self.filepath = filepath
        self.output_path = output_path
        self.file_header_format = '<iiQiQQ'
        self.file_header_params = [
            'file_type',
            'frames_per_file',
            'bits_per_pixel',
            'pixel_format',
            'image_width',
            'image_height'
        ]

        self.frame_header_format = '<QQQQ'
        self.frame_header_params = [
            'marker',
            'frame_id',
            'frame_timestamp',
            'system_timestamp',
        ]

        # exports
        self.tiff_exports = []
        self.jpeg_exports = []
        self.thumbnail_exports = []
        self.image_items = []

        self.can_do_ram = False

        self.file_fmt = 0
        self.file_header_length = 36
        self.frame_header_size = 32
        self.file_header = None
        self.frame_headers = []
        self.frame_data = []
        self.field_correction = None
        self.background = None

        # actual image width and height with raw size and binning
        self.img_width = 0
        self.img_height = 0

        # Try to open the file and read
        try:
            self.file_handle = open(self.filepath, "rb")
            self.file_valid = True
            self.file_size = os.path.getsize(self.filepath)
            self.file_valid = self.file_size > 0
        except FileNotFoundError as e:
            logger.error(e)
            self.file_valid = False

        if not self.file_valid:
            logger.error("Zero size file.")
            return

        # read the file header and set image sizes accordingly
        self.read_file_header()

        # check if we can load all frames into RAM
        self.ram_available = virtual_memory().available

        # Get the number of frames in the file
        self.frames_in_file = int((self.file_size - self.file_header_length) / self.frame_size_in_bytes())
        logger.info('Found ' + str(self.frames_in_file) + ' frames in ' + self.filepath)

    def unpack(self, format_string, names, raw_data):

        output = {}

        if not self.file_valid:
            return

        try:
            logger.debug("Unpacking data in format: " + format_string)
            logger.debug("Expected header size in bytes: " + str(struct.calcsize(format_string)))
            data = struct.unpack(format_string, raw_data)

            for ind, d in enumerate(data):
                output[names[ind]] = d

        except struct.error as e:
            logger.error(e)

        return output

    def correct_flat_field(self, data):

        if not self.file_valid:
            return

        if self.field_correction is None:
            logger.info("Estimating flat field correction from images")

            data_array = None

            for i in range(0, self.frames_in_file):
            
                header, raw_data = self.read_frame(i)

                if data_array is None:
                    data_array = np.zeros((data.shape[0],data.shape[1],self.frames_in_file))
                
                data_array[:,:,i] = raw_data

            self.field_correction = np.median(data_array, 2)
            self.field_correction[self.field_correction < 1] = 1
            self.field_correction = self.field_correction.astype('float') / np.min(self.field_correction)

            # remove color from the correction
            h, w = self.field_correction.shape
            self.field_correction = cv2.resize(self.field_correction, (int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
            self.field_correction = cv2.resize(self.field_correction, (w, h), interpolation=cv2.INTER_LINEAR)
            self.background = np.median(data_array, 2) / self.field_correction
        
        # apply the correction
        data = data / self.field_correction
        data = data - self.background
        data[data<0.0] = 0.0

        if self.bpp == 2:
            data = data.astype('uint16')
        else:
            data = data.astype('uint8')

        return data
            

    
    def export_8bit_jpegs(self, output_path=None, bayer_pattern=cv2.COLOR_BayerRG2RGB, flat_field=True, gamma=1.0, thumbnails=True):
        
        if not self.file_valid:
            return
        
        file_info = {}
        file_info['file_header'] = self.file_header
        file_info['frame_headers'] = []

        self.jpeg_exports = []
        self.thumbnail_exports = []

        for i in range(0, self.frames_in_file):
            
            image_item = {}

            header, data = self.read_frame(i)

            if flat_field:
                data = self.correct_flat_field(data)
                
            if bayer_pattern:
                logger.debug("Converting color...")
                data = cv2.cvtColor(data,bayer_pattern) # RGB needed to get RGB format in opencv

            # Clip any negative values
            if self.bpp == 2:
                data = data/256/256
            else:
                data = data/256
            
            data = data**(gamma)
            data=data*255
            data[data>255] = 255
            data = np.uint8(data)


            filename = os.path.basename(self.filepath)[:-4] + '-' + str(header['frame_id']) + '-' + str(header['system_timestamp']) +'.jpeg'
            if thumbnails:
                thumbnail_name = os.path.basename(self.filepath)[:-4] + '-' + str(header['frame_id']) + '-' + str(header['system_timestamp']) +'_thumb.jpeg'

            timestamp = int(header['system_timestamp']/1000000)
            subdir = str(int(timestamp/864))

            if output_path is None:
                if self.output_path is None:
                    output_path = os.path.dirname(self.filepath)
                    output_subdir = os.path.join(os.path.dirname(self.filepath), subdir)
                else:
                    output_path = self.output_path
                    output_subdir = os.path.join(self.output_path, subdir)
            else:
                output_subdir = os.path.join(output_path, subdir)

            if not os.path.exists(output_subdir):
                try:
                    os.makedirs(output_subdir)
                except FileExistsError as e:
                    logger.warning(e)
                    pass



            jpeg_path = os.path.join(output_subdir, filename)
            self.jpeg_exports.append(jpeg_path)
            cv2.imwrite(jpeg_path, data)
            logger.info(jpeg_path)

            image_item['src'] = os.path.basename(os.path.normpath(output_path)) + '/' + subdir + '/' + filename
            image_item['fullWidth'] = data.shape[1]
            image_item['fullHeight'] = data.shape[0]

            timestamp = datetime.datetime.fromtimestamp(float(header['system_timestamp'])/1000000)
            image_item['timestring'] = timestamp.isoformat()

            if thumbnails:
                thumbnail_size = (int(data.shape[1]/10), int(data.shape[0]/10))
                jpeg_thumbnail_path = os.path.join(output_subdir, thumbnail_name)
                cv2.imwrite(jpeg_thumbnail_path, cv2.resize(data, thumbnail_size, interpolation=cv2.INTER_LINEAR))
                self.thumbnail_exports.append(jpeg_thumbnail_path)
                image_item['thumbnailWidth'] = thumbnail_size[0]
                image_item['thumbnailHeight'] = thumbnail_size[1]
                image_item['thumbnail'] = os.path.basename(os.path.normpath(output_path)) + '/' + subdir + '/' + thumbnail_name
            else:
                image_item['thumbnailWidth'] = 0
                image_item['thumbnailHeight'] = 0
                image_item['thumbnail'] = ''

            self.image_items.append(image_item)

    def export_as_tiff(self, output_path=None, bayer_pattern=cv2.COLOR_BayerRG2BGR, flat_field=True):
        
        if not self.file_valid:
            return

        file_info = {}
        file_info['file_header'] = self.file_header
        file_info['frame_headers'] = []

        self.tiff_exports = []

        if output_path is None:
            if self.output_path is None:
                tiff_path = self.filepath[:-4] + '.tif'
                json_path = self.filepath[:-4] + '.json'
            else:
                tiff_path = os.path.join(self.output_path, os.path.basename(self.filepath)[:-4] + '.tif')
                json_path = os.path.join(self.output_path, os.path.basename(self.filepath)[:-4] + '.json')
        else:
            tiff_path = os.path.join(output_path, os.path.basename(self.filepath)[:-4]) + '.tif'
            json_path = os.path.join(output_path, os.path.basename(self.filepath)[:-4]) + '.json'

        self.tiff_exports.append(tiff_path)

        logger.info("Exporting headers to " + json_path)
        logger.info("Exporting frames to " + tiff_path)

        with TiffWriter(tiff_path, append=True) as tif:

            for i in range(0, self.frames_in_file):
                
                header, data = self.read_frame(i)

                if flat_field:
                    data = self.correct_flat_field(data)

                if bayer_pattern:
                    logger.debug("Converting color...")
                    data = cv2.cvtColor(data,bayer_pattern) # BGR needed to get RGB format in TiffWriter


                timestamp = float(header['system_timestamp'])
                dt = datetime.datetime.fromtimestamp(timestamp/1000000.0)
                file_info['frame_headers'].append(header)
                frame_header_string = json.dumps(header)
                xtag = (65000, 's', 0, frame_header_string, False)
                tif.save(
                    data,
                    description=json.dumps(file_info),
                    datetime=dt,
                    extratags=[xtag]
                )

        with open(json_path, "w") as f:
            json.dump(file_info, f, indent=4, sort_keys=True)

    def read_file_header(self):

        if self.file_valid:

            # read the rest of the header
            self.file_handle.seek(0)
            res = self.file_handle.read(self.file_header_length)
            logger.debug("Read file header of length: " + str(self.file_header_length))
            self.file_header = self.unpack(self.file_header_format, self.file_header_params, res)

            self.file_fmt = self.file_header['file_type']
            self.img_height = int(self.file_header['image_height'])
            self.img_width = int(self.file_header['image_width'])
            self.bpp = int(self.file_header['bits_per_pixel'] / 8) # in bytes-per-pixel from bits-per-pixel

            logger.debug(self.file_header)

    def frame_size_in_bytes(self):

        if self.file_valid:
            return self.frame_header_size + self.bpp * self.img_width * self.img_height
        else:
            return 0

    def frame_pixels(self):
        
        if self.file_valid:
            return self.img_width*self.img_height
        else:
            return 0

    def read_frame(self, index):

        if not self.file_valid:
            return

        frame_offset = self.file_header_length + index * self.frame_size_in_bytes()
        frame_pixels = self.frame_pixels()

        if self.file_valid:

            # seek to the start of the frame
            self.file_handle.seek(frame_offset, 0)

            # read the header
            frame_header = self.unpack(
                self.frame_header_format,
                self.frame_header_params,
                self.file_handle.read(self.frame_header_size)
            )

            # read the frame pixels
            if self.bpp == 1:
                res = np.fromfile(self.file_handle, dtype='uint8', count=frame_pixels)
            else:
                res = np.fromfile(self.file_handle, dtype='uint16', count=frame_pixels)
            if len(res) <= 0:
                logger.error('Error reading frame ' + str(index) + " from file " + self.filepath)
                return None
            else:
                # reshape into image
                image = np.reshape(res, (self.img_height, self.img_width))
                return frame_header, image
