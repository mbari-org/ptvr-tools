# -*- coding: utf-8 -*-

import json
import datetime
import numpy as np

from scipy.signal import medfilt
from parse import parse
from loguru import logger
from pandas import DataFrame

__author__ = "Paul Roberts"
__copyright__ = "Copyright 2020 Guatek"
__credits__ = ["Guatek"]
__license__ = "MIT"
__maintainer__ = "Paul Roberts"
__email__ = "proberts@guatek.com"
__doc__ = '''

Text file parser tools.

@author: __author__
@status: __status__
@license: __license__
'''

class DualMagLog:
    """Read and parse Dual-Mag log files"""

    def __init__(self, filepath):

        self.lines = []
        self.data = np.array([])
        self.file_okay = False
        self.log_start_time = ''

        try:
            with open(filepath,"r") as f:
                self.lines = f.readlines()
            self.file_okay = True
        except IOError as ioe:
            logger.error(ioe)

    def parse_lines(self):
        """Parse lines from the log file in the format: 
        
        {date_time_string} ({elapsed_time}) [{thread_name}]{source}|{message}

        time, date, unixtime, and floating point data values are stored in a 2D numpy array
        """

        log_format = "{date_time_string} ({elapsed_time}) [{thread_name}]{source}|{message}"

        # container for log data
        ctrl_data_lines = []
        lowmag_data_lines = []
        highmag_data_lines = []

        # parse all avaialable lines
        for line in self.lines:
            parsed_line = parse(log_format,line)
            if parsed_line and "message" in parsed_line:

                try:
                    # extract time and date of entry
                    dtobj = datetime.datetime.strptime(parsed_line['date_time_string'], '%Y-%m-%d %H:%M:%S.%f')
                    line_timestamp = dtobj.timestamp()*1000000 # in microseconds since epoch
                    if self.log_start_time == '':
                        self.log_start_time = parsed_line['date_time_string']
                        
                except Exception as e:
                    print(e)
                    continue

                data_bits = parsed_line["message"].split(",")
                try:
                    # $DMCTRL
                    if (data_bits[0].lstrip(' ') == "$DMCTRL" and len(data_bits) == 20):

                        data_line = [line_timestamp,float(parsed_line["elapsed_time"][:-1])]
                        date_string = data_bits[1].split(" ")[0]
                        time_string = data_bits[1].split(" ")[1]
                        for i in range(0,3):
                            data_line.append(float(date_string.split("-")[i]))
                        for i in range(0,3):
                            data_line.append(float(time_string.split(":")[i]))
                        for i in range(2,17):
                            data_line.append(float(data_bits[i]))
                        
                        ctrl_data_lines.append(data_line) 

                    # $DMCAML
                    elif (data_bits[0].lstrip(' ') == "$DMCAML" and len(data_bits) >= 11):
                        # ROI,37,0,39.4,4.89,0.842,149,VID,0,0,
                        data_line = [line_timestamp]
                        for i in range(2,8):
                            data_line.append(float(data_bits[i]))
                        for i in range(9,11):
                            data_line.append(float(data_bits[i]))

                        lowmag_data_lines.append(data_line)

                    # $DMCAMH
                    elif (data_bits[0].lstrip(' ') == "$DMCAMH" and len(data_bits) >= 11):
                        # ROI,37,0,39.4,4.89,0.842,149,VID,0,0,
                        data_line = [line_timestamp]
                        for i in range(2,8):
                            data_line.append(float(data_bits[i]))
                        for i in range(9,11):
                            data_line.append(float(data_bits[i]))

                        highmag_data_lines.append(data_line)


                except ValueError as ve:
                    logger.error(ve)
                    pass
        
        self.ctrl_data = np.array(ctrl_data_lines)
        self.lowmag_data = np.array(lowmag_data_lines)
        self.highmag_data = np.array(highmag_data_lines)
        
        logger.info(self.ctrl_data.shape)
        logger.info(self.lowmag_data.shape)
        logger.info(self.highmag_data.shape)

        # collate data into one matrix with camera string strinstamp
        self.all_data = np.zeros((self.ctrl_data.shape[0],self.ctrl_data.shape[1] + 2*(self.lowmag_data.shape[1]-1)))
        
        self.all_data[:,0:self.ctrl_data.shape[1]] = self.ctrl_data

        for i in range(0,self.all_data.shape[0]):

            ind1 = np.argwhere(int(self.all_data[i,0]/1000000) == (self.lowmag_data[:,0]/1000000).astype('int'))
            if len(ind1) > 0:
                self.all_data[i,self.ctrl_data.shape[1]:(self.ctrl_data.shape[1]+self.lowmag_data.shape[1]-1)] = self.lowmag_data[ind1[0][0],1:]
            else:
                logger.info("Line: " + str(i) + ", could not find matching time: " + str(int(self.all_data[i,0]/1000000)))

            ind2 = np.argwhere(int(self.all_data[i,0]/1000000) == (self.highmag_data[:,0]/1000000).astype('int'))
            if len(ind2) > 0:
                self.all_data[i,(self.ctrl_data.shape[1]+self.lowmag_data.shape[1]-1):] = self.highmag_data[ind2[0][0],1:]
            else:
                logger.info("Line: " + str(i) + ", could not find matching time: " + str(int(self.all_data[i,0]/1000000)))

        # filter out spikes and single-line data errors
        for j in [14,15,16]:
            self.all_data[:,j] = medfilt(self.all_data[:,j],3)

        return self.all_data

    def export(self, filename):
        """Saves collated log data to csv file

        Args:
            filename (string): path to file
        """
        
        col_names = [
            'timestamp',
            'elapsed (s)',
            'year',
            'month',
            'day',
            'hours',
            'minute',
            'second',
            'temperature (C)',
            'pressure (kPa)',
            'humidity %',
            'input_voltage (V)',
            'system_power (W)',
            'strobe_power (W)',
            'ctd_cond (psu)',
            'ctd_temp (C)',
            'ctd_pressure (dBar)',
            'profile_state',
            'camera_on',
            'flash_type',
            'lowmag_duration (us)',
            'highmag_duration (us)',
            'frame_rate (Hz)',
            'lowmag_frames (#)',
            'lowmag_queued (#)',
            'lowmag_detections (#/s)',
            'lowmag_rois (#/s)',
            'lowmag_saved_rois (#/s)',
            'lowmag_average_area (pixels)',
            'lowmag_video_frame (#)',
            'lowmag_video_queued (#)',
            'highmag_frames (#)',
            'highmag_queued (#)',
            'highmag_detections (#/s)',
            'highmag_rois (#/s)',
            'highmag_saved_rois (#/s)',
            'highmag_average_area (pixels)',
            'highmag_video_frame (#)',
            'highmag_video_queued (#)',
        ]

        df = DataFrame(self.all_data, columns=col_names)
        df.to_csv(filename, index=False)

    






