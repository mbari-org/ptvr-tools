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

from loguru import logger
from log_parser import DualMagLog


if __name__=="__main__":

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    with logger.catch():

        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()

        if len(sys.argv) != 2:
            logger.exception('Usage: python ptvr_to_rims.py [deployment_dir] ')
            logger.exception('Example: python python ptvr_to_rims.py X:\902004_Planktivore\Data\LRAUV\Ahi-2023-10-24')
            exit(1)
        
        data_dirs = glob.glob(os.path.join(sys.argv[1], '20*'))
        
        full_data_array = None
        
        for data_dir in data_dirs:
        
            log_file = glob.glob(os.path.join(sys.argv[1], '*.log'))
            if len(log_file) != 1:
                logger.error("Missing or multiple log files")
                exit(1)
            else:
                log_file = log_file[0]
        
            # # Process and export log file
            dml = DualMagLog(log_file)
            dml.parse_lines()
            
            if full_data_array == None:
                full_data_array = dml.all_data
            else:
                full_data_array = np.concatenate((full_data_array, dml.all_data))
        
        # Export the giant csv
        df = DataFrame(full_data_array, columns=dml.col_names)
        df.to_csv(os.path.join(sys.argv[1], os.path.basename(sys.argv[1]) + '.csv'), index=False)
        

        
        
