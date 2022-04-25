import sys
import h5py
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
       
    
class LRAUVData:
    
    def __init__(self, hdf5_path,
                 fields_to_export= [
                     'WetLabsBB2FL', 
                     'depth',   
                     'latitude', 
                     'longitude', 
                     'mass_concentration_of_chlorophyll_in_sea_water',  
                     'platform_speed_wrt_sea_water', 
                     'sea_water_density', 
                     'sea_water_electrical_conductivity', 
                     'sea_water_pressure', 
                     'sea_water_salinity', 
                     'sea_water_temperature'
                    ]
                 ):
        
        self.fields_to_export = fields_to_export
        self.hdf5_path = hdf5_path
        
    
    def extract(self):
        
        logger.info('Loading file: ' + self.hdf5_path)
        hdf = h5py.File(self.hdf5_path,'r')
        
        tmp_dict = {}
        # will fill in timestamp latter with updated values
        tmp_dict['timestamp'] = np.ndarray.flatten(hdf.get('depth').get('time')[:])
        
        # Get the time index to use for interp
        tmp_dict['datenum'] = np.ndarray.flatten(hdf.get('depth').get('time')[:])
        
        logger.info('Extracting fields:')
        for f in self.fields_to_export:
            logger.info('Extracting: ' + f)
            if f == 'WetLabsBB2FL':
                tmp_dict['VolumeScatCoeff117deg470nm'] = np.interp(tmp_dict['datenum'],
                                                                   np.ndarray.flatten(hdf.get(f).get('VolumeScatCoeff117deg470nm').get('time')[:]),
                                                                   np.ndarray.flatten(hdf.get(f).get('VolumeScatCoeff117deg470nm').get('value')[:])
                )
                tmp_dict['VolumeScatCoeff117deg650nm'] = np.interp(tmp_dict['datenum'], 
                                                                   np.ndarray.flatten(hdf.get(f).get('VolumeScatCoeff117deg650nm').get('time')[:]),
                                                                   np.ndarray.flatten(hdf.get(f).get('VolumeScatCoeff117deg650nm').get('value')[:])
                )
            else:
                tmp_dict[f] = np.interp(tmp_dict['datenum'], np.ndarray.flatten(hdf.get(f).get('time')[:]), np.ndarray.flatten(hdf.get(f).get('value')[:]))
        
        logger.info('Convering to DataFrame...')
        df = pd.DataFrame.from_dict(tmp_dict)
        
        logger.info('Converting timestamps to unixtime')
        df['timestamp'] = df['datenum'].apply(lambda matlab_datenum: datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366))
        
        logger.info('Saving CSV...')
        df.to_csv(self.hdf5_path[0:-4] + '.csv')
        
        
if __name__=="__main__":
    
    lr = LRAUVData(sys.argv[1])
    lr.extract()