import os
import sys
import glob

class ROIFileInfo:
    
    def __init__(self, delim='_', ext='.tif'):
        
        # define the required infomation in the file
        # name based on format spec
        self.info_tokens = [
            'deployment', 
            'sys_time',
            'camera',
            'frame_number',
            'roi_number',
            'roi_left',
            'roi_top', 
            'roi_front',
            'roi_width',
            'roi_height',
            'roi_extent',
        ]
        
        # Hold values for tokens here
        self.info_values = {}
        
        # Token delimeter and file extension
        self.delim = delim
        self.ext = ext
    
    def build_filename(self, output_path, delim='_', ext='tif'):
        
        output_name = ''
        
        if len(self.info_values) != len(self.info_tokens):
            raise ValueError('ROI info not populated. Populate values before calling this function.')
        
        for t in self.info_values:
            output_name += str(self.info_values[t]) + self.delim
        
        output_name += '.' + self.ext
        
        output_path = os.path.join(output_path, output_name)
        
        return output_path