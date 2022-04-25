import os
import cv2
import datetime


def build_filename(output_path, roi_info, delim='_', ext='tif'):
    
    
    
    output_name = roi_info['platform'][0:4].upper()
    output_name += roi_info['deployment_number'] + delim
    output_name += roi_info['sys_time'] + delim
    output_name += roi_info['camera'] + delim
    output_name += roi_info['frame_number'] + delim
    output_name += roi_info['roi_number'] + delim
    output_name += roi_info['roi_left'] + delim
    output_name += roi_info['roi_top'] + delim
    output_name += roi_info['roi_front'] + delim
    output_name += roi_info['roi_width'] + delim
    output_name += roi_info['roi_height'] + delim
    output_name += roi_info['roi_extent']
    
    output_name += '.' + ext
    
    output_path = os.path.join(output_path, output_name)
    
    return output_path

def export_roi(output_path, 
               filename, 
               bb, 
               roi,
               frame_number,
               roi_number,
               platform='LRGA', 
               deployment='2',
               instrument='AyeRIS01',
               camera='AyeRISCAM00',
               ):
    
    # parse out the roi_info from filename and bounding box
    # Example: 2021-11-11-09-04-06.292372550-000000-
    #dto = datetime.datetime.strptime(filename[0:23] + "UTC",'%Y-%m-%d-%H-%M-%S.%f%Z')
    timestamp = list(filename[0:26])
    timestamp[10] = 'T'
    timestamp = "".join(timestamp).replace('-','') + "Z"
    roi_info = {}
    roi_info['platform'] = platform
    roi_info['deployment_number'] = deployment
    roi_info['sys_time'] = timestamp
    roi_info['instrument'] = instrument
    roi_info['camera'] = camera
    roi_info['frame_time'] = roi_info['sys_time']
    roi_info['frame_number'] = str(frame_number)
    roi_info['roi_number'] = str(roi_number)
    roi_info['roi_left'] = str(bb[1])
    roi_info['roi_top'] = str(bb[0])
    roi_info['roi_front'] = '0'
    roi_info['roi_width'] = str(bb[3] - bb[1])
    roi_info['roi_height'] = str(bb[2] - bb[0])
    roi_info['roi_extent'] = '0'
    
    output_path = build_filename(output_path, roi_info)
    
    print(output_path)
    
    cv2.imwrite(output_path, roi)