import os
import sys
import glob
import shutil
import datetime
from loguru import logger

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

class ROIInfo:
    """Extract roi info from filepath
    """

    def __init__(self, filepath):
        # low_mag_cam-1656948837612122-11054092130728-110363-018-688-1406-28-24_rawcolor.jpeg
        self.filepath = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)
        self.tokens = self.filename.split('-')
        if len(self.tokens) != 9:
            logger.warning(self.filename + ' does not match expected format, skipping...')
            return
        else:
            self.camera = self.tokens[0]
            self.timestamp = datetime.datetime.fromtimestamp(float(self.tokens[1])/1000000)
            self.timestring = self.timestamp.isoformat()
            # high_mag_cam-1649701840911317-21594687728-3-023-498-914-452-44_binary
            self.frame_number = int(self.tokens[3])
            self.left = int(self.tokens[5])
            self.top = int(self.tokens[6])
            self.width = int(self.tokens[7])
            self.height = int(self.tokens[8].split('_')[0]) # remove extra cruft on the right end of last token

    def get_area(self):
        return self.width*self.height

if __name__=="__main__":

    if len(sys.argv) < 2:
        logger.error("Please provide absolute path to directory of ROI subsirectories as the first argument")

    roi_path = sys.argv[1]

    total_rois = []

    roi_dirs = glob.glob(os.path.join(roi_path,'[0-9]*'))

    # get all of the rois
    for subdir in roi_dirs:
        rois = glob.glob(os.path.join(subdir,'*_rawcolor.jpeg'))
        logger.info('Loading subdir: ' + subdir)
        for roi in rois:
            total_rois.append(ROIInfo(roi))

    # make dir to hold removed duplicates for review
    dup_dir = 'duplicates'
    if not os.path.exists(os.path.join(roi_path,dup_dir)):
        os.makedirs(os.path.join(roi_path,dup_dir))

    # Remove duplicates by searching within a frame number and detecting any ROIs that overlap and
    # removing all of the overlapping ROIs but the largest one
    if len(total_rois) > 0:
        rois_in_frame = [total_rois[0]]
        rois_to_delete = []
        last_frame_number = total_rois[0].frame_number
        for roi in total_rois:
            if roi.frame_number == last_frame_number:
                rois_in_frame.append(roi)
            else:
                # create list of duplicates to delete
                logger.info('Processing frame ' + str(last_frame_number))
                for i, roi_1 in enumerate(rois_in_frame[:-1]):
                    for j, roi_2 in enumerate(rois_in_frame[i+1:]):
                        iou = bb_intersection_over_union(roi_1, roi_2)
                        if iou > 0.0:
                            if roi_1.get_area() >= roi_2.get_area():
                                if roi_2 not in rois_to_delete:
                                    rois_to_delete.append(roi_2)
                            else:
                                if roi_1 not in rois_to_delete:
                                    rois_to_delete.append(roi_1)

                # clear the roi lists and update frame number
                for roi in rois_to_delete:
                    matching_rois = glob.glob(os.path.join(roi.filepath, '*'+ '-'.join(roi.tokens[1:8]) +'*'))
                    for mroi in matching_rois:
                        logger.warning('Moving ' + roi.filename + ' to duplicate dir')
                        shutil.move(mroi, os.path.join(roi_path, dup_dir))
                
                # reset lists and frame number for next frame
                rois_in_frame = [roi]
                last_frame_number = roi.frame_number
                rois_to_delete = []


