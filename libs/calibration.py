import numpy as np
import cv2
import glob
import sys
import os
import json

# numpy to json helper
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# draws a grid on an image
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

# main function
if __name__=="__main__":

    if len(sys.argv) < 2:
        print('Please input path to images as first argument')
        exit()

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    board_shape = (10,10)
    image_path = sys.argv[1]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_shape[0]*board_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(image_path,'*.jpg'))
    
    # create output folder
    output_dir = os.path.join(sys.argv[1],'cal_output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    fname_list = []
    img_list = []
    gray_list = []

    # Detect the corners
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), 0, 0, cv2.INTER_AREA)
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img = cv2.bilateralFilter(img, 8, 25, 25)
        
        fname_list.append(fname)
        img_list.append(img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4,4))
        gray = clahe.apply(gray)
        #gray = self.contrast_boost*gray.astype(float) 
        #gray[gray > 255] = 255
        gray = gray.astype('uint8')
        
        gray_list.append(gray)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, board_shape,None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, board_shape, corners2,ret)
            cv2.imshow('img',cv2.resize(img,(int(img.shape[1]/2), int(img.shape[0]/2)), 0, 0, cv2.INTER_AREA))
            cv2.imwrite(os.path.join(output_dir,os.path.basename(fname)[:-4]+'_corners.jpg'),img)
            cv2.waitKey(100)

    # Do the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # show the RMS re-projection error
    print("RMS error: ", ret)

    # Save out the results
    if ret < 1.0:
        print('Calibration okay.')
        output = {}
        output['mtx'] = mtx
        output['dist'] = dist
        output['rvecs'] = rvecs
        output['tvecs'] = tvecs
        output['image_width'] = img.shape[1]
        output['image_height'] = img.shape[0]
        output['rms_error'] = ret
        with open(os.path.join(output_dir,'calibration.json'), 'w+') as outfile:
                json.dump(output, outfile,sort_keys=True, indent=4,cls=NumpyEncoder)

    # revise the calibration matrix
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # build the distortion corection mapping
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)

    # undistort the calibration images
    for i, img in enumerate(img_list):

        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(fname_list[i])[:-4] + '_remap.jpg'),dst)

    # invert the undistortion brute-force like (need some extra pixel room in the arrays)
    padding = 200
    mapx_inv = np.zeros((h+padding,w+padding),np.float32)
    mapy_inv = np.zeros((h+padding,w+padding),np.float32)
    for i in range(0,mapx.shape[0]):
        for j in range(0,mapx.shape[1]):
            mapx_inv[int(mapy[i,j]),int(mapx[i,j])] = j
            mapy_inv[int(mapy[i,j]),int(mapx[i,j])] = i
            
    # create a qualitative example of the distortion of a grid
    grid_img = 0*(img.copy())
    draw_grid(grid_img)
    dst = cv2.remap(grid_img,mapx_inv,mapy_inv,cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir,'distortion_grid.jpg'),dst[0:1080-1,0:1920-1])
        


