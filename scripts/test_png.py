import cv2
import sys
import time
import numpy as np
#import imageio
#from scipy.fftpack import dct

def sub_background(img, dtype=np.uint16):

    #bg = cv2.GaussianBlur(img,(21,21),0)

    img = img.astype(float)
    #img = img - bg
    m = np.mean(img)
    s = np.std(img)
    #print([m,s])
    img = img - (m + s)
    img[img < 0] = 0

    return img.astype(dtype)

output_fmt = 'png'

img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

img_c = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)

img_c[:,:,0] = img_c[:,:,0]*1.7
img_c[:,:,2] = img_c[:,:,2]*2.4

cv2.imwrite(sys.argv[1] + '.ds-color.png', img_c/16)

start_time = time.time()
#print(time.time())

img_r = sub_background(img[0:-1:2,0:-1:2])
img_g1 = sub_background(img[1::2,0:-1:2])
img_g2 = sub_background(img[0:-1:2,1::2])
img_b = sub_background(img[1::2,1::2])

img_rgba = np.zeros((img_r.shape[0],img_r.shape[1],4),dtype=np.uint16)

img_rgba[:,:,0] = img_r
img_rgba[:,:,1] = img_g1
img_rgba[:,:,2] = img_b
img_rgba[:,:,3] = img_g2

cv2.imwrite(sys.argv[1] + '.ds-rgba-L.' + output_fmt, np.bitwise_and(img_rgba, 255).astype(np.uint8))
cv2.imwrite(sys.argv[1] + '.ds-rgba-H.' + output_fmt, (img_rgba/256).astype(np.uint8))

print('elasped time: ' + str(time.time() - start_time))
print('reconsutrcting image...')

img_l = cv2.imread(sys.argv[1] + '.ds-rgba-L.' + output_fmt, cv2.IMREAD_UNCHANGED)
img_h = cv2.imread(sys.argv[1] + '.ds-rgba-H.' + output_fmt, cv2.IMREAD_UNCHANGED)

img_full = img_l.astype(np.uint16) + 256*img_h.astype(np.uint16)

img_out = np.zeros((img_full.shape[0]*2, img_full.shape[1]*2), dtype=np.uint16)

img_out[0:-1:2,0:-1:2] = img_full[:,:,0]
img_out[1::2,0:-1:2] = img_full[:,:,1]
img_out[0:-1:2,1::2] = img_full[:,:,2]
img_out[1::2,1::2] = img_full[:,:,3]

cv2.imwrite(sys.argv[1] + '.reconst.tif', img_out)

#print(time.time()-start_time)
