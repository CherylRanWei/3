from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte



print '************** Problem 1 **************'
img = cv2.imread('3_1.bmp')
print img.shape
cv2.imshow('origianl', img)

X_img = img.astype(np.float) 

for i in range(X_img.shape[0]):
    for j in range(X_img.shape[1]):
        X_img[i,j,2]  = X_img[i,j,2]*0.7
        X_img[i,j,1]  = X_img[i,j,1]*1.15
        X_img[i,j,0]  = X_img[i,j,0]*0.95

        if X_img[i,j,1] > 255:
            X_img[i,j,1] = 255
        if X_img[i,j,2] > 255:
            X_img[i,j,2] = 255
        if X_img[i,j,0] > 255:
            X_img[i,j,0] = 255

        if X_img[i,j,1] < 0:
            X_img[i,j,1] = 0
        if X_img[i,j,2] < 0:
            X_img[i,j,2] = 0
        if X_img[i,j,0] < 0:
            X_img[i,j,0] = 0 
Y_img = X_img.astype(np.uint8)


hsv_img = cv2.cvtColor(Y_img, cv2.COLOR_BGR2HSV)
new_img = hsv_img.astype(np.float) 

for i in range(new_img.shape[0]):
    for j in range(new_img.shape[1]):
        new_img[i,j,2] += 50.0
        if new_img[i,j,2] > 255.0:
            new_img[i,j,2] = 255.0

uint_img = new_img.astype(np.uint8)      
back_img = cv2.cvtColor(uint_img, cv2.COLOR_HSV2BGR)


cv2.imshow('nature', back_img)
cv2.imwrite('Nature.png', back_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
