from __future__ import division
import cv2
import numpy as np
import function 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte



print '************** Problem 2 **************'
'''
 Figure 3_2.jpg for implementation
'''

X = cv2.imread('3_2.jpg')
img = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)

'''
Origianl Histogram and CDF Graph for 3_2.jpg
'''
hist_val = function.histogram(img)
x_data, y_data = function.cdf(img)

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.bar(np.arange(256), hist_val)
plt.title('Histogram for OriginalValue Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Counts')
plt.grid(True)
plt.savefig('Histogram of Original 3_2.png')

f2 = plt.figure()
ax2 = f2.add_subplot(111)
plt.title('CDF of Original Value Channel')
ax2.plot(x_data, y_data, marker= '.', linestyle= 'none')
plt.grid(True)
plt.ylabel("CDF")
plt.savefig('CDF of Original 3_2.png')


'''
Linear Stretched Histogram and CDF Graph for 3_2.jpg
'''
ls_img = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
function.LinStrtch(ls_img, 157)   # linear stretching
ls_hist_val = function.histogram(ls_img)
ls_x_data, ls_y_data = function.cdf(ls_img)

f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.bar(np.arange(256), ls_hist_val)
plt.title('Histogram for Linear Stretched Value Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Counts')
plt.grid(True)
plt.savefig('Histogram of Linear Stretching 3_2.png')

f4 = plt.figure()
ax4 = f4.add_subplot(111)
plt.title('CDF of Linear Stretched Value Channel')
ax4.plot(ls_x_data, ls_y_data, marker= '.', linestyle= 'none')
plt.grid(True)
plt.ylabel("CDF")
plt.savefig('CDF of Linear Stretching 3_2.png')



'''
Histogram Equalization and CDF Graph for 3_2.jpg
'''
he_img = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
function.HistEq(he_img)  # Histogram Equalization
he_hist_val = function.histogram(he_img)
he_x_data, he_y_data = function.cdf(he_img)

f5 = plt.figure()
ax5 = f5.add_subplot(111)
ax5.bar(np.arange(256), he_hist_val)
plt.title('Histogram for Histogram Equalization Value Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Counts')
plt.grid(True)
plt.savefig('Histogram of Histogram Equalization 3_2.png')

f6 = plt.figure()
ax6 = f6.add_subplot(111)
plt.title('CDF of Histogram Equalization Value Channel')
ax6.plot(he_x_data, he_y_data, marker= '.', linestyle= 'none')
plt.grid(True)
plt.ylabel("CDF")
plt.savefig('CDF of Histogram Equalization 3_2.png')


'''
Histogram Specification and CDF Graph for 3_2.jpg
'''
mu = 50
sigma = 30
hs_img = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
function.HistSp(hs_img, mu, sigma)  # Histogram Specification
hs_hist_val = function.histogram(hs_img)
hs_x_data, hs_y_data = function.cdf(hs_img)

f7 = plt.figure()
ax7 = f7.add_subplot(111)
ax7.bar(np.arange(256), hs_hist_val)
plt.title('Histogram for Histogram Specification Value Channel')
plt.xlabel('Pixel Value')
plt.ylabel('Counts')
plt.grid(True)
plt.savefig('Histogram of Histogram Specification 3_2.png')

f8 = plt.figure()
ax8 = f8.add_subplot(111)
plt.title('CDF of Histogram Specification Value Channel')
ax8.plot(hs_x_data, hs_y_data, marker= '.', linestyle= 'none')
plt.grid(True)
plt.ylabel("CDF")
plt.savefig('CDF of Histogram Specification 3_2.png')


'''
Convert to RGB for Display and Save After Linear Stretching for 3_2.jpg
'''
ls_back_img = cv2.cvtColor(ls_img, cv2.COLOR_HSV2RGB)

plt.figure()
plt.axis("off")
plt.imshow(ls_back_img)
plt.savefig('Linear Stretching for 3_2.png')


'''
Convert to RGB for Display and Save After Histogram Equalization for 3_2.jpg
'''
he_back_img = cv2.cvtColor(he_img, cv2.COLOR_HSV2RGB)

plt.figure()
plt.axis("off")
plt.imshow(he_back_img)
plt.savefig('Histogram Equalization for 3_2.png')


'''
Convert to RGB for Display and Save After Specification for 3_2.jpg
'''
hs_back_img = cv2.cvtColor(hs_img, cv2.COLOR_HSV2RGB)

plt.figure()
plt.axis("off")
plt.imshow(hs_back_img)
plt.savefig('Specification for 3_2.png')



