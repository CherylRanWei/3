from __future__ import division
import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_ubyte


def histogram(img):    # pdf
    count = np.zeros(256)
    for p_val in range(256):         # 0-255
        count[p_val] = np.sum((img[:,:,2] == p_val))
    return count

def cdf(img):
    data = histogram(img)
    data = data.astype(float)
    pdf = np.zeros(data.shape)
    cdf = np.zeros(data.shape)
    ttl = 0
    
    n = len(data)
    x = np.arange(n)
    
    for i in range(n):
        pdf[i] = data[i]/(img.shape[0]*img.shape[1])   # Normalize pdf
        
    for j in range(n):
        ttl = ttl + pdf[j]
        cdf[j] = ttl

    return x, cdf


def LinStrtch(img, inup):
    img[:,:,2] = img[:,:,2] * ((255)/(inup))
    

def HistEq(img):
    x_axis, cdf_val = cdf(img)
    
    s_k = np.uint8(cdf_val*255)
    img[:,:, 2] = s_k[img[:,:,2]]


def check(goal, array):
    diff = np.absolute(goal - array[:])
    idex = diff.index(np.amin(diff))
    return idex


def HistSp(img, mu, sigma):
    x = np.linspace(0, 255, 255)

    y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) *  (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))))

    summ = np.sum(y)
    cdf = np.zeros(y.shape)
    ttl = 0

    for i in range(len(y)):   #cdf
        ttl += y[i] 
        cdf[i]  = ttl

    cdf = cdf / summ          #normalized cdf [0,1]
    s_u = np.uint8(cdf*255)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,2] > 254:
                img[i,j,2] = 254
            img[i,j, 2] = s_u[img[i,j,2]]








    
